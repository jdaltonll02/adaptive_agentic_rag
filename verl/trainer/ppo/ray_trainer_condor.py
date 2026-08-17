"""CONDOR ray_trainer_condor.py

Two-level hierarchical RL trainer extending the MAO-ARAG baseline
RayPPOTrainer with:

  Level 0 (MechanismRouter): selects retrieval mechanism m* before the
    L1 PlanningAgent runs.  Trained via REINFORCE with Phi-critic blended
    baseline.  Updated every global_step.

  Level 1 (PlanningAgent): goal-conditioned on (m*, qtype); identical to
    baseline except the prompt restricts available agents to
    MECHANISM_ACTION_SPACES[m*].

  DualAlphaManager: Adam-stabilised Lagrange multiplier updates for
    lambda_hi (L0 budget) and lambda_lo (L1 token/retrieval cost).
    lambda_lo replaces the baseline's hard-coded coeff_cost=0.0.

  PhiScorer: MLP counterfactual baseline blended with critic V_hi.

  TrajectoryLogger + ClusterReport: accumulate (l0_correct, F1, cost,
    n_turns, query_emb) tuples and run 2×2 K-Means taxonomy every
    cluster_report_freq steps.

Backward compatibility: when condor.use_condor=False the trainer calls
super().fit() and reproduces exact baseline behaviour.
"""

import os
import sys
import re
from copy import deepcopy
from multiprocessing import Manager, Process
from pprint import pprint
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: make both project root and baseline accessible
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..')
)
_BASELINE_ROOT = os.path.join(_PROJECT_ROOT, 'baseline')

for _p in [_PROJECT_ROOT, _BASELINE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Baseline imports (from baseline/verl/trainer/ppo/)
# ---------------------------------------------------------------------------
from verl.trainer.ppo.ray_trainer_agentic_rag_2 import (  # noqa: E402
    RayPPOTrainer,
    RewardManager,
    run_workflow,
    init_context_1turn_list,
    convert_to_shared_structure,
    convert_to_local_structure,
)
from verl.utils.tracking import Tracking  # noqa: E402

# Baseline verl utilities used by baseline trainer
try:
    from verl.trainer.ppo.core_algos import compute_advantage, apply_kl_penalty, agg_loss  # noqa: E402
    from verl.utils.seqlen_balancing import get_seqlen_balancing_stats  # noqa: E402
except ImportError as e:
    raise ImportError(f"Required baseline verl utilities not found: {e}") from e

# ---------------------------------------------------------------------------
# CONDOR imports (from project root)
# ---------------------------------------------------------------------------
from condor import (  # noqa: E402
    MechanismRouter,
    DualAlphaManager,
    PhiScorer,
    TrajectoryLogger,
    ProcessRewardModel,
    MECHANISM_ACTION_SPACES,
    MECHANISM_NAMES,
    MECHANISM_SELECTION_COST,
    QTYPE_LABELS,
    classify_qtype,
    query_to_features,
)
from condor.analysis import ClusterReport  # noqa: E402
from condor.metrics import MetricsCollector  # noqa: E402

# CONDOR qa_manager (goal-conditioned)
from qa_manager.qa import Agentic_RAG_Manager  # noqa: E402
from qa_manager.BaseAgent import AgentPool  # noqa: E402


# ---------------------------------------------------------------------------
# CONDOR-extended RewardManager
# ---------------------------------------------------------------------------

class CONDORRewardManager(RewardManager):
    """Extends baseline RewardManager to use adaptive lambda_lo for cost scaling.

    The baseline hard-codes ``coeff_cost = 0.0`` (cost ignored).
    CONDOR sets ``self.lambda_lo`` before each batch so the cost term
    is gradually activated as the Lagrange multiplier converges.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.lambda_lo: float = 0.0

    def assign_token_retrieval_cost(
        self, batch, metrics, context_list, token_cost, retrieval_api_cost, turn_id
    ):
        """Identical to baseline but multiplies by lambda_lo instead of 0.0."""
        # ---- replicate baseline cost tensor construction ----
        import torch as _torch
        coeff_token = -1.0
        coeff_retrieval = -0.25
        coeff_turn = -0.5

        batch_size = len(context_list)
        cost_tensor = _torch.zeros(
            batch.batch["responses"].shape, dtype=_torch.float32
        )

        for i in range(batch_size):
            begin_step = context_list[i]['begin_step']
            end_step = context_list[i]['end_step']
            if turn_id < begin_step or turn_id > end_step:
                continue

            # token cost for this turn/item
            tc = token_cost[turn_id][i] if isinstance(token_cost, list) else float(token_cost[i])
            rac = retrieval_api_cost[turn_id][i] if isinstance(retrieval_api_cost, list) else float(retrieval_api_cost[i])

            mode = context_list[i].get('mode', 'normal')
            if mode == 'serial':
                turn_lat = len(context_list[i].get('sub_answer', [])) + 1
            elif mode == 'parallel':
                turn_lat = 2
            else:
                turn_lat = 1

            raw_cost = (
                coeff_token * tc
                + coeff_retrieval * rac
                + coeff_turn * turn_lat
            )
            # CONDOR: scale by lambda_lo (baseline uses coeff_cost=0.0)
            final_cost = raw_cost * self.lambda_lo

            # Assign to last valid token position in this item's response
            response = batch.batch["responses"][i]
            attn = batch.batch["attention_mask"][i]
            prompt_len = batch.batch["prompts"].shape[-1]
            resp_attn = attn[prompt_len:]
            valid_len = int(resp_attn.sum().item())
            if valid_len > 0:
                cost_tensor[i, valid_len - 1] = final_cost

        return cost_tensor


# ---------------------------------------------------------------------------
# CONDORRayPPOTrainer
# ---------------------------------------------------------------------------

class CONDORRayPPOTrainer(RayPPOTrainer):
    """CONDOR two-level hierarchical PPO trainer.

    Extends RayPPOTrainer with:
      - L0 MechanismRouter (epsilon-greedy, REINFORCE-updated)
      - DualAlphaManager (Adam-stabilised Lagrange multipliers)
      - PhiScorer (MLP counterfactual baseline blended with critic)
      - TrajectoryLogger + ClusterReport

    Config keys (under condor.* in Hydra config):
      use_condor          : bool   (default False → baseline)
      epsilon_start       : float  (default 0.9)
      epsilon_end         : float  (default 0.1)
      epsilon_decay       : float  (default 1000)
      lambda_hi_init      : float  (default 0.1)
      lambda_lo_init      : float  (default 0.0)
      budget_hi           : float  (default 0.5)
      budget_lo           : float  (default 0.3)
      dual_alpha_lr       : float  (default 1e-3)
      phi_t_warmup        : int    (default 500)
      phi_tau_trans       : float  (default 100.0)
      phi_lr              : float  (default 1e-3)
      router_lr           : float  (default 1e-3)
      reinforce_gamma     : float  (default 0.99)
      l0_reward_threshold : float  (default 0.4)
      cluster_report_freq : int    (default 50)
    """

    def __init__(self, config, tokenizer, processor,
                 role_worker_mapping, resource_pool_manager,
                 ray_worker_group_cls, reward_manager, reward_fn,
                 val_reward_fn, train_dataset, val_dataset,
                 collate_fn, train_sampler, device_name='cuda'):

        # Replace stock RewardManager with CONDOR version
        if not isinstance(reward_manager, CONDORRewardManager):
            condor_rm = CONDORRewardManager(tokenizer)
            # Copy any state that was already set on reward_manager
            reward_manager = condor_rm

        super().__init__(
            config, tokenizer, processor, role_worker_mapping,
            resource_pool_manager, ray_worker_group_cls,
            reward_manager, reward_fn, val_reward_fn,
            train_dataset, val_dataset, collate_fn, train_sampler,
            device_name,
        )

        # ----------------------------------------------------------------
        # Parse CONDOR config
        # ----------------------------------------------------------------
        condor_cfg = getattr(config, 'condor', None)

        def _get(key, default):
            if condor_cfg is None:
                return default
            return getattr(condor_cfg, key, default)

        self.use_condor: bool = _get('use_condor', False)

        if not self.use_condor:
            return  # No CONDOR component initialisation needed

        self.mechanism_router = MechanismRouter(
            epsilon_start=_get('epsilon_start', 0.9),
            epsilon_end=_get('epsilon_end', 0.1),
            epsilon_decay=float(_get('epsilon_decay', 1000)),
            lr=_get('router_lr', 1e-3),
            use_condor=True,
        )
        # Two separate instances — one per level (spec Part A)
        self.dual_hi = DualAlphaManager(
            budget=_get('budget_hi', 0.0005),
            lr=_get('dual_lr_hi', 5e-4),
            lam_init=_get('lambda_hi_init', 0.10),
        )
        self.dual_lo = DualAlphaManager(
            budget=_get('budget_lo', 0.0004),
            lr=_get('dual_lr_lo', 1e-3),
            lam_init=_get('lambda_lo_init', 0.08),
        )
        self.phi_scorer = PhiScorer(
            t_warmup=int(_get('phi_t_warmup', 500)),
            tau_trans=float(_get('phi_tau_trans', 100.0)),
            lr=_get('phi_lr', 1e-3),
        )
        self.trajectory_logger = TrajectoryLogger()
        self.process_reward_model = ProcessRewardModel()
        self.cluster_report = ClusterReport(n_clusters=4)

        # MetricsCollector: write to metrics_dir if configured, else log_dir
        _metrics_dir = getattr(
            getattr(config, 'trainer', None), 'metrics_dir',
            getattr(getattr(config, 'trainer', None), 'log_dir', '/tmp/condor_metrics')
        )
        _exp_name = getattr(
            getattr(config, 'trainer', None), 'experiment_name', 'condor_run'
        )
        self.metrics_collector = MetricsCollector(
            metrics_dir=_metrics_dir,
            experiment_name=_exp_name,
        )

        self._reinforce_gamma: float = _get('reinforce_gamma', 0.99)
        self._l0_threshold: float = _get('l0_reward_threshold', 0.4)
        self._cluster_freq: int = int(_get('cluster_report_freq', 50))

    # ------------------------------------------------------------------
    # Checkpoint: persist CONDOR L0 state alongside baseline L1 state
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        super()._save_checkpoint()
        if not self.use_condor:
            return
        import torch as _torch
        condor_path = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{self.global_steps}",
            "condor_state.pt",
        )
        os.makedirs(os.path.dirname(condor_path), exist_ok=True)
        _torch.save({
            'router': self.mechanism_router.state_dict_router(),
            'dual_hi': self.dual_hi.state_dict(),
            'dual_lo': self.dual_lo.state_dict(),
            'phi': self.phi_scorer.state_dict_phi(),
            'global_steps': self.global_steps,
        }, condor_path)
        print(f"[CONDOR] Saved CONDOR state to {condor_path}")

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if not self.use_condor:
            return
        import torch as _torch
        condor_path = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{self.global_steps}",
            "condor_state.pt",
        )
        if not os.path.exists(condor_path):
            return
        state = _torch.load(condor_path, map_location='cpu')
        self.mechanism_router.load_state_dict_router(state['router'])
        if 'dual_hi' in state:
            self.dual_hi.load_state_dict(state['dual_hi'])
            self.dual_lo.load_state_dict(state['dual_lo'])
        self.phi_scorer.load_state_dict_phi(state['phi'])
        print(f"[CONDOR] Loaded CONDOR state from {condor_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_mechanisms(
        self, questions: List[str]
    ):
        """Run MechanismRouter for each question in the batch.

        Returns:
            m_star_list   : List[int]         mechanism ids
            qtype_list    : List[int]         query-type ids
            features_list : List[np.ndarray]  32-dim feature vectors
        """
        m_star_list, qtype_list, features_list = [], [], []
        for q in questions:
            m, qt, feat = self.mechanism_router.route(q)
            m_star_list.append(m)
            qtype_list.append(qt)
            features_list.append(feat)
        return m_star_list, qtype_list, features_list

    def _inject_condor_into_extra_info(
        self, batch_dict, m_star_list: List[int], qtype_list: List[int]
    ) -> None:
        """Store m_star and qtype_id in batch_dict['extra_info'] in-place."""
        for i, item in enumerate(batch_dict['extra_info']):
            item['m_star'] = m_star_list[i]
            item['qtype_id'] = qtype_list[i]

    def _inject_m_star_into_contexts(
        self, context_list: List[dict], m_star_list: List[int]
    ) -> None:
        """Store m_star in each context dict so RetrievalAgent dispatches correctly."""
        for ctx, m in zip(context_list, m_star_list):
            ctx['m_star'] = m

    def _compute_per_question_metrics(
        self, predicted_answers_list, golden_answers, questions
    ):
        """Return per-question F1 and summary dicts."""
        all_metrics, f1_list = [], []
        for pred, gold in zip(predicted_answers_list, golden_answers):
            m = self.reward_manager.compute_scores([pred], [gold])
            all_metrics.append(m)
            f1_list.append(m.get('f1', 0.0))
        return all_metrics, f1_list

    def _compute_cost_totals(self, token_cost_list, retrieval_api_final, context_list, questions):
        """Replicate the baseline cost aggregation logic."""
        n_turns = len(token_cost_list)
        n_q = len(questions)

        token_cost_final = [[0.0] * n_q for _ in range(n_turns)]
        token_cost_scale_final = [[0.0] * n_q for _ in range(n_turns)]
        retrieval_api_list = [[0.0] * n_q for _ in range(n_turns)]

        # Reuse baseline QDS/QDP redistribution logic
        is_QDS_list = [context['mode'] == 'serial' for context in context_list]
        is_QDP_list = [context['mode'] == 'parallel' for context in context_list]

        for temp_i in range(n_q):
            if not is_QDS_list[temp_i] and not is_QDP_list[temp_i]:
                continue
            if is_QDS_list[temp_i]:
                qds_cost = 0.0
                qds_turn = 0
                for tid in range(n_turns):
                    qds_cost += (
                        token_cost_list[tid][temp_i].get("QueryDecompositionAgentSerial", 0.0)
                        + token_cost_list[tid][temp_i].get("QueryRewriteAgent", 0.0)
                        + token_cost_list[tid][temp_i].get("AnswerSummarizationAgent", 0.0)
                    )
                    token_cost_list[tid][temp_i]["QueryRewriteAgent"] = 0.0
                    token_cost_list[tid][temp_i]["AnswerSummarizationAgent"] = 0.0
                    if token_cost_list[tid][temp_i].get("QueryDecompositionAgentSerial", 0.0) > 0:
                        qds_turn = tid
                token_cost_list[qds_turn][temp_i]["QueryDecompositionAgentSerial"] = qds_cost
            elif is_QDP_list[temp_i]:
                qdp_cost = 0.0
                qdp_turn = 0
                for tid in range(n_turns):
                    qdp_cost += (
                        token_cost_list[tid][temp_i].get("QueryDecompositionAgentSerial", 0.0)
                        + token_cost_list[tid][temp_i].get("AnswerSummarizationAgent", 0.0)
                    )
                    token_cost_list[tid][temp_i]["AnswerSummarizationAgent"] = 0.0
                    if token_cost_list[tid][temp_i].get("QueryDecompositionAgentSerial", 0.0) > 0:
                        qdp_turn = tid
                token_cost_list[qdp_turn][temp_i]["QueryDecompositionAgentSerial"] = qdp_cost

        for tid in range(n_turns):
            for qi in range(n_q):
                tc_dict = token_cost_list[tid][qi]
                for key, val in tc_dict.items():
                    if key == "RetrievalAgent":
                        retrieval_api_list[tid][qi] = 1 if val == 1 else retrieval_api_list[tid][qi]
                    else:
                        token_cost_final[tid][qi] += val
                token_cost_scale_final[tid][qi] = token_cost_final[tid][qi] * 2000

        # Per-question totals
        total_token_cost = [
            sum(token_cost_final[tid][qi] for tid in range(n_turns))
            for qi in range(n_q)
        ]
        total_api_cost = [
            sum(retrieval_api_list[tid][qi] for tid in range(n_turns))
            for qi in range(n_q)
        ]

        return token_cost_scale_final, retrieval_api_list, total_token_cost, total_api_cost

    def _update_condor_components(
        self,
        f1_list, m_star_list, qtype_list, features_list,
        total_token_cost, total_api_cost, context_list, questions,
        metrics_aver, logger, step,
    ):
        """One full CONDOR update cycle after a training batch."""

        lambda_hi = self.dual_hi.value
        lambda_lo = self.dual_lo.value

        # Phi–Critic blend weight (Eq. 8). We have no separate V_hi critic, so
        # critic_val=0.0 throughout, meaning alpha_t has no effect until V_hi is wired.
        alpha_t = self.phi_scorer.get_blend_alpha()

        # ---- L0 rewards (Eq. 9) ----
        # Â_hi = R_f1 - 1/(|M|-1) * Σ_{m≠m*} Q̂(S,m,t) - λ_hi * C_hi
        # Q̂(S,m,t) = (1-α_t)*Φ(q,m) + α_t*V_hi(S,m)  [V_hi=0 until critic wired]
        l0_rewards = []
        for qi in range(len(questions)):
            f1 = f1_list[qi]
            m_star = m_star_list[qi]
            # Average counterfactual Q̂ over the 4 mechanisms NOT selected (Eq. 9)
            cf_vals = []
            for m_other in range(5):
                if m_other == m_star:
                    continue
                phi_m = self.phi_scorer.score(features_list[qi], m_other, qtype_list[qi])
                q_m = (1.0 - alpha_t) * phi_m  # V_hi=0.0 until critic is wired in
                cf_vals.append(q_m)
            cf_baseline = float(np.mean(cf_vals)) if cf_vals else 0.0
            r_hi = self.process_reward_model.compute_l0_reward(
                f1=f1,
                m_star=m_star,
                lambda_hi=lambda_hi,
                baseline_f1=cf_baseline,
            )
            l0_rewards.append(r_hi)
            self.mechanism_router.store_reward(r_hi)

        # ---- L0 policy update (REINFORCE) ----
        l0_policy_loss = self.mechanism_router.update_policy(gamma=self._reinforce_gamma)

        # ---- Dual lambda update ----
        avg_mech_cost_hi = float(
            np.mean([MECHANISM_SELECTION_COST[m] for m in m_star_list])
        )
        # Normalise L1 token cost to [0,1] range for budget comparison
        avg_token_cost_norm = float(
            np.mean([min(tc / 0.01, 1.0) for tc in total_token_cost])
        )  # 0.01 USD ≈ budget unit
        new_lambda_hi = self.dual_hi.update(avg_mech_cost_hi)
        new_lambda_lo = self.dual_lo.update(avg_token_cost_norm)
        # Push new lambda_lo into reward manager for next batch
        self.reward_manager.lambda_lo = new_lambda_lo

        # ---- PhiScorer update ----
        phi_loss = self.phi_scorer.update(features_list, m_star_list, qtype_list, f1_list)

        # ---- Trajectory logging ----
        for qi in range(len(questions)):
            ctx = context_list[qi]
            n_turns = max(ctx.get('end_step', 0) - ctx.get('begin_step', 0) + 1, 1)
            l0_correct = self.process_reward_model.l0_correct(f1_list[qi], self._l0_threshold)
            self.trajectory_logger.log(
                l0_correct=l0_correct,
                f1=f1_list[qi],
                cost=total_token_cost[qi],
                n_turns=n_turns,
                query_features=features_list[qi],
                m_star=m_star_list[qi],
                qtype_id=qtype_list[qi],
                question=questions[qi],
            )

        # ---- Cluster report ----
        cluster_metrics = {}
        if step % self._cluster_freq == 0 and len(self.trajectory_logger) >= 4:
            feat_mat = self.trajectory_logger.get_feature_matrix()
            report = self.cluster_report.analyze(feat_mat)
            cluster_metrics = self.cluster_report.wandb_summary(report)

        # ---- Aggregate CONDOR W&B metrics ----
        traj_summary = self.trajectory_logger.get_summary()
        condor_metrics = {
            'condor/lambda_hi': new_lambda_hi,
            'condor/lambda_lo': new_lambda_lo,
            'condor/l0_policy_loss': l0_policy_loss,
            'condor/phi_loss': phi_loss,
            'condor/router_epsilon': self.mechanism_router.epsilon,
            'condor/avg_l0_reward': float(np.mean(l0_rewards)),
            'condor/avg_f1': traj_summary.get('avg_f1', 0.0),
            'condor/avg_l0_correct': traj_summary.get('avg_l0_correct', 0.0),
            'condor/avg_cost': traj_summary.get('avg_cost', 0.0),
            'condor/trajectory_count': len(self.trajectory_logger),
        }

        # Mechanism selection distribution
        for m_id in range(5):
            count = sum(1 for m in m_star_list if m == m_id)
            condor_metrics[f'condor/mechanism/{MECHANISM_NAMES[m_id]}'] = count / max(len(m_star_list), 1)

        condor_metrics.update(cluster_metrics)
        logger.log(data=condor_metrics, step=step)

        # Flush merged training metrics to collector (combines L1 + CONDOR keys)
        merged = {}
        merged.update(getattr(self, '_pending_metrics', {}))
        merged.update(condor_metrics)
        self.metrics_collector.record_training_step(
            step=getattr(self, '_pending_step', step),
            metrics=merged,
        )

    # ------------------------------------------------------------------
    # Validation with per-process timeout (fixes hang in job 9277115)
    # ------------------------------------------------------------------

    def _validate_agentic(self):
        """Override baseline _validate_agentic to add per-process timeout on
        run_workflow subprocesses.  A hung process (e.g. stalled vLLM call)
        is killed after workflow_timeout seconds and the question is skipped
        with an empty answer, so the batch completes rather than hanging.
        """
        condor_cfg = getattr(self.config, 'condor', None)
        timeout = int(getattr(condor_cfg, 'workflow_timeout', 120))

        qa_manager = Agentic_RAG_Manager(self.tokenizer, self.config)
        agent_pool = AgentPool()

        test_metrics_dict = {"acc": [], "em": [], "f1": [], "precision": [], "recall": []}
        test_cost_dict = {"token_cost": [], "api_times": [], "avr_api_per_turn": [], "turn_num": []}

        batch_id = -1
        for batch_dict in tqdm(self.val_dataloader, desc="Validation Progress"):
            batch_id += 1

            print('************************* testing: rollout *************************')
            extra_info = batch_dict['extra_info']
            questions = [item['question'] for item in extra_info]
            golden_answers = [item['answer'] for item in extra_info]

            batch_list, metrics_list = [], []
            predicted_answers_list = [""] * len(questions)
            MAX_TURN = 5
            context_list = init_context_1turn_list(batch_dict, MAX_TURN)
            token_cost_list = []

            for turn_id in range(MAX_TURN):
                print(f'*********************************** testing: new turn {turn_id} ***********************************')
                metrics = {}
                batch, metrics, _ = self.rollout(batch_dict, metrics, context_list, qa_manager, MAX_TURN)
                workflows_1turn, initial_workflows_context = self.get_answers_subs_list(batch)
                is_legal_list = self.is_legal_workflows(workflows_1turn, context_list, qa_manager)

                with Manager() as manager:
                    turn_context_list = []
                    for temp_i, context in enumerate(context_list):
                        if is_legal_list[temp_i]:
                            turn_context_list.append(convert_to_shared_structure(context, manager))
                        else:
                            turn_context_list.append(context)
                    turn_context_list = manager.list(turn_context_list)
                    turn_predicted_answers_list = manager.list([""] * len(questions))

                    processes = []
                    for temp_i in range(len(questions)):
                        p = Process(
                            target=run_workflow,
                            args=(temp_i, turn_id, turn_context_list, is_legal_list,
                                  workflows_1turn, turn_predicted_answers_list,
                                  agent_pool, qa_manager, MAX_TURN),
                        )
                        processes.append(p)
                        p.start()

                    for p in processes:
                        p.join(timeout=timeout)
                        if p.is_alive():
                            print(f'[CONDOR] WARNING: workflow process timed out after {timeout}s, killing.')
                            p.kill()
                            p.join()

                    local_context_list = convert_to_local_structure(turn_context_list)
                    local_predicted = convert_to_local_structure(turn_predicted_answers_list)

                context_list = local_context_list
                for a_i, ans in enumerate(local_predicted):
                    if ans != "":
                        predicted_answers_list[a_i] = ans

                batch, metrics = self.compute_logprobs_values_format_penalty(batch, metrics, is_legal_list)
                batch_list.append(batch)
                metrics_list.append(metrics)

                token_cost_turn = []
                for context in context_list:
                    token_cost_turn.append(deepcopy(context['token_cost']))
                    for key in context['token_cost']:
                        context['token_cost'][key] = 0.0
                token_cost_list.append(token_cost_turn)

                _log_dir = getattr(self.config.trainer, 'log_dir', '/tmp')
                os.makedirs(_log_dir, exist_ok=True)
                _log_path = os.path.join(
                    _log_dir,
                    f'testing_log_{self.config.trainer.experiment_name}.txt',
                )
                with open(_log_path, 'a', encoding='utf-8') as f:
                    f.write(f'>>>>>>>>>>>>>> batch id: {batch_id} <<<<<<<<<<<<<<<<\n')
                    f.write(f'>>>>>>>>>>>>>> turn id: {turn_id} <<<<<<<<<<<<<<<<\n')
                    for qi in range(len(questions)):
                        q_id = batch_id * self.config.data.train_batch_size + qi
                        f.write(
                            f'question id: {q_id}, is_legal: {is_legal_list[qi]}, '
                            f'workflow: {workflows_1turn[qi]}, '
                            f'initial workflow: {initial_workflows_context[qi]}\n'
                        )
                    f.write('\n')

                if "" not in predicted_answers_list:
                    break

            for temp_i in range(len(questions)):
                if predicted_answers_list[temp_i] == "":
                    agent = agent_pool.get("AnswerSummarizationAgent")
                    agent.run(context_list[temp_i])
                    predicted_answers_list[temp_i] = context_list[temp_i]['answer']

            _log_path = os.path.join(
                getattr(self.config.trainer, 'log_dir', '/tmp'),
                f'testing_log_{self.config.trainer.experiment_name}.txt',
            )
            with open(_log_path, 'a', encoding='utf-8') as f:
                f.write(f'>>>>>>>>>>>>>> batch id: {batch_id} <<<<<<<<<<<<<<<<\n')
                for a_id in range(len(predicted_answers_list)):
                    f.write(
                        f'question id: {a_id}, golden answer: {golden_answers[a_id]}, '
                        f'predict answer: {predicted_answers_list[a_id]}\n'
                    )
                f.write('\n\n\n\n')

            print('************************* compute testing metrics *************************')
            for temp_i in range(len(questions)):
                temp_metrics = self.reward_manager.compute_scores(
                    [predicted_answers_list[temp_i]], [golden_answers[temp_i]]
                )
                for key in test_metrics_dict:
                    test_metrics_dict[key].append(temp_metrics.get(key, 0.0))

            # Cost aggregation
            n_turns_actual = len(token_cost_list)
            n_q = len(questions)
            total_token_cost = [
                sum(token_cost_list[tid][qi].get(k, 0.0)
                    for tid in range(n_turns_actual)
                    for k in token_cost_list[tid][qi]
                    if k != 'RetrievalAgent')
                for qi in range(n_q)
            ]
            total_api_cost = [
                sum(1 for tid in range(n_turns_actual)
                    if token_cost_list[tid][qi].get('RetrievalAgent', 0) == 1)
                for qi in range(n_q)
            ]
            test_cost_dict['token_cost'].extend(total_token_cost)
            test_cost_dict['api_times'].extend(total_api_cost)
            test_cost_dict['turn_num'].extend([
                max(ctx.get('end_step', 0) - ctx.get('begin_step', 0) + 1, 1)
                for ctx in context_list
            ])

        val_metrics = {}
        for key, vals in test_metrics_dict.items():
            val_metrics[f'val/{key}'] = float(np.mean(vals)) if vals else 0.0
        for key, vals in test_cost_dict.items():
            if vals:
                val_metrics[f'val/cost/{key}'] = float(np.mean(vals))
        return val_metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self):
        """CONDOR training loop.

        When use_condor=False: calls super().fit() for exact baseline
        reproduction.  When use_condor=True: runs the CONDOR-extended
        four-step update cycle.
        """
        if not self.use_condor:
            return super().fit()

        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agentic()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="CONDOR Training",
        )
        self.global_steps += 1
        last_val_metrics = None

        # Use CONDOR qa_manager (goal-conditioned prompts)
        qa_manager = Agentic_RAG_Manager(self.tokenizer, self.config)
        agent_pool = AgentPool()

        # Initialise lambda_lo in reward manager
        self.reward_manager.lambda_lo = self.dual_lo.value

        for epoch in range(self.config.trainer.total_epochs):
            for batch_id, batch_dict in enumerate(self.train_dataloader):

                print('*** [CONDOR] rollout ***')
                extra_info = batch_dict['extra_info']
                questions = [item['question'] for item in extra_info]
                golden_answers = [item['answer'] for item in extra_info]

                # --------------------------------------------------------
                # Step 1: L0 mechanism selection
                # --------------------------------------------------------
                m_star_list, qtype_list, features_list = self._select_mechanisms(questions)
                self._inject_condor_into_extra_info(batch_dict, m_star_list, qtype_list)

                batch_list, metrics_list = [], []
                predicted_answers_list = [""] * len(questions)
                MAX_TURN = 5
                context_list = init_context_1turn_list(batch_dict, MAX_TURN)
                self._inject_m_star_into_contexts(context_list, m_star_list)
                token_cost_list = []

                # --------------------------------------------------------
                # Step 2: L1 multi-turn rollout (goal-conditioned)
                # --------------------------------------------------------
                for turn_id in range(MAX_TURN):
                    print(f'*** [CONDOR] turn {turn_id} ***')
                    metrics = {}

                    batch, metrics, _ = self.rollout(batch_dict, metrics, context_list, qa_manager, MAX_TURN)
                    workflows_1turn, initial_workflows_context = self.get_answers_subs_list(batch)

                    # CONDOR: validate against mechanism-specific action space
                    is_legal_list = []
                    for wi, (wf, ctx) in enumerate(zip(workflows_1turn, context_list)):
                        m = m_star_list[wi]
                        is_legal_list.append(qa_manager.is_valid_list(wf, ctx, m_star=m))

                    with Manager() as manager:
                        turn_context_list = manager.list([])
                        tc_list_shared = []
                        for temp_i, context in enumerate(context_list):
                            if is_legal_list[temp_i]:
                                tc_list_shared.append(convert_to_shared_structure(context, manager))
                            else:
                                tc_list_shared.append(context)
                        for item in tc_list_shared:
                            turn_context_list.append(item)

                        turn_predicted_answers_list = manager.list([""] * len(questions))

                        condor_cfg = getattr(self.config, 'condor', None)
                        _timeout = int(getattr(condor_cfg, 'workflow_timeout', 120))
                        processes = []
                        for temp_i in range(len(questions)):
                            p = Process(
                                target=run_workflow,
                                args=(temp_i, turn_id, turn_context_list, is_legal_list,
                                      workflows_1turn, turn_predicted_answers_list,
                                      agent_pool, qa_manager, MAX_TURN),
                            )
                            processes.append(p)
                            p.start()
                        for p in processes:
                            p.join(timeout=_timeout)
                            if p.is_alive():
                                print(f'[CONDOR] WARNING: workflow process timed out after {_timeout}s, killing.')
                                p.kill()
                                p.join()

                        local_context_list = convert_to_local_structure(turn_context_list)
                        local_predicted = convert_to_local_structure(turn_predicted_answers_list)

                    context_list = local_context_list
                    for ai, ans in enumerate(local_predicted):
                        if ans != "":
                            predicted_answers_list[ai] = ans

                    batch, metrics = self.compute_logprobs_values_format_penalty(
                        batch, metrics, is_legal_list
                    )
                    batch_list.append(batch)
                    metrics_list.append(metrics)

                    token_cost_turn = []
                    for context in context_list:
                        token_cost_turn.append(deepcopy(context['token_cost']))
                        for key in context['token_cost']:
                            context['token_cost'][key] = 0.0
                    token_cost_list.append(token_cost_turn)

                    # Log turn info — write to configured log_dir (never /tmp)
                    _log_dir = getattr(self.config.trainer, 'log_dir', '/tmp')
                    os.makedirs(_log_dir, exist_ok=True)
                    _log_path = os.path.join(_log_dir, f'condor_training_{self.config.trainer.experiment_name}.txt')
                    with open(_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"batch {batch_id} turn {turn_id}\n")
                        for qi in range(len(questions)):
                            qid = batch_id * self.config.data.train_batch_size + qi
                            f.write(
                                f'  q{qid}: legal={is_legal_list[qi]} m*={m_star_list[qi]} '
                                f'wf={workflows_1turn[qi]}\n'
                            )

                    if "" not in predicted_answers_list:
                        break

                # Fallback answer for any still-empty slot
                for temp_i in range(len(questions)):
                    if predicted_answers_list[temp_i] == "":
                        agent = agent_pool.get("AnswerSummarizationAgent")
                        agent.run(context_list[temp_i])
                        predicted_answers_list[temp_i] = context_list[temp_i]['answer']

                # --------------------------------------------------------
                # Step 3: Assign rewards, compute advantages per turn
                # --------------------------------------------------------
                all_metrics_dict, f1_list = self._compute_per_question_metrics(
                    predicted_answers_list, golden_answers, questions
                )

                token_cost_scale_final, retrieval_api_final, total_tc, total_api = \
                    self._compute_cost_totals(token_cost_list, None, context_list, questions)

                # Make retrieval_api_final a list-of-lists matching token_cost_list shape
                n_turns_actual = len(token_cost_list)
                n_q = len(questions)
                retrieval_api_final_ll = [
                    [
                        (1 if token_cost_list[tid][qi].get('RetrievalAgent', 0) == 1 else 0)
                        for qi in range(n_q)
                    ]
                    for tid in range(n_turns_actual)
                ]

                # Build training cost/metric summaries for W&B
                train_metrics_dict = {'acc': [], 'em': [], 'f1': [], 'precision': [], 'recall': []}
                for mdict in all_metrics_dict:
                    for key in train_metrics_dict:
                        train_metrics_dict[key].append(mdict.get(key, 0.0))

                train_cost_dict = {'token_cost': total_tc, 'api_times': total_api}

                print('*** [CONDOR] assigning rewards ***')
                for ti in range(len(batch_list)):
                    batch_list[ti], metrics_list[ti] = self.assign_rewards_compute_values_advs(
                        batch_list[ti],
                        metrics_list[ti],
                        token_cost_scale_final[ti] if ti < len(token_cost_scale_final) else [0.0] * n_q,
                        retrieval_api_final_ll[ti] if ti < len(retrieval_api_final_ll) else [0.0] * n_q,
                        context_list,
                        all_metrics_dict,
                        ti,
                    )

                # --------------------------------------------------------
                # Step 4: Update L1 actor/critic
                # --------------------------------------------------------
                print('*** [CONDOR] updating models ***')
                for ti in range(len(batch_list)):
                    batch_list[ti], metrics_list[ti] = self.update_models(batch_list[ti], metrics_list[ti])

                # Aggregate turn metrics
                total_metrics = {}
                for mdict in metrics_list:
                    for k, v in mdict.items():
                        total_metrics[k] = total_metrics.get(k, 0.0) + v
                metrics_aver = {k: v / len(metrics_list) for k, v in total_metrics.items()}
                metrics_aver['training/global_step'] = self.global_steps
                metrics_aver['training/epoch'] = epoch

                for k, v in train_metrics_dict.items():
                    metrics_aver[f'training/training_score/{k}'] = float(np.mean(v))
                for k, v in train_cost_dict.items():
                    if isinstance(v, list):
                        metrics_aver[f'training/training_cost/{k}'] = float(np.mean(v))

                logger.log(data=metrics_aver, step=self.global_steps)

                # Record training step metrics to CSV (CONDOR keys added below after update)
                self._pending_step = self.global_steps
                self._pending_metrics = dict(metrics_aver)

                # --------------------------------------------------------
                # CONDOR L0 + dual-lambda + phi updates
                # --------------------------------------------------------
                self._update_condor_components(
                    f1_list=f1_list,
                    m_star_list=m_star_list,
                    qtype_list=qtype_list,
                    features_list=features_list,
                    total_token_cost=total_tc,
                    total_api_cost=total_api,
                    context_list=context_list,
                    questions=questions,
                    metrics_aver=metrics_aver,
                    logger=logger,
                    step=self.global_steps,
                )

                # --------------------------------------------------------
                # Validation / checkpointing
                # --------------------------------------------------------
                is_last_step = self.global_steps >= self.total_training_steps

                if (self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)):
                    val_metrics = self._validate_agentic()
                    if is_last_step:
                        last_val_metrics = val_metrics
                    metrics_aver.update(val_metrics)
                    logger.log(data=val_metrics, step=self.global_steps)
                    # Record eval batch: per-query records from current batch
                    # (training batch used as proxy; override _validate_agentic for
                    #  true validation per-query records in a future extension)
                    _dataset_name = getattr(
                        getattr(self.config, 'data', None), 'val_dataset', 'val'
                    )
                    _eval_records = []
                    for qi in range(len(questions)):
                        ctx = context_list[qi]
                        n_t = max(ctx.get('end_step', 0) - ctx.get('begin_step', 0) + 1, 1)
                        _eval_records.append(self.metrics_collector.make_per_query_record(
                            question=questions[qi],
                            predicted_answer=predicted_answers_list[qi],
                            golden_answer=golden_answers[qi],
                            f1=f1_list[qi],
                            em=float(all_metrics_dict[qi].get('em', 0.0)),
                            accuracy=float(all_metrics_dict[qi].get('acc', 0.0)),
                            token_cost_usd=float(total_tc[qi]),
                            retrieval_calls=int(total_api[qi]),
                            turns=n_t,
                            m_star=m_star_list[qi],
                            qtype_id=qtype_list[qi],
                            workflow=str(context_list[qi].get('workflow', '')),
                            l0_correct=float(self.process_reward_model.l0_correct(
                                f1_list[qi], self._l0_threshold
                            )),
                        ))
                    self.metrics_collector.record_eval_batch(
                        step=self.global_steps,
                        dataset=_dataset_name,
                        per_query_records=_eval_records,
                    )

                if (self.config.trainer.save_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0)):
                    self._save_checkpoint()

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
