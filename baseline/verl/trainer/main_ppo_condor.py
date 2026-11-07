"""CONDOR entry point for verl PPO training.

Mirrors main_ppo_agentic_rag.py but wires in the CONDOR two-level RL
components: MechanismRouter (L0), DualAlphaManager, PhiScorer,
MetricsCollector, and TrajectoryLogger.

Usage (via slurm/train_condor_nq.slurm):
    python -m verl.trainer.main_ppo_condor \
        --config-path /path/to/configs \
        --config-name condor_nq
"""

import collections
import re
import string
from typing import Any, Dict, List

from rouge_score import rouge_scorer as _rouge_scorer_module

_ROUGE = _rouge_scorer_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

import hydra
import ray
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer_agentic_rag_2 import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


# ---------------------------------------------------------------------------
# Text normalisation helpers (shared with baseline)
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    def _remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def _white_space_fix(text):
        return " ".join(text.split())

    def _remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return _white_space_fix(_remove_articles(_remove_punc(s.lower())))


def _remove_trailing_marker(text: str) -> str:
    marker = "<|im_end|>"
    return text[: -len(marker)] if text.endswith(marker) else text


# ---------------------------------------------------------------------------
# CONDOR reward manager
# ---------------------------------------------------------------------------

class CondorRewardManager:
    """Extends the baseline RewardManager with CONDOR L0 components.

    The manager holds:
        router       — MechanismRouter (epsilon-greedy MLP, REINFORCE)
        dual_alpha   — DualAlphaManager (Lagrange multipliers λ_hi, λ_lo)
        phi          — PhiScorer (blended counterfactual baseline)
        traj_logger  — TrajectoryLogger (failure taxonomy buffer)

    These are initialised once on the driver and updated after every
    assign_rewards call.  All heavy RL work (actor / critic) still
    runs inside the distributed Ray workers via RayPPOTrainer.
    """

    def __init__(self, tokenizer, condor_cfg):
        self.tokenizer = tokenizer
        self.cfg = condor_cfg

        # Import CONDOR components (PYTHONPATH includes project root)
        from condor import (
            DualAlphaManager,
            MechanismRouter,
            PhiScorer,
            TrajectoryLogger,
        )
        from condor.metrics.collector import MetricsCollector

        self.router = MechanismRouter(
            epsilon_start=condor_cfg.epsilon_start,
            epsilon_end=condor_cfg.epsilon_end,
            epsilon_decay=condor_cfg.epsilon_decay,
            lr=condor_cfg.router_lr,
            gamma=condor_cfg.reinforce_gamma,
        )
        self.dual_hi = DualAlphaManager(
            budget=condor_cfg.budget_hi,
            lr=condor_cfg.dual_lr_hi,
            lam_init=condor_cfg.lambda_hi_init,
        )
        self.dual_lo = DualAlphaManager(
            budget=condor_cfg.budget_lo,
            lr=condor_cfg.dual_lr_lo,
            lam_init=condor_cfg.lambda_lo_init,
        )
        # Unified accessor kept for assign_token_retrieval_cost
        self.dual_alpha = self.dual_lo
        self.phi = PhiScorer(
            t_warmup=condor_cfg.phi_t_warmup,
            tau_trans=condor_cfg.phi_tau_trans,
            lr=condor_cfg.phi_lr,
        )
        self.traj_logger = TrajectoryLogger()
        self.metrics_collector = MetricsCollector()

        self._global_step = 0
        self._l0_reward_threshold = condor_cfg.l0_reward_threshold

    # ---- score helpers -------------------------------------------------------

    def compute_scores(self, predict_answers: List[str], golden_answers: List[Any]) -> Dict:
        assert len(predict_answers) == len(golden_answers)
        total = len(predict_answers)
        final = {
            "acc": 0.0, "em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
        }

        for prediction, ground_truths in zip(predict_answers, golden_answers):
            if isinstance(ground_truths, str):
                ground_truths = [ground_truths]
            norm_pred = _normalize_answer(prediction)
            best = {
                "acc": 0.0, "em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
            }
            for gt in ground_truths:
                norm_gt = _normalize_answer(gt)
                if norm_pred in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
                    continue
                if norm_gt in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
                    continue
                if norm_gt in norm_pred:
                    best["acc"] = 1.0
                if norm_pred == norm_gt:
                    best["em"] = 1.0
                pred_toks = norm_pred.split()
                gt_toks = norm_gt.split()
                common = collections.Counter(pred_toks) & collections.Counter(gt_toks)
                num_same = sum(common.values())
                if num_same > 0:
                    p = num_same / len(pred_toks)
                    r = num_same / len(gt_toks)
                    f1 = 2 * p * r / (p + r)
                    best["precision"] = max(p, best["precision"])
                    best["recall"] = max(r, best["recall"])
                    best["f1"] = max(f1, best["f1"])
                # ROUGE scores (F-measure)
                if norm_pred and norm_gt:
                    rouge_scores = _ROUGE.score(norm_gt, norm_pred)
                    best["rouge1"] = max(rouge_scores["rouge1"].fmeasure, best["rouge1"])
                    best["rouge2"] = max(rouge_scores["rouge2"].fmeasure, best["rouge2"])
                    best["rougeL"] = max(rouge_scores["rougeL"].fmeasure, best["rougeL"])
            for k in final:
                final[k] += best[k]

        for k in final:
            final[k] /= total
        return final

    # ---- reward interface (matches baseline RewardManager) -------------------

    def get_format_penalty(self, data: DataProto, is_legal_list: List[bool]):
        fmt = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i in range(len(data)):
            item = data[i]
            plen = item.batch["prompts"].shape[-1]
            vlen = item.batch["attention_mask"][plen:].sum()
            if not is_legal_list[i]:
                fmt[i, vlen - 1] = -1.0
        return fmt

    def assign_rewards(self, data, metrics, context_list, all_metrics_dict_list, turn_id):
        f1_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i in range(len(context_list)):
            item = data[i]
            plen = item.batch["prompts"].shape[-1]
            vlen = item.batch["attention_mask"][plen:].sum()
            ctx = context_list[i]
            if ctx["begin_step"] <= turn_id <= ctx["end_step"]:
                f1_tensor[i, vlen - 1] = all_metrics_dict_list[i]["f1"]
        return f1_tensor

    def assign_token_retrieval_cost(
        self, data, metrics, context_list, token_cost_list, retrieval_api_cost_list, turn_id
    ):
        """Baseline cost tensor scaled by λ_lo (CONDOR dual variable)."""
        cost_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        lambda_lo = float(self.dual_lo.value)

        for i in range(len(context_list)):
            item = data[i]
            plen = item.batch["prompts"].shape[-1]
            vlen = item.batch["attention_mask"][plen:].sum()
            ctx = context_list[i]

            token_cost = token_cost_list[i]
            retrieval_cost = retrieval_api_cost_list[i]

            if ctx.get("mode") == "serial" and turn_id == ctx["begin_step"]:
                latency = 0.25 * len(ctx.get("sub_query", []))
            elif ctx.get("mode") == "parallel" and turn_id == ctx["begin_step"]:
                latency = 0.25
            else:
                latency = 0.0

            raw_cost = -1.0 * token_cost + -0.25 * retrieval_cost + -0.5 * latency
            # λ_lo scales from 0 (baseline) upward as budget is violated
            if ctx["begin_step"] <= turn_id <= ctx["end_step"]:
                cost_tensor[i, vlen - 1] = lambda_lo * raw_cost

        return cost_tensor

    def get_rewards(self, data: DataProto, is_legal_list: List[bool]):
        """Used during validation rollouts."""
        metric_acc  = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        metric_em   = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        metric_f1   = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        metric_pre  = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        metric_rec  = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        metric_r1   = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        metric_r2   = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        metric_rl   = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        fmt = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(len(data)):
            item = data[i]
            plen = item.batch["prompts"].shape[-1]
            vlen = item.batch["attention_mask"][plen:].sum()
            resp_ids = item.batch["responses"][:vlen]
            seq_str = _remove_trailing_marker(self.tokenizer.decode(resp_ids))
            gt = item.non_tensor_batch["reward_model"]["ground_truth"]
            m = self.compute_scores([seq_str], [gt])

            metric_acc[i, vlen - 1]  = m["acc"]
            metric_em[i, vlen - 1]   = m["em"]
            metric_f1[i, vlen - 1]   = m["f1"]
            metric_pre[i, vlen - 1]  = m["precision"]
            metric_rec[i, vlen - 1]  = m["recall"]
            metric_r1[i, vlen - 1]   = m["rouge1"]
            metric_r2[i, vlen - 1]   = m["rouge2"]
            metric_rl[i, vlen - 1]   = m["rougeL"]
            if not is_legal_list[i]:
                fmt[i, vlen - 1] = -1.0

        final = metric_f1 + fmt
        return final, [metric_acc, metric_em, metric_f1, metric_pre, metric_rec,
                       metric_r1, metric_r2, metric_rl]


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                }
            },
            num_cpus=config.ray_init.num_cpus if hasattr(config, "ray_init") else None,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name == "vllm":
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
                CriticWorker,
            )

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer_agentic_rag_2 import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        from verl.utils.dataset.rl_dataset import collate_fn

        # CONDOR reward manager (wraps baseline with L0 components)
        reward_manager = CondorRewardManager(tokenizer, condor_cfg=config.condor)

        train_dataset = _create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = _create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = _create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_manager=reward_manager,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


# ---------------------------------------------------------------------------
# Dataset / sampler helpers (copied from main_ppo_agentic_rag.py)
# ---------------------------------------------------------------------------

def _create_rl_dataset(data_paths, data_config, tokenizer, processor):
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"Custom dataset class '{data_config.custom_cls.name}' must inherit from Dataset"
            )
    else:
        dataset_cls = RLHFDataset

    print(f"Using dataset class: {dataset_cls.__name__}")
    return dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )


def _create_rl_sampler(data_config, dataset):
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        gen = torch.Generator()
        gen.manual_seed(data_config.get("seed", 1))
        return RandomSampler(data_source=dataset, generator=gen)
    return SequentialSampler(data_source=dataset)


if __name__ == "__main__":
    main()
