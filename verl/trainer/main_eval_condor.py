"""CONDOR main_eval_condor.py

Standalone evaluation script for a trained CONDOR checkpoint.
Runs the full CONDOR pipeline (mechanism selection + goal-conditioned planning)
on a held-out evaluation set and writes per-query results, aggregate metrics,
and all plots.

Usage
-----
Via SLURM (preferred):
    sbatch slurm/eval_condor_nq.slurm

Via shell:
    python -m verl.trainer.main_eval_condor \\
        --config-path /home/jgibson2/projects/adaptive_rag/configs \\
        --config-name eval_condor_nq \\
        "eval.checkpoint_path=/data/user_data/jgibson2/condor/checkpoints/condor_nq_<id>"

The script writes:
    {metrics_dir}/{experiment}_eval.jsonl
    {metrics_dir}/{experiment}_eval_summary.json
    {plots_dir}/fig3_pareto.{pdf,png}
    {plots_dir}/mechanism_f1_bar.{pdf,png}
    {plots_dir}/tableD_pareto_summary.{csv,md,tex}
    ... (all tables and plots)
"""

import os
import sys
from typing import List

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
)
_BASELINE_ROOT = os.path.join(_PROJECT_ROOT, 'baseline')

for _p in [_PROJECT_ROOT, _BASELINE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import re
import string
from collections import defaultdict
from multiprocessing import Manager, Process
from pprint import pprint

import hydra
import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Baseline verl infrastructure
from verl.utils.fs import copy_local_path_from_hdfs
from transformers import AutoTokenizer

# CONDOR components
from condor import MechanismRouter, MECHANISM_NAMES
from condor.metrics import MetricsCollector
from condor.metrics.tables import TablesGenerator
from condor.process_reward import ProcessRewardModel

# CONDOR qa_manager
from qa_manager.qa import Agentic_RAG_Manager
from qa_manager.BaseAgent import AgentPool

# Baseline trainer utilities
from verl.trainer.ppo.ray_trainer_agentic_rag_2 import (
    RewardManager,
    run_workflow,
    init_context_1turn_list,
    convert_to_shared_structure,
    convert_to_local_structure,
)


# ---------------------------------------------------------------------------
# Normalisation helpers (matches baseline scoring)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction: str, ground_truths) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    best_f1 = 0.0
    for gt in ground_truths:
        pred_tokens = normalize_answer(prediction).split()
        gt_tokens = normalize_answer(gt).split()
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue
        p = len(common) / len(pred_tokens) if pred_tokens else 0
        r = len(common) / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        best_f1 = max(best_f1, f1)
    return best_f1


def compute_em(prediction: str, ground_truths) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    norm_pred = normalize_answer(prediction)
    return float(any(norm_pred == normalize_answer(gt) for gt in ground_truths))


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    pprint(OmegaConf.to_container(config, resolve=True))

    # ---- Output dirs ----
    for _d in [
        str(config.trainer.get('metrics_dir', '')),
        str(config.trainer.get('plots_dir', '')),
        str(config.trainer.get('log_dir', '')),
    ]:
        if _d:
            os.makedirs(_d, exist_ok=True)

    # ---- Tokenizer ----
    ckpt_path = str(config.eval.checkpoint_path)
    local_path = copy_local_path_from_hdfs(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    # ---- Ray ----
    if not ray.is_initialized():
        ray.init(
            runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true'}},
            num_cpus=config.trainer.get('ray_num_cpus', None),
        )

    # ---- CONDOR components ----
    router = MechanismRouter(
        epsilon_start=0.0,   # pure exploitation during eval
        epsilon_end=0.0,
        epsilon_decay=1.0,
        use_condor=config.condor.get('use_condor', True),
    )
    # Load router weights if provided
    if config.eval.get('router_checkpoint') and os.path.exists(str(config.eval.router_checkpoint)):
        import torch
        router.policy.load_state_dict(torch.load(str(config.eval.router_checkpoint)))

    reward_manager = RewardManager(tokenizer)
    prm = ProcessRewardModel()
    l0_threshold = float(config.condor.get('l0_reward_threshold', 0.4))

    metrics_collector = MetricsCollector(
        metrics_dir=str(config.trainer.metrics_dir),
        experiment_name=str(config.trainer.experiment_name),
    )

    qa_manager = Agentic_RAG_Manager(tokenizer, config)
    agent_pool = AgentPool()

    # ---- Load eval data ----
    import pandas as pd
    val_files = OmegaConf.to_container(config.data.val_files)
    dfs = [pd.read_parquet(f) for f in val_files]
    val_df = pd.concat(dfs, ignore_index=True)

    dataset_name = str(config.data.get('val_dataset', 'eval'))
    MAX_TURN = int(config.eval.get('max_turn', 5))

    per_query_records = []
    print(f'[eval] Running on {len(val_df)} examples, dataset={dataset_name}')

    # Process in mini-batches matching train_batch_size for memory efficiency
    batch_size = int(config.eval.get('eval_batch_size', 8))
    rows = val_df.to_dict('records')

    for batch_start in tqdm(range(0, len(rows), batch_size), desc='Eval batches'):
        batch_rows = rows[batch_start: batch_start + batch_size]
        questions = [r.get('question', '') or r.get('prompt', '') for r in batch_rows]
        golden_answers = [r.get('answer', r.get('reward_model', {}).get('ground_truth', ''))
                          for r in batch_rows]

        # Ensure lists
        questions = [str(q) if not isinstance(q, str) else q for q in questions]

        # L0: mechanism selection (pure exploitation, epsilon=0)
        m_star_list, qtype_list, features_list = [], [], []
        for q in questions:
            m, qt, feat = router.route(q)
            m_star_list.append(m)
            qtype_list.append(qt)
            features_list.append(feat)

        # Build dummy batch_dict for context initialisation
        batch_dict = {
            'extra_info': [
                {'question': q, 'answer': a, 'm_star': m, 'qtype_id': qt}
                for q, a, m, qt in zip(questions, golden_answers, m_star_list, qtype_list)
            ]
        }

        context_list = init_context_1turn_list(batch_dict, MAX_TURN)
        for ctx, m in zip(context_list, m_star_list):
            ctx['m_star'] = m

        predicted_answers_list = [''] * len(questions)
        token_cost_list = []

        # Multi-turn rollout
        for turn_id in range(MAX_TURN):
            # Build a minimal batch for rollout — evaluation uses generate() directly
            # using the agentic manager to format prompts
            is_legal_list = [True] * len(questions)
            workflows_1turn = [None] * len(questions)

            with Manager() as manager:
                tc_list_shared = [convert_to_shared_structure(ctx, manager) for ctx in context_list]
                turn_context_list = manager.list(tc_list_shared)
                turn_predicted = manager.list([''] * len(questions))

                processes = []
                for qi in range(len(questions)):
                    p = Process(
                        target=run_workflow,
                        args=(qi, turn_id, turn_context_list, is_legal_list,
                              workflows_1turn, turn_predicted, agent_pool,
                              qa_manager, MAX_TURN),
                    )
                    processes.append(p)
                    p.start()
                for p in processes:
                    p.join()

                local_ctx = convert_to_local_structure(turn_context_list)
                local_pred = convert_to_local_structure(turn_predicted)

            context_list = local_ctx
            for ai, ans in enumerate(local_pred):
                if ans != '':
                    predicted_answers_list[ai] = ans

            token_cost_turn = []
            for ctx in context_list:
                from copy import deepcopy
                token_cost_turn.append(deepcopy(ctx['token_cost']))
                for k in ctx['token_cost']:
                    ctx['token_cost'][k] = 0.0
            token_cost_list.append(token_cost_turn)

            if '' not in predicted_answers_list:
                break

        # Fallback
        for qi in range(len(questions)):
            if predicted_answers_list[qi] == '':
                agent = agent_pool.get('AnswerSummarizationAgent')
                agent.run(context_list[qi])
                predicted_answers_list[qi] = context_list[qi].get('answer', '')

        # Build per-query records
        n_turns_actual = len(token_cost_list)
        for qi in range(len(questions)):
            pred = predicted_answers_list[qi]
            gold = golden_answers[qi]
            f1 = compute_f1(pred, gold)
            em = compute_em(pred, gold)

            total_tc = sum(
                sum(v for k, v in token_cost_list[tid][qi].items() if k != 'RetrievalAgent')
                for tid in range(n_turns_actual)
            )
            total_ret = sum(
                1 if token_cost_list[tid][qi].get('RetrievalAgent', 0) == 1 else 0
                for tid in range(n_turns_actual)
            )
            ctx = context_list[qi]
            n_t = max(ctx.get('end_step', 0) - ctx.get('begin_step', 0) + 1, 1)

            record = metrics_collector.make_per_query_record(
                question=questions[qi],
                predicted_answer=pred,
                golden_answer=gold,
                f1=f1,
                em=em,
                accuracy=em,
                token_cost_usd=float(total_tc),
                retrieval_calls=int(total_ret),
                turns=n_t,
                m_star=m_star_list[qi],
                qtype_id=qtype_list[qi],
                workflow=str(ctx.get('workflow', '')),
                l0_correct=float(prm.l0_correct(f1, l0_threshold)),
            )
            per_query_records.append(record)

    # ---- Flush eval metrics ----
    metrics_collector.record_eval_batch(
        step=0,
        dataset=dataset_name,
        per_query_records=per_query_records,
    )
    metrics_collector.close()

    # ---- Tables ----
    tables = TablesGenerator(
        metrics_dir=str(config.trainer.metrics_dir),
        output_dir=str(config.trainer.plots_dir),
        experiment_name=str(config.trainer.experiment_name),
    )
    tables.generate_all()

    # ---- Plots ----
    from condor.plots.pareto import plot_pareto
    from condor.plots.condor_specific import plot_mechanism_f1_bar

    plots_dir = str(config.trainer.plots_dir)
    plot_pareto(per_query_records,
                output_path=os.path.join(plots_dir, 'fig3_pareto.pdf'),
                dataset_label=dataset_name)
    plot_pareto(per_query_records,
                output_path=os.path.join(plots_dir, 'fig3_pareto.png'),
                dataset_label=dataset_name)

    eval_summary_path = os.path.join(
        str(config.trainer.metrics_dir),
        f'{config.trainer.experiment_name}_eval_summary.json'
    )
    if os.path.exists(eval_summary_path):
        with open(eval_summary_path) as fh:
            eval_summary = json.load(fh)
        plot_mechanism_f1_bar(eval_summary,
                              output_path=os.path.join(plots_dir, 'mechanism_f1_bar.pdf'))
        plot_mechanism_f1_bar(eval_summary,
                              output_path=os.path.join(plots_dir, 'mechanism_f1_bar.png'))

    # ---- Print summary ----
    total_f1 = float(np.mean([r['f1'] for r in per_query_records]))
    total_em = float(np.mean([r['em'] for r in per_query_records]))
    total_tc = float(np.mean([r['token_cost_usd'] * 1000 for r in per_query_records]))
    total_ret = float(np.mean([r['retrieval_calls'] for r in per_query_records]))

    print('\n' + '=' * 60)
    print(f'Eval complete — {len(per_query_records)} queries on {dataset_name}')
    print(f'  F1:             {total_f1 * 100:.2f}%')
    print(f'  EM:             {total_em * 100:.2f}%')
    print(f'  Token cost:     {total_tc:.3f} mUSD/query')
    print(f'  Retrieval calls:{total_ret:.2f}/query')
    mech_counts = defaultdict(int)
    for r in per_query_records:
        mech_counts[MECHANISM_NAMES.get(r['m_star'], f"m{r['m_star']}")] += 1
    for mn, cnt in sorted(mech_counts.items()):
        print(f'  {mn}: {cnt} queries ({100*cnt/len(per_query_records):.1f}%)')
    print('=' * 60)
    print(f'Outputs in: {config.trainer.metrics_dir} and {config.trainer.plots_dir}')


@hydra.main(config_path='../../../configs', config_name='eval_condor_nq')
def main(config: DictConfig) -> None:
    run_eval(config)


if __name__ == '__main__':
    main()
