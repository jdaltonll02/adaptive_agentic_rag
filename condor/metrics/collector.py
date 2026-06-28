"""MetricsCollector — records all training and evaluation signals.

During training the trainer calls:
    collector.record_training_step(step, metrics_dict)

During evaluation the trainer calls:
    collector.record_eval_batch(step, dataset, per_query_records)

All data is flushed to disk after every call so that partially-completed
runs are still analysable.

Output files (in metrics_dir):
    {exp}_training.csv        — one row per training step
    {exp}_eval.jsonl          — one JSON line per eval batch
    {exp}_eval_summary.json   — latest per-dataset aggregate snapshot
"""

import csv
import json
import os
from typing import Any, Dict, List, Optional


# Metrics collected per training step
_TRAINING_COLS = [
    'step', 'epoch',
    # L1 quality
    'train/f1', 'train/em', 'train/accuracy', 'train/precision', 'train/recall',
    # L1 cost
    'train/token_cost_mean', 'train/retrieval_calls_mean', 'train/turns_mean',
    # L1 RL
    'actor/entropy_loss', 'actor/pg_loss', 'actor/kl',
    'critic/vf_loss',
    # CONDOR L0
    'condor/lambda_hi', 'condor/lambda_lo',
    'condor/l0_policy_loss', 'condor/phi_loss',
    'condor/avg_l0_reward', 'condor/avg_l0_correct', 'condor/avg_f1',
    'condor/router_epsilon',
    # Mechanism distribution
    'condor/mech_Pure_LLM', 'condor/mech_Standard_RAG',
    'condor/mech_Advanced_Iterative_RAG', 'condor/mech_Graph_RAG',
    'condor/mech_Web_Search_RAG',
]

# Keys logged per query during evaluation
_QUERY_KEYS = [
    'question', 'predicted_answer', 'golden_answer',
    'f1', 'em', 'accuracy',
    'token_cost_usd', 'retrieval_calls', 'turns',
    'm_star', 'qtype_id', 'workflow',
    'l0_correct',
]


class MetricsCollector:
    """Streams training and evaluation metrics to disk.

    Parameters
    ----------
    metrics_dir : str
        Directory for light-output CSV/JSON files (e.g. outputs/metrics/).
    experiment_name : str
        Unique run identifier; becomes the filename prefix.
    """

    def __init__(self, metrics_dir: str, experiment_name: str):
        self.metrics_dir = metrics_dir
        self.experiment_name = experiment_name
        os.makedirs(metrics_dir, exist_ok=True)

        self._train_path = os.path.join(metrics_dir, f'{experiment_name}_training.csv')
        self._eval_path = os.path.join(metrics_dir, f'{experiment_name}_eval.jsonl')
        self._eval_summary_path = os.path.join(metrics_dir, f'{experiment_name}_eval_summary.json')

        self._train_writer: Optional[csv.DictWriter] = None
        self._train_fh = None
        self._eval_summary: Dict[str, Any] = {}

        self._init_training_csv()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _init_training_csv(self) -> None:
        self._train_fh = open(self._train_path, 'w', newline='', encoding='utf-8')
        self._train_writer = csv.DictWriter(
            self._train_fh, fieldnames=_TRAINING_COLS, extrasaction='ignore'
        )
        self._train_writer.writeheader()
        self._train_fh.flush()

    def record_training_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """Write one row to the training CSV.

        Accepts any flat dict from the trainer; unknown keys are silently
        dropped (extrasaction='ignore').  Missing columns are written as ''.
        """
        row = {'step': step}
        row.update(metrics)
        # Normalise nested CONDOR mechanism keys
        for m_name in ['Pure_LLM', 'Standard_RAG', 'Advanced_Iterative_RAG',
                        'Graph_RAG', 'Web_Search_RAG']:
            wandb_key = f'condor/mechanism/{m_name}'
            if wandb_key in row:
                row[f'condor/mech_{m_name}'] = row.pop(wandb_key)
        self._train_writer.writerow(row)
        self._train_fh.flush()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def record_eval_batch(
        self,
        step: int,
        dataset: str,
        per_query_records: List[Dict[str, Any]],
    ) -> None:
        """Append one evaluation result block and update the summary snapshot.

        Parameters
        ----------
        step : int
        dataset : str
            Name of the evaluation dataset (e.g. 'nq', 'hotpotqa').
        per_query_records : list[dict]
            One dict per query with keys matching _QUERY_KEYS.
        """
        if not per_query_records:
            return

        # Aggregate
        def mean(key):
            vals = [r[key] for r in per_query_records if key in r and r[key] is not None]
            return float(sum(vals) / len(vals)) if vals else 0.0

        aggregate = {
            'step': step,
            'dataset': dataset,
            'n_queries': len(per_query_records),
            'f1': mean('f1'),
            'em': mean('em'),
            'accuracy': mean('accuracy'),
            'token_cost_usd_mean': mean('token_cost_usd'),
            'token_cost_milliusd_mean': mean('token_cost_usd') * 1000,
            'retrieval_calls_mean': mean('retrieval_calls'),
            'turns_mean': mean('turns'),
            'l0_correct_mean': mean('l0_correct'),
        }

        # Per-mechanism breakdown (CONDOR only)
        for m_id in range(5):
            qs = [r for r in per_query_records if r.get('m_star') == m_id]
            if qs:
                aggregate[f'mech_{m_id}_f1'] = float(sum(r['f1'] for r in qs) / len(qs))
                aggregate[f'mech_{m_id}_count'] = len(qs)

        # Per-workflow turn cost (analogous to Table 4)
        wf_turns: Dict[str, List[float]] = {}
        for r in per_query_records:
            wf = r.get('workflow', 'Other')
            wf_turns.setdefault(wf, []).append(r.get('turns', 0))
        aggregate['per_workflow_turns'] = {k: sum(v) / len(v) for k, v in wf_turns.items()}

        # Write JSONL
        record = {'aggregate': aggregate, 'per_query': per_query_records}
        with open(self._eval_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, default=str) + '\n')

        # Update summary snapshot
        self._eval_summary[dataset] = aggregate
        with open(self._eval_summary_path, 'w', encoding='utf-8') as fh:
            json.dump(self._eval_summary, fh, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def make_per_query_record(
        self,
        question: str,
        predicted_answer: str,
        golden_answer: Any,
        f1: float,
        em: float,
        accuracy: float,
        token_cost_usd: float,
        retrieval_calls: int,
        turns: int,
        m_star: Optional[int] = None,
        qtype_id: Optional[int] = None,
        workflow: Optional[str] = None,
        l0_correct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convenience constructor for a single per-query record dict."""
        return {
            'question': question,
            'predicted_answer': predicted_answer,
            'golden_answer': golden_answer,
            'f1': f1,
            'em': em,
            'accuracy': accuracy,
            'token_cost_usd': token_cost_usd,
            'retrieval_calls': retrieval_calls,
            'turns': turns,
            'm_star': m_star,
            'qtype_id': qtype_id,
            'workflow': workflow,
            'l0_correct': l0_correct,
        }

    def close(self) -> None:
        if self._train_fh:
            self._train_fh.close()
