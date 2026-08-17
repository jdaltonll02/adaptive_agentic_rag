"""CONDOR data/nq_open.py

Extended version of baseline data/nq_open.py.

Adds an optional `mechanism_label` field to each example so that
offline L0 oracle supervision can be applied if desired.
The mechanism label is derived from a simple heuristic on the question;
training code may ignore this field.

Backward compatible: all existing baseline fields are preserved.

Usage
-----
    python -m data.nq_open \\
        --train-json /data/user_data/jgibson2/condor/data/nq/nq_train_questions_and_answers.json \\
        --test-json  /data/user_data/jgibson2/condor/data/nq/nq_test_questions_and_answers.json \\
        --local-dir  /data/user_data/jgibson2/condor/data/nq
"""

import argparse
import json
import os
import sys

from datasets import Dataset
from typing import List

# ---------------------------------------------------------------------------
# Optional: add project root to path so condor module is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from condor.mechanism_router import classify_qtype
    _CONDOR_AVAILABLE = True
except ImportError:
    _CONDOR_AVAILABLE = False


def extract_solution(solution_list: List) -> str:
    if isinstance(solution_list, str):
        return solution_list
    return ', '.join(solution_list)


def _oracle_mechanism(question: str) -> int:
    """Simple heuristic oracle for the L0 mechanism label (not used in RL training)."""
    if not _CONDOR_AVAILABLE:
        return 1  # default to Standard RAG
    qtype = classify_qtype(question)
    return {0: 1, 1: 2, 2: 3, 3: 4}.get(qtype, 1)


def make_map_fn(split: str, data_source: str):
    def process_fn(example, idx):
        question_raw = example.pop('question') if 'question' in example else ''
        answer_raw = example.pop('answer') if 'answer' in example else []
        mechanism_label = _oracle_mechanism(question_raw)
        return {
            'data_source': data_source,
            'prompt': [{'role': 'user', 'content': question_raw}],
            'ability': 'QA',
            'reward_model': {'style': 'rule', 'ground_truth': answer_raw},
            'extra_info': {
                'split': split,
                'index': idx,
                'answer': answer_raw,
                'question': question_raw,
                'mechanism_label': mechanism_label,
            },
        }
    return process_fn


def build_dataset(json_path: str, split: str, data_source: str = 'google-research-datasets/nq_open') -> Dataset:
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    ds = Dataset.from_list(records)
    return ds.map(function=make_map_fn(split, data_source), with_indices=True)


if __name__ == '__main__':
    _CONDOR_DATA = '/data/user_data/jgibson2/condor/data/nq'

    parser = argparse.ArgumentParser(description='Prepare NQ parquet files for CONDOR training.')
    parser.add_argument('--train-json',
                        default=os.path.join(_CONDOR_DATA, 'nq_train_questions_and_answers.json'),
                        help='Path to NQ train JSON')
    parser.add_argument('--test-json',
                        default=os.path.join(_CONDOR_DATA, 'nq_test_questions_and_answers.json'),
                        help='Path to NQ test JSON')
    parser.add_argument('--local-dir', default=_CONDOR_DATA,
                        help='Output directory for parquet files')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip test split generation')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of training examples (None = all)')
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print(f'[nq_open] Building train split from {args.train_json}')
    train_dataset = build_dataset(args.train_json, split='train')
    if args.limit:
        train_dataset = train_dataset.select(range(min(args.limit, len(train_dataset))))
    train_out = os.path.join(args.local_dir, 'train.parquet')
    train_dataset.to_parquet(train_out)
    print(f'[nq_open] Train parquet ({len(train_dataset)} rows) → {train_out}')

    if not args.skip_test and os.path.exists(args.test_json):
        print(f'[nq_open] Building test split from {args.test_json}')
        test_dataset = build_dataset(args.test_json, split='test')
        test_out = os.path.join(args.local_dir, 'test_996.parquet')
        test_dataset.to_parquet(test_out)
        print(f'[nq_open] Test parquet  ({len(test_dataset)} rows) → {test_out}')
    else:
        print('[nq_open] Skipping test split.')

    print('[nq_open] Done.')
