"""CONDOR data/hotpot_qa.py

Extended version of baseline data/hotpot_qa.py.

Adds mechanism_label (oracle L0 heuristic) to each example.
All existing baseline fields are preserved.
"""

import argparse
import json
import os
import sys

from datasets import Dataset

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from condor.mechanism_router import classify_qtype
    _CONDOR_AVAILABLE = True
except ImportError:
    _CONDOR_AVAILABLE = False


def _oracle_mechanism(question: str) -> int:
    if not _CONDOR_AVAILABLE:
        return 2  # HotpotQA is multi-hop → default to m2
    qtype = classify_qtype(question)
    return {0: 1, 1: 2, 2: 3, 3: 4}.get(qtype, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./hotpot_qa")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    data_source = "hotpotqa/hotpot_qa"

    train_dir = "./hotpot_qa/hotpotqa_train_questions_and_answers.json"
    test_dir = "./hotpot_qa/hotpotqa_test_questions_and_answers.json"

    with open(train_dir, 'r', encoding='utf-8') as f:
        train_dataset = Dataset.from_list(json.load(f))

    with open(test_dir, 'r', encoding='utf-8') as f:
        test_dataset = Dataset.from_list(json.load(f)[:996])

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw
            solution = example.pop("answer")
            mechanism_label = _oracle_mechanism(question_raw)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "QA",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question_raw,
                    "mechanism_label": mechanism_label,  # CONDOR addition
                },
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_json(os.path.join(local_dir, "train.jsonl"))
    test_dataset.to_json(os.path.join(local_dir, "test_996.jsonl"))

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test_996.parquet"))

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
