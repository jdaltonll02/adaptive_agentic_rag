"""CONDOR data/strategy_qa.py

Prepares the StrategyQA dataset (ChilleD/StrategyQA) for CONDOR training/eval.

StrategyQA requires implicit multi-step reasoning to answer yes/no questions.
Boolean answers are converted to "yes"/"no" strings.

Splits
------
  train  : ~2290 examples
  test   : ~490 examples (capped at 996, written as test_996.parquet)

Usage
-----
    python -m data.strategy_qa \\
        --local-dir /data/user_data/jgibson2/condor/data/strategy_qa
"""

import argparse
import os
import sys

from datasets import load_dataset

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from condor.mechanism_router import classify_qtype
    _CONDOR_AVAILABLE = True
except ImportError:
    _CONDOR_AVAILABLE = False

DATA_SOURCE = "ChilleD/StrategyQA"
_DEFAULT_MECHANISM = 1  # Standard RAG (implicit reasoning, short factoid)


def _bool_to_str(answer) -> str:
    """Convert boolean answer to 'yes'/'no' string."""
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    if isinstance(answer, str):
        return answer.lower().strip()
    # Fallback for numeric or other types
    return "yes" if answer else "no"


def _oracle_mechanism(question: str) -> int:
    if not _CONDOR_AVAILABLE:
        return _DEFAULT_MECHANISM
    qtype = classify_qtype(question)
    return {0: 1, 1: 2, 2: 3, 3: 4}.get(qtype, _DEFAULT_MECHANISM)


def make_map_fn(split: str):
    def process_fn(example, idx):
        question_raw = example.get("question", "")
        answer_raw = example.get("answer", False)
        answer_str = _bool_to_str(answer_raw)
        # Wrap in list so any-match evaluator works consistently
        ground_truth = [answer_str]
        mechanism_label = _oracle_mechanism(question_raw)
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": question_raw}],
            "ability": "QA",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": ground_truth,
                "question": question_raw,
                "mechanism_label": mechanism_label,
            },
        }
    return process_fn


if __name__ == "__main__":
    _DEFAULT_OUT = "/data/user_data/jgibson2/condor/data/strategy_qa"

    parser = argparse.ArgumentParser(description="Prepare StrategyQA parquets for CONDOR.")
    parser.add_argument("--local-dir", default=_DEFAULT_OUT,
                        help="Output directory for parquet files")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip writing the train parquet")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print("[strategy_qa] Downloading ChilleD/StrategyQA …")
    raw = load_dataset(DATA_SOURCE)

    # Determine test split name (dataset may call it 'test' or 'validation')
    test_split_name = "test" if "test" in raw else "validation"

    if not args.skip_train and "train" in raw:
        train_ds = raw["train"]
        train_ds = train_ds.map(function=make_map_fn("train"), with_indices=True)
        train_out = os.path.join(args.local_dir, "train.parquet")
        train_ds.to_parquet(train_out)
        print(f"[strategy_qa] Train parquet ({len(train_ds)} rows) → {train_out}")

    test_ds = raw[test_split_name]
    test_ds = test_ds.select(range(min(996, len(test_ds))))
    test_ds = test_ds.map(function=make_map_fn("test"), with_indices=True)
    test_out = os.path.join(args.local_dir, "test_996.parquet")
    test_ds.to_parquet(test_out)
    print(f"[strategy_qa] Test parquet ({len(test_ds)} rows) → {test_out}")
    print("[strategy_qa] Done.")
