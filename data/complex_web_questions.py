"""CONDOR data/complex_web_questions.py

Prepares the ComplexWebQuestions dataset (rmanluo/RoG-cwq) for CONDOR training/eval.

CWQ questions are compositional and SPARQL-derived, often requiring multi-entity
reasoning over a knowledge graph.  Default mechanism is 3 (Graph RAG).

Splits
------
  train  : full train split
  test   : test split capped at 996, written as test_996.parquet

Usage
-----
    python -m data.complex_web_questions \\
        --local-dir /data/user_data/jgibson2/condor/data/cwq
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

DATA_SOURCE = "rmanluo/RoG-cwq"
_DEFAULT_MECHANISM = 3  # Graph RAG (SPARQL-derived compositional questions)


def _oracle_mechanism(question: str) -> int:
    if not _CONDOR_AVAILABLE:
        return _DEFAULT_MECHANISM
    qtype = classify_qtype(question)
    return {0: 1, 1: 2, 2: 3, 3: 4}.get(qtype, _DEFAULT_MECHANISM)


def _normalize_answers(answers) -> list:
    """Ensure answers is a non-empty list of strings."""
    if isinstance(answers, str):
        return [answers] if answers else [""]
    if isinstance(answers, list):
        result = [str(a) for a in answers if a is not None]
        return result if result else [""]
    return [""]


def make_map_fn(split: str):
    def process_fn(example, idx):
        question_raw = example.get("question", "")
        answers_raw = example.get("answers", [])
        ground_truth = _normalize_answers(answers_raw)
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
    _DEFAULT_OUT = "/data/user_data/jgibson2/condor/data/cwq"

    parser = argparse.ArgumentParser(description="Prepare CWQ parquets for CONDOR.")
    parser.add_argument("--local-dir", default=_DEFAULT_OUT,
                        help="Output directory for parquet files")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip writing the train parquet")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print("[cwq] Downloading rmanluo/RoG-cwq …")
    raw = load_dataset(DATA_SOURCE)

    if not args.skip_train and "train" in raw:
        train_ds = raw["train"]
        train_ds = train_ds.map(function=make_map_fn("train"), with_indices=True)
        train_out = os.path.join(args.local_dir, "train.parquet")
        train_ds.to_parquet(train_out)
        print(f"[cwq] Train parquet ({len(train_ds)} rows) → {train_out}")

    test_split_name = "test" if "test" in raw else list(raw.keys())[-1]
    test_ds = raw[test_split_name]
    test_ds = test_ds.select(range(min(996, len(test_ds))))
    test_ds = test_ds.map(function=make_map_fn("test"), with_indices=True)
    test_out = os.path.join(args.local_dir, "test_996.parquet")
    test_ds.to_parquet(test_out)
    print(f"[cwq] Test parquet ({len(test_ds)} rows) → {test_out}")
    print("[cwq] Done.")
