"""CONDOR data/musique.py

Prepares the MuSiQue (Multi-hop Questions via Single-hop Question Composition)
benchmark for CONDOR evaluation.

MuSiQue is a hard multi-hop QA benchmark.  The official test split withholds
gold answers, so we use the validation split as our held-out eval set (same
convention as KILT-NQ).  Only answerable questions are kept.

HuggingFace dataset: bdsaglam/musique
  - answerable / unanswerable variants; we use answerable (musique_ans).
  - Fields: id, paragraphs, question, question_decomposition, answer,
            answer_aliases, answerable

Usage
-----
    python -m data.musique \\
        --local-dir /data/user_data/jgibson2/condor/data/musique
"""

import argparse
import os
import sys

from datasets import load_dataset

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from condor.mechanism_router import classify_qtype
    _CONDOR_AVAILABLE = True
except ImportError:
    _CONDOR_AVAILABLE = False

DATA_SOURCE = "bdsaglam/musique"
HF_CONFIG = "answerable"
_DEFAULT_MECHANISM = 2  # Advanced Iterative RAG — MuSiQue is always multi-hop


def _oracle_mechanism(question: str) -> int:
    if not _CONDOR_AVAILABLE:
        return _DEFAULT_MECHANISM
    qtype = classify_qtype(question)
    return {0: 1, 1: 2, 2: 3, 3: 4}.get(qtype, _DEFAULT_MECHANISM)


def make_map_fn(split: str):
    def process_fn(example, idx):
        question_raw = example.get("question", "")

        # Ground-truth answers: primary answer + all aliases
        answer_main = example.get("answer", "")
        aliases = example.get("answer_aliases", []) or []
        if isinstance(aliases, str):
            aliases = [aliases]
        ground_truth = list(dict.fromkeys(
            [a for a in [answer_main] + list(aliases) if a]
        ))
        if not ground_truth:
            ground_truth = [""]

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
    _DEFAULT_OUT = "/data/user_data/jgibson2/condor/data/musique"

    parser = argparse.ArgumentParser(description="Prepare MuSiQue parquet for CONDOR eval.")
    parser.add_argument("--local-dir", default=_DEFAULT_OUT,
                        help="Output directory for parquet files")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print(f"[musique] Downloading {DATA_SOURCE} (config={HF_CONFIG}) …")
    raw = load_dataset(DATA_SOURCE, HF_CONFIG)

    # Train split
    if "train" in raw:
        train_ds = raw["train"]
        train_ds = train_ds.map(function=make_map_fn("train"), with_indices=True)
        train_out = os.path.join(args.local_dir, "train.parquet")
        train_ds.to_parquet(train_out)
        print(f"[musique] Train parquet ({len(train_ds)} rows) → {train_out}")

    # Validation split used as held-out test (official test has no gold answers)
    val_split = "validation" if "validation" in raw else "test"
    test_ds = raw[val_split]
    test_ds = test_ds.select(range(min(996, len(test_ds))))
    test_ds = test_ds.map(function=make_map_fn("test"), with_indices=True)
    test_out = os.path.join(args.local_dir, "test_996.parquet")
    test_ds.to_parquet(test_out)
    print(f"[musique] Test parquet ({len(test_ds)} rows, from {val_split} split) → {test_out}")
    print("[musique] Done.")
