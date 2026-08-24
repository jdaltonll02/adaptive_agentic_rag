"""CONDOR data/frames.py

Prepares the FRAMES benchmark (google/frames-benchmark) for CONDOR evaluation.

FRAMES is a multi-hop QA benchmark — no public train split exists, so only the
test parquet is written.  Each question typically requires 2–15 reasoning hops,
so the default mechanism label is 2 (Advanced Iterative RAG).

Usage
-----
    python -m data.frames \\
        --local-dir /data/user_data/jgibson2/condor/data/frames
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

DATA_SOURCE = "google/frames-benchmark"
_DEFAULT_MECHANISM = 2  # Advanced Iterative RAG (multi-hop)


def _oracle_mechanism(question: str) -> int:
    if not _CONDOR_AVAILABLE:
        return _DEFAULT_MECHANISM
    qtype = classify_qtype(question)
    # qtype 1 → multi-hop; map to mechanism 2 by default
    return {0: 1, 1: 2, 2: 3, 3: 4}.get(qtype, _DEFAULT_MECHANISM)


def make_map_fn(split: str):
    def process_fn(example, idx):
        question_raw = example.get("Prompt", "")
        answer_raw = example.get("Answer", "")
        mechanism_label = _oracle_mechanism(question_raw)
        # FRAMES answers can be multi-sentence; wrap in a list so the evaluator
        # uses any-match F1 consistently.
        ground_truth = [answer_raw] if answer_raw else [""]
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
    _DEFAULT_OUT = "/data/user_data/jgibson2/condor/data/frames"

    parser = argparse.ArgumentParser(description="Prepare FRAMES parquet for CONDOR eval.")
    parser.add_argument("--local-dir", default=_DEFAULT_OUT,
                        help="Output directory for parquet files")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print("[frames] Downloading google/frames-benchmark …")
    raw = load_dataset(DATA_SOURCE)

    # FRAMES only has a test split; use it directly.
    test_split_name = "test" if "test" in raw else list(raw.keys())[0]
    test_ds = raw[test_split_name]
    test_ds = test_ds.select(range(min(996, len(test_ds))))

    test_ds = test_ds.map(function=make_map_fn("test"), with_indices=True)

    out_path = os.path.join(args.local_dir, "test_996.parquet")
    test_ds.to_parquet(out_path)
    print(f"[frames] Test parquet ({len(test_ds)} rows) → {out_path}")
    print("[frames] Done.")
