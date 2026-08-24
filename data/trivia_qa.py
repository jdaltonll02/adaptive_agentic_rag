"""CONDOR data/trivia_qa.py

Prepares TriviaQA open-domain (mandarjoshi/trivia_qa, rc.nocontext config) for
CONDOR training and evaluation.

The rc.nocontext config strips evidence passages so the model must retrieve
supporting facts itself — matching the open-domain setting.

Splits
------
  train      : full train split   → train.parquet
  validation : used as eval split → test_996.parquet (capped at 996)
  (no public test answers in the original dataset)

Ground truth
------------
  answer.aliases covers all valid answer surface forms and is used as the
  ground_truth list so the any-match F1 evaluator can credit any correct form.

Usage
-----
    python -m data.trivia_qa \\
        --local-dir /data/user_data/jgibson2/condor/data/trivia_qa
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

DATA_SOURCE = "mandarjoshi/trivia_qa"
HF_CONFIG = "rc.nocontext"
_DEFAULT_MECHANISM = 1  # Standard RAG (factoid questions)


def _oracle_mechanism(question: str) -> int:
    if not _CONDOR_AVAILABLE:
        return _DEFAULT_MECHANISM
    qtype = classify_qtype(question)
    return {0: 1, 1: 2, 2: 3, 3: 4}.get(qtype, _DEFAULT_MECHANISM)


def _extract_aliases(answer_field) -> list:
    """Extract all valid answer strings from the nested answer dict."""
    if isinstance(answer_field, dict):
        aliases = answer_field.get("aliases", [])
        value = answer_field.get("value", "")
        # Combine: aliases first, then the canonical value if not already present
        all_answers = list(aliases) if aliases else []
        if value and value not in all_answers:
            all_answers.append(value)
        return all_answers if all_answers else [""]
    if isinstance(answer_field, str):
        return [answer_field] if answer_field else [""]
    return [""]


def make_map_fn(split: str):
    def process_fn(example, idx):
        question_raw = example.get("question", "")
        answer_field = example.get("answer", {})
        ground_truth = _extract_aliases(answer_field)
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
    _DEFAULT_OUT = "/data/user_data/jgibson2/condor/data/trivia_qa"

    parser = argparse.ArgumentParser(description="Prepare TriviaQA parquets for CONDOR.")
    parser.add_argument("--local-dir", default=_DEFAULT_OUT,
                        help="Output directory for parquet files")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip writing the train parquet")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print(f"[trivia_qa] Downloading {DATA_SOURCE} (config={HF_CONFIG}) …")
    raw = load_dataset(DATA_SOURCE, HF_CONFIG)

    if not args.skip_train and "train" in raw:
        train_ds = raw["train"]
        train_ds = train_ds.map(function=make_map_fn("train"), with_indices=True)
        train_out = os.path.join(args.local_dir, "train.parquet")
        train_ds.to_parquet(train_out)
        print(f"[trivia_qa] Train parquet ({len(train_ds)} rows) → {train_out}")

    # Use validation split as the eval test set (no public test answers)
    val_ds = raw["validation"]
    val_ds = val_ds.select(range(min(996, len(val_ds))))
    val_ds = val_ds.map(function=make_map_fn("validation"), with_indices=True)
    test_out = os.path.join(args.local_dir, "test_996.parquet")
    val_ds.to_parquet(test_out)
    print(f"[trivia_qa] Val→test parquet ({len(val_ds)} rows) → {test_out}")
    print("[trivia_qa] Done.")
