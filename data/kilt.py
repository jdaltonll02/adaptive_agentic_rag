"""CONDOR data/kilt.py

Prepares KILT-NQ (kilt_tasks, nq config) for CONDOR evaluation.

Provenance titles from the KILT output field are stored in extra_info so that
the eval script can compute page-level R-precision (retrieved titles vs. gold
Wikipedia page titles).  This is the standard lightweight KILT provenance
metric used in RAG papers without the full KILT eval library.

KILT-NQ uses the same Natural Questions questions as the original NQ dataset
but maps each answer to a Wikipedia passage as provenance.

Splits
------
  test (KILT) : written as test_996.parquet (capped at 996)
  (train split available but not written — NQ train already covers this)

Usage
-----
    python -m data.kilt \\
        --local-dir /data/user_data/jgibson2/condor/data/kilt_nq
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

DATA_SOURCE = "facebook/kilt_tasks"
HF_CONFIG = "nq"
_DEFAULT_MECHANISM = 1  # Standard RAG


def _oracle_mechanism(question: str) -> int:
    if not _CONDOR_AVAILABLE:
        return _DEFAULT_MECHANISM
    qtype = classify_qtype(question)
    return {0: 1, 1: 2, 2: 3, 3: 4}.get(qtype, _DEFAULT_MECHANISM)


def _extract_answers(output_field) -> list:
    """Extract answer strings from KILT output list.

    KILT format: output is a list of dicts, each with an "answer" key that is
    itself a list of strings.  We take output[0]["answer"] per the task spec.
    """
    if not output_field:
        return [""]
    first = output_field[0] if isinstance(output_field, list) else output_field
    if isinstance(first, dict):
        answers = first.get("answer", [])
        if isinstance(answers, list):
            result = [str(a) for a in answers if a]
            return result if result else [""]
        if isinstance(answers, str):
            return [answers] if answers else [""]
    if isinstance(first, str):
        return [first] if first else [""]
    return [""]


def _extract_provenance_titles(output_field) -> list:
    """Extract gold Wikipedia page titles from KILT provenance field.

    Each output entry may have a "provenance" list of dicts with a "title" key.
    We collect all unique titles across all output entries (some questions have
    multiple valid provenance pages).
    """
    if not output_field:
        return []
    seen = set()
    titles = []
    entries = output_field if isinstance(output_field, list) else [output_field]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for prov in (entry.get("provenance") or []):
            if not isinstance(prov, dict):
                continue
            title = prov.get("title", "")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
    return titles


def make_map_fn(split: str):
    def process_fn(example, idx):
        question_raw = example.get("input", "")
        output_field = example.get("output", [])
        ground_truth = _extract_answers(output_field)
        provenance_titles = _extract_provenance_titles(output_field)
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
                "provenance_titles": provenance_titles,
            },
        }
    return process_fn


if __name__ == "__main__":
    _DEFAULT_OUT = "/data/user_data/jgibson2/condor/data/kilt_nq"

    parser = argparse.ArgumentParser(description="Prepare KILT-NQ parquet for CONDOR eval.")
    parser.add_argument("--local-dir", default=_DEFAULT_OUT,
                        help="Output directory for parquet files")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print(f"[kilt] Downloading {DATA_SOURCE} (config={HF_CONFIG}) …")
    raw = load_dataset(DATA_SOURCE, HF_CONFIG)

    # Use the validation split as test — KILT test labels are withheld.
    test_split_name = "validation" if "validation" in raw else "test"
    test_ds = raw[test_split_name]
    test_ds = test_ds.select(range(min(996, len(test_ds))))
    test_ds = test_ds.map(function=make_map_fn("test"), with_indices=True)

    test_out = os.path.join(args.local_dir, "test_996.parquet")
    test_ds.to_parquet(test_out)
    print(f"[kilt] Test parquet ({len(test_ds)} rows) → {test_out}")
    print("[kilt] Done.")
