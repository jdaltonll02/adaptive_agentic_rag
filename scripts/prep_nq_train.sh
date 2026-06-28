#!/bin/bash
# prep_nq_train.sh
#
# One-time script: generate NQ train.parquet for CONDOR from the existing
# aqawflow JSON source. The test split is already available in aqawflow/data
# so we skip it here.
#
# Run once before submitting train_condor_nq.slurm:
#   bash scripts/prep_nq_train.sh
#
# Output: /data/user_data/jgibson2/condor/data/nq/train.parquet

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_ROOT="${PROJECT_ROOT}/baseline"
export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"

AQAWFLOW=/data/user_data/jgibson2/aqawflow/data/nq
OUTPUT_DIR=/data/user_data/jgibson2/condor/data/nq

echo "[prep] Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "[prep] Generating NQ train.parquet..."
python3 -m data.nq_open \
    --train-json "${AQAWFLOW}/nq_train_questions_and_answers.json" \
    --local-dir  "${OUTPUT_DIR}" \
    --skip-test

echo "[prep] Done. Verify:"
python3 -c "
import pandas as pd
df = pd.read_parquet('${OUTPUT_DIR}/train.parquet')
print(f'  rows    : {len(df)}')
print(f'  columns : {list(df.columns)}')
print(f'  sample  : {df.iloc[0][\"extra_info\"][\"question\"][:80]}')
"
