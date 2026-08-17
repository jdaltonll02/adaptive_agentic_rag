#!/bin/bash
# run_condor.sh
#
# Lightweight development launcher — use the SLURM scripts for real training.
#
# Usage
#   bash run_condor.sh                        # NQ, CONDOR on
#   bash run_condor.sh --config-name condor_hotpot
#   bash run_condor.sh "condor.use_condor=false"       # exact baseline
#   bash run_condor.sh "data.train_batch_size=32"      # Hydra override
#
# All parameters live in configs/condor_nq.yaml (or condor_hotpot.yaml).
# Do not add parameter flags here — edit the YAML instead.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
BASELINE_ROOT="${PROJECT_ROOT}/baseline"

export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true

# LLM API credentials (CMU gateway)
export OPENAI_API_BASE="${OPENAI_BASE_URL:-https://ai-gateway.andrew.cmu.edu}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

python -m verl.trainer.main_ppo_condor \
    --config-path "${PROJECT_ROOT}/configs" \
    --config-name  condor_nq \
    "wandb.experiment=condor_nq_dev_${TIMESTAMP}" \
    "$@"
