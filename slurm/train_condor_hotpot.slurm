#!/bin/bash
#SBATCH --job-name=condor_hotpot
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40S:6
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/data/user_data/jgibson2/condor/logs/slurm/%j_condor_hotpot.out
#SBATCH --error=/data/user_data/jgibson2/condor/logs/slurm/%j_condor_hotpot.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate condor   # update to your conda/venv environment name

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
BASELINE_ROOT=${PROJECT_ROOT}/baseline
export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

echo "[SLURM] Job ID        : ${SLURM_JOB_ID}"
echo "[SLURM] Node          : $(hostname)"
echo "[SLURM] GPUs          : ${CUDA_VISIBLE_DEVICES}"
echo "[SLURM] Config        : condor_hotpot"
echo "[SLURM] Start time    : $(date)"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="condor_qwen2.5_1.5b_hotpot_${SLURM_JOB_ID}_${TIMESTAMP}"

# ── Launch ────────────────────────────────────────────────────────────────────
cd ${PROJECT_ROOT}

python -m verl.trainer.main_ppo_condor \
    --config-path "${PROJECT_ROOT}/configs" \
    --config-name  condor_hotpot \
    "wandb.experiment=${EXPERIMENT_NAME}" \
    "$@"

EXIT_CODE=$?
echo "[SLURM] End time: $(date)  Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
