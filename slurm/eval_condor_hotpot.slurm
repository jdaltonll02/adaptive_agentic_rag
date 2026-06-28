#!/bin/bash
#SBATCH --job-name=condor_eval_hotpot
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/user_data/jgibson2/condor/logs/slurm/%j_eval_condor_hotpot.out
#SBATCH --error=/data/user_data/jgibson2/condor/logs/slurm/%j_eval_condor_hotpot.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

source ~/.bashrc
conda activate condor

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
BASELINE_ROOT=${PROJECT_ROOT}/baseline
export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0

echo "[SLURM] Job ID        : ${SLURM_JOB_ID}"
echo "[SLURM] Node          : $(hostname)"
echo "[SLURM] GPUs          : ${CUDA_VISIBLE_DEVICES}"
echo "[SLURM] Config        : eval_condor_hotpot"
echo "[SLURM] Start time    : $(date)"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="condor_eval_hotpot_${SLURM_JOB_ID}_${TIMESTAMP}"

cd ${PROJECT_ROOT}

python -m verl.trainer.main_eval_condor \
    --config-path "${PROJECT_ROOT}/configs" \
    --config-name  eval_condor_hotpot \
    "wandb.experiment=${EXPERIMENT_NAME}" \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    "$@"

EXIT_CODE=$?
echo "[SLURM] End time: $(date)  Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
