#!/bin/bash
#SBATCH --job-name=condor_eval_nq
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/user_data/jgibson2/condor/logs/slurm/%j_eval_condor_nq.out
#SBATCH --error=/data/user_data/jgibson2/condor/logs/slurm/%j_eval_condor_nq.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

# Usage
#   sbatch slurm/eval_condor_nq.slurm
#   sbatch slurm/eval_condor_nq.slurm "eval.checkpoint_path=/data/.../condor/checkpoints/<id>"
#
# All parameters in configs/eval_condor_nq.yaml; override on the command line
# by passing quoted Hydra overrides as arguments to sbatch.

source ~/.bashrc
conda activate condor   # update to your conda/venv environment name

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
BASELINE_ROOT=${PROJECT_ROOT}/baseline
export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0

echo "[SLURM] Job ID        : ${SLURM_JOB_ID}"
echo "[SLURM] Node          : $(hostname)"
echo "[SLURM] GPUs          : ${CUDA_VISIBLE_DEVICES}"
echo "[SLURM] Config        : eval_condor_nq"
echo "[SLURM] Start time    : $(date)"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="condor_eval_nq_${SLURM_JOB_ID}_${TIMESTAMP}"

cd ${PROJECT_ROOT}

python -m verl.trainer.main_eval_condor \
    --config-path "${PROJECT_ROOT}/configs" \
    --config-name  eval_condor_nq \
    "wandb.experiment=${EXPERIMENT_NAME}" \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    "$@"

EXIT_CODE=$?
echo "[SLURM] End time: $(date)  Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
