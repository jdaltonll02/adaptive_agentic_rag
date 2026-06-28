#!/bin/bash
#SBATCH --job-name=condor_eval_all
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user_data/jgibson2/condor/logs/slurm/%j_eval_all.out
#SBATCH --error=/data/user_data/jgibson2/condor/logs/slurm/%j_eval_all.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

# Evaluates a trained CONDOR checkpoint on all 7 datasets sequentially.
#
# Usage
#   sbatch slurm/eval_condor_all_datasets.slurm
#   sbatch slurm/eval_condor_all_datasets.slurm "eval.checkpoint_path=/data/.../condor/checkpoints/rl/<id>"
#
# All 7 eval results are written to outputs/metrics/ and outputs/plots/.
# Plots and tables are generated once at the end from all collected eval summaries.

source ~/.bashrc
conda activate condor

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
BASELINE_ROOT=${PROJECT_ROOT}/baseline
export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0

echo "[SLURM] Job ID     : ${SLURM_JOB_ID}"
echo "[SLURM] Node       : $(hostname)"
echo "[SLURM] Start time : $(date)"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_EXP="condor_eval_all_${SLURM_JOB_ID}_${TIMESTAMP}"

cd ${PROJECT_ROOT}

# ── Per-dataset evaluation ────────────────────────────────────────────────────
DATASETS=(nq hotpot ambigqa 2wiki musique bamboogle popqa)
CONFIGS=(eval_condor_nq eval_condor_hotpot eval_condor_ambigqa eval_condor_2wiki eval_condor_musique eval_condor_bamboogle eval_condor_popqa)

OVERALL_EXIT=0

for i in "${!DATASETS[@]}"; do
    DS="${DATASETS[$i]}"
    CFG="${CONFIGS[$i]}"
    EXP="${BASE_EXP}_${DS}"

    echo ""
    echo "========================================"
    echo "[eval] Dataset: ${DS}  Config: ${CFG}"
    echo "========================================"

    python -m verl.trainer.main_eval_condor \
        --config-path "${PROJECT_ROOT}/configs" \
        --config-name  "${CFG}" \
        "wandb.experiment=${EXP}" \
        "trainer.experiment_name=${EXP}" \
        "$@"

    DS_EXIT=$?
    if [ ${DS_EXIT} -ne 0 ]; then
        echo "[eval] WARNING: ${DS} exited with code ${DS_EXIT}"
        OVERALL_EXIT=${DS_EXIT}
    else
        echo "[eval] ${DS} completed successfully."
    fi
done

# ── Generate all plots and tables from collected metrics ──────────────────────
echo ""
echo "[plots] Generating figures from all eval summaries..."

# Run generate_all for each experiment prefix (one per dataset)
for i in "${!DATASETS[@]}"; do
    DS="${DATASETS[$i]}"
    EXP="${BASE_EXP}_${DS}"
    TRAIN_CSV="${PROJECT_ROOT}/outputs/metrics/${EXP}_training.csv"

    python -m condor.plots.generate_all \
        --metrics-dir "${PROJECT_ROOT}/outputs/metrics" \
        --plots-dir   "${PROJECT_ROOT}/outputs/plots/${DS}" \
        --experiment  "${EXP}" \
        --dataset     "${DS^^}" || true
done

echo ""
echo "[SLURM] End time: $(date)  Overall exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}
