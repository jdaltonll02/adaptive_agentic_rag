#!/bin/bash
#SBATCH --job-name=condor_eval_all
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_eval_all.out
#SBATCH --error=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_eval_all.err
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
source .venv/bin/activate

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
BASELINE_ROOT=${PROJECT_ROOT}/baseline
export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true

# LLM API credentials (CMU gateway)
export OPENAI_API_BASE="${OPENAI_BASE_URL:-https://ai-gateway.andrew.cmu.edu}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export CUDA_VISIBLE_DEVICES=0

# ── Retrieval server ──────────────────────────────────────────────────────────
export RETRIEVER_ROOT=/data/user_data/jgibson2/condor/retriever
RETRIEVAL_LOG=/tmp/flask_server_${SLURM_JOB_ID}.log
python ${PROJECT_ROOT}/scripts/start_retrieval_server.py > ${RETRIEVAL_LOG} 2>&1 &
RETRIEVAL_PID=$!
trap "kill ${RETRIEVAL_PID} 2>/dev/null" EXIT
echo "[SLURM] Retrieval server PID: ${RETRIEVAL_PID}  (log: ${RETRIEVAL_LOG})"
echo "[SLURM] Waiting for retrieval server on port 8000..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/search -d '{"questions":["test"],"N":1}' -H 'Content-Type: application/json' > /dev/null 2>&1; then
        echo "[SLURM] Retrieval server ready (${i}s)"; break
    fi
    if ! kill -0 ${RETRIEVAL_PID} 2>/dev/null; then
        echo "[SLURM] ERROR: Retrieval server died. Check ${RETRIEVAL_LOG}"; exit 1
    fi
    sleep 5
done

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
