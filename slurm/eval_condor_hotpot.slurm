#!/bin/bash
#SBATCH --job-name=condor_eval_hotpot
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_eval_condor_hotpot.out
#SBATCH --error=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_eval_condor_hotpot.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

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

echo "[SLURM] Job ID        : ${SLURM_JOB_ID}"
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
