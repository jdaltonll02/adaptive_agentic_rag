#!/bin/bash
#SBATCH --job-name=condor_eval_smoke
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=192G
#SBATCH --time=2:00:00
#SBATCH --output=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_eval_smoke.out
#SBATCH --error=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_eval_smoke.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

source ~/.bashrc
source .venv/bin/activate

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
BASELINE_ROOT=${PROJECT_ROOT}/baseline
export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true

# LLM API credentials (OpenRouter — OpenAI-compatible)
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://openrouter.ai/api/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export CONDOR_AGENT_MODEL="${CONDOR_AGENT_MODEL:-google/gemini-2.5-flash-lite}"

export CUDA_VISIBLE_DEVICES=0
export DISABLE_FLASH_ATTN_TRITON_CE=1
export PYTHONUNBUFFERED=1

# Job-specific Ray temp dir to avoid stale /tmp/ray permission conflicts on requeue
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}
mkdir -p ${RAY_TMPDIR}

# ── Retrieval server ──────────────────────────────────────────────────────────
export RETRIEVER_ROOT=/data/user_data/jgibson2/condor/retriever
RETRIEVAL_LOG=${PROJECT_ROOT}/log/slurm_logs/${SLURM_JOB_ID}_retrieval_server.log
python ${PROJECT_ROOT}/scripts/start_retrieval_server.py > ${RETRIEVAL_LOG} 2>&1 &
RETRIEVAL_PID=$!
trap "kill ${RETRIEVAL_PID} 2>/dev/null" EXIT

echo "[SLURM] Retrieval server PID: ${RETRIEVAL_PID}  (log: ${RETRIEVAL_LOG})"
echo "[SLURM] Waiting for retrieval server on port 8000 (FAISS index load ~25 min)..."
for i in $(seq 1 360); do
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

CHECKPOINT=/data/user_data/jgibson2/condor/checkpoints/rl/verl_condor_hotpot_gae/condor_qwen2.5_7b_hotpot_gae_10369967/global_step_400/actor_hf

cd ${PROJECT_ROOT}

echo "=== SMOKE TEST: NQ, 4 examples ==="

python "${BASELINE_ROOT}/verl/trainer/main_eval_condor.py" \
    --config-path "${PROJECT_ROOT}/configs" \
    --config-name  eval_condor_nq \
    "wandb.experiment=smoke_nq_${SLURM_JOB_ID}" \
    "trainer.experiment_name=smoke_nq_${SLURM_JOB_ID}" \
    "eval.checkpoint_path=${CHECKPOINT}" \
    "data.val_files=[\"/data/user_data/jgibson2/condor/data/smoke_nq_4.parquet\"]" \
    "data.train_batch_size=4" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=4"

SMOKE_EXIT=$?
echo "Smoke exit: ${SMOKE_EXIT}"
echo "[SLURM] End time: $(date)"
exit ${SMOKE_EXIT}
