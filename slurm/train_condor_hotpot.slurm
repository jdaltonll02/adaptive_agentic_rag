#!/bin/bash
#SBATCH --job-name=condor_hotpot
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40S:5
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_condor_hotpot.out
#SBATCH --error=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_condor_hotpot.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
source .venv/bin/activate

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
BASELINE_ROOT=${PROJECT_ROOT}/baseline
export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=true
# GPUs 0,1,2,3 → training (FSDP actor/rollout/ref/critic; 2 DP×TP-2 rollout groups)
# GPU  4       → local executor vLLM server (no rate limits)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
# Disable torch.compile globally: prevents triton 3.2.0 PY_SSIZE_T_CLEAN crash on Python 3.12.
export TORCHDYNAMO_DISABLE=1
export DISABLE_FLASH_ATTN_TRITON_CE=1
# NOTE: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is intentionally NOT set.
# It conflicts with NCCL's internal cudaMalloc during FSDP sync_module_states BROADCAST.
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ENABLE_V1_MULTIPROCESSING=0
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
VRAM_GB=$(( VRAM_MB / 1024 ))
if   [ "${VRAM_GB}" -ge 70 ]; then GMU=0.40   # A100_80GB / H100 80GB
elif [ "${VRAM_GB}" -ge 40 ]; then GMU=0.85   # L40S 44GB / A100_40GB / A6000 48GB
else                                GMU=0.85   # smaller GPU
fi
echo "[SLURM] GPU: ${GPU_TYPE}  VRAM: ${VRAM_GB}GB  →  gpu_memory_utilization=${GMU}"

# Use a job-specific Ray temp dir to avoid permission conflicts with stale /tmp/ray files
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}
mkdir -p ${RAY_TMPDIR}
# Clear stale Ray / torch caches from previous jobs on this node
rm -rf /tmp/ray /tmp/torchinductor_${USER} /tmp/triton_${USER}
# Give Ray's GCS more time to start on a busy node
export RAY_wait_register_center_timeout=600

# ── Cache 7B planner checkpoint to local NVMe ─────────────────────────────────
NFS_7B=/data/user_data/jgibson2/condor/checkpoints/sft/qwen2.5_7b_instruct
LOCAL_7B=/tmp/${SLURM_JOB_ID}_qwen2.5_7b
echo "[SLURM] Copying 7B checkpoint to local NVMe: ${LOCAL_7B}"
mkdir -p "${LOCAL_7B}"
rsync -a --info=progress2 "${NFS_7B}/" "${LOCAL_7B}/"
echo "[SLURM] 7B checkpoint cached locally ($(du -sh ${LOCAL_7B} | cut -f1))"

# ── Cache retriever index to local NVMe ──────────────────────────────────────
NFS_RETRIEVER=/data/user_data/jgibson2/condor/retriever
LOCAL_RETRIEVER=/tmp/${SLURM_JOB_ID}_retriever
echo "[SLURM] Copying retriever index to local NVMe: ${LOCAL_RETRIEVER}"
mkdir -p "${LOCAL_RETRIEVER}"
rsync -a --info=progress2 \
    "${NFS_RETRIEVER}/wikipedia.e5" \
    "${NFS_RETRIEVER}/psgs_w100.tsv" \
    "${NFS_RETRIEVER}/intfloat" \
    "${LOCAL_RETRIEVER}/"
echo "[SLURM] Retriever cached locally ($(du -sh ${LOCAL_RETRIEVER} | cut -f1))"

# ── Retrieval server ──────────────────────────────────────────────────────────
export RETRIEVER_ROOT=${LOCAL_RETRIEVER}
RETRIEVAL_LOG=${PROJECT_ROOT}/log/slurm_logs/${SLURM_JOB_ID}_retrieval_server.log

python ${PROJECT_ROOT}/scripts/start_retrieval_server.py > ${RETRIEVAL_LOG} 2>&1 &
RETRIEVAL_PID=$!
trap "kill ${RETRIEVAL_PID} ${EXECUTOR_PID} 2>/dev/null; rm -rf ${LOCAL_7B} ${LOCAL_RETRIEVER}" EXIT

echo "[SLURM] Retrieval server PID: ${RETRIEVAL_PID}  (log: ${RETRIEVAL_LOG})"
echo "[SLURM] Waiting for retrieval server on port 8000 (FAISS index load ~25 min)..."
RETRIEVAL_READY=0
for i in $(seq 1 360); do
    if curl -sf http://localhost:8000/search -d '{"questions":["test"],"N":1}' -H 'Content-Type: application/json' > /dev/null 2>&1; then
        echo "[SLURM] Retrieval server ready (${i}s)"
        RETRIEVAL_READY=1
        break
    fi
    if ! kill -0 ${RETRIEVAL_PID} 2>/dev/null; then
        echo "[SLURM] ERROR: Retrieval server died. Check ${RETRIEVAL_LOG}"
        exit 1
    fi
    sleep 5
done
if [ ${RETRIEVAL_READY} -eq 0 ]; then
    echo "[SLURM] ERROR: Retrieval server did not become ready within 1800s. Check ${RETRIEVAL_LOG}"
    exit 1
fi

# ── Local executor vLLM server (GPU 2, port 8001) ─────────────────────────────
EXECUTOR_CKPT=/data/user_data/jgibson2/condor/checkpoints/sft/qwen2.5_14b_instruct
EXECUTOR_LOG=${PROJECT_ROOT}/log/slurm_logs/${SLURM_JOB_ID}_executor_vllm.log

CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "${EXECUTOR_CKPT}" \
    --served-model-name llama-3.1-8b-instant \
    --port 8001 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --max-model-len 8192 \
    > ${EXECUTOR_LOG} 2>&1 &
EXECUTOR_PID=$!

echo "[SLURM] Executor vLLM PID: ${EXECUTOR_PID}  (log: ${EXECUTOR_LOG})"
echo "[SLURM] Waiting for executor vLLM server on port 8001 (model fully loaded)..."
EXECUTOR_READY=0
for i in $(seq 1 240); do
    if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "[SLURM] Executor vLLM server ready (${i}s)"
        EXECUTOR_READY=1
        break
    fi
    if ! kill -0 ${EXECUTOR_PID} 2>/dev/null; then
        echo "[SLURM] ERROR: Executor vLLM server died. Check ${EXECUTOR_LOG}"
        exit 1
    fi
    sleep 5
done
if [ ${EXECUTOR_READY} -eq 0 ]; then
    echo "[SLURM] ERROR: Executor vLLM server did not become ready within 1200s. Check ${EXECUTOR_LOG}"
    exit 1
fi

# Point executor agents at local vLLM — no rate limits, no external API
export OPENAI_API_BASE="http://localhost:8001/v1"
export OPENAI_API_KEY="local"

# Restrict training to GPUs 0–3; GPU 4 is the executor
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "[SLURM] Job ID        : ${SLURM_JOB_ID}"
echo "[SLURM] Node          : $(hostname)"
echo "[SLURM] Training GPUs : 0,1,2,3  Executor GPU: 4"
echo "[SLURM] Config        : condor_hotpot"
echo "[SLURM] Start time    : $(date)"

# ── Stable run ID across preempt requeues ────────────────────────────────────
# SLURM_JOB_ID changes on every requeue; store the original ID in an NFS file
# so the checkpoint directory path stays constant across restarts.
# To start a FRESH run: delete /data/user_data/jgibson2/condor/logs/hotpot_run_id
RUN_ID_FILE="/data/user_data/jgibson2/condor/logs/hotpot_run_id"
mkdir -p "$(dirname ${RUN_ID_FILE})"
if [ ! -f "${RUN_ID_FILE}" ]; then
    echo "${SLURM_JOB_ID}" > "${RUN_ID_FILE}"
    echo "[SLURM] Fresh run — stored run ID ${SLURM_JOB_ID} → ${RUN_ID_FILE}"
fi
ORIGINAL_JOB_ID=$(cat "${RUN_ID_FILE}")
EXPERIMENT_NAME="condor_qwen2.5_7b_hotpot_${ORIGINAL_JOB_ID}"
CKPT_BASE="/data/user_data/jgibson2/condor/checkpoints/rl/verl_condor_hotpot/${EXPERIMENT_NAME}"
echo "[SLURM] Run ID: ${ORIGINAL_JOB_ID}  (current SLURM_JOB_ID=${SLURM_JOB_ID})"

# ── Auto-resume: scan stable checkpoint dir for latest global_step ───────────
RESUME_ARGS="trainer.resume_mode=disable"
LATEST_CKPT=$(ls -d "${CKPT_BASE}/global_step_"* 2>/dev/null \
    | sort -V | tail -1)
if [ -n "${LATEST_CKPT}" ]; then
    echo "[SLURM] Requeue detected — resuming from: ${LATEST_CKPT}"
    RESUME_ARGS="trainer.resume_mode=resume_path trainer.resume_from_path=${LATEST_CKPT}"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
cd ${PROJECT_ROOT}

python -m verl.trainer.main_ppo_condor \
    --config-path "${PROJECT_ROOT}/configs" \
    --config-name  condor_hotpot \
    "wandb.project=verl_condor_hotpot" \
    "wandb.experiment=${EXPERIMENT_NAME}" \
    "paths.sft_checkpoint=${LOCAL_7B}" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GMU}" \
    "trainer.default_local_dir=${CKPT_BASE}" \
    ${RESUME_ARGS} \
    "$@"

EXIT_CODE=$?
echo "[SLURM] End time: $(date)  Exit code: ${EXIT_CODE}"

# Clean exit → remove run ID file so next sbatch starts a fresh experiment.
# Preemption exits non-zero, so the file is preserved for the requeue.
[ ${EXIT_CODE} -eq 0 ] && rm -f "${RUN_ID_FILE}" && echo "[SLURM] Run complete — cleared ${RUN_ID_FILE}"

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[SLURM] === Ray worker logs (last 150 lines each) ==="
    find ${RAY_TMPDIR} -name "worker-*.log" -newer /proc/1/cmdline 2>/dev/null \
        | head -8 \
        | while read f; do
            echo "--- ${f} ---"
            tail -150 "${f}" 2>/dev/null
          done
fi

exit ${EXIT_CODE}
