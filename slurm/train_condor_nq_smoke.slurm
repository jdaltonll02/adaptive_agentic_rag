#!/bin/bash
#SBATCH --job-name=condor_nq_smoke
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100_80GB:3
#SBATCH --constraint="A100_80GB|A100_80G"
#SBATCH --mem=400G
#SBATCH --exclude=babel-w5-24
#SBATCH --time=4:00:00
#SBATCH --output=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_condor_nq_smoke.out
#SBATCH --error=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_condor_nq_smoke.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
source .venv/bin/activate

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
BASELINE_ROOT=${PROJECT_ROOT}/baseline
export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=true
# GPUs 0,1 → training (FSDP actor/rollout/ref/critic)
# GPU  2   → local executor vLLM server
export CUDA_VISIBLE_DEVICES=0,1,2
# Disable torch.compile: prevents triton 3.2.0 PY_SSIZE_T_CLEAN crash on Python 3.12.
export TORCHDYNAMO_DISABLE=1
# Disable flash_attn Triton cross-entropy kernel — falls back to pure PyTorch.
export DISABLE_FLASH_ATTN_TRITON_CE=1
# Flush Python stdout immediately so SLURM log reflects real-time progress
export PYTHONUNBUFFERED=1
# NOTE: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is intentionally NOT set here.
# With CPUOffload(True) on the ref model, FSDP must temporarily move CPU params to GPU
# for the sync_module_states NCCL BROADCAST. Expandable segments conflict with NCCL's
# internal cudaMalloc calls, causing rank 0 to fail silently and rank 1 to time out.
# Targets 48-80 GiB GPUs. GMU is set per-GPU-type below:
#   A100_80GB/H100: GMU=0.4 → 32 GiB vLLM budget (plenty of headroom)
#   L40S/L40/A6000/RTX_PRO_6000: GMU=0.6 → 28.8 GiB vLLM budget (tight but sufficient)
# After sleep(), vLLM weights+KV return to CuMem pool for FSDP Adam state init.
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "[SLURM] GPU: ${GPU_TYPE}  →  gpu_memory_utilization=0.4"
GMU=0.4
# Show "Sleep mode freed X GiB" log from vLLM workers in Ray's logs.
export VLLM_LOGGING_LEVEL=INFO
# external_launcher sets this to 0 automatically (vllm/config.py:1726), but be explicit.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Job-specific Ray temp dir to avoid stale /tmp/ray permission conflicts
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}
mkdir -p ${RAY_TMPDIR}
rm -rf /tmp/torchinductor_${USER} /tmp/triton_${USER}

# ── Cache 7B planner checkpoint to local NVMe ─────────────────────────────────
# The same 7B checkpoint is loaded 3× (actor, ref, critic). NFS is intermittently
# slow and causes NCCL hangs during FSDP init. Copy once to local disk so all
# three loads hit fast NVMe instead of NFS.
NFS_7B=/data/user_data/jgibson2/condor/checkpoints/sft/qwen2.5_7b_instruct
LOCAL_7B=/tmp/${SLURM_JOB_ID}_qwen2.5_7b
echo "[SLURM] Copying 7B checkpoint to local NVMe: ${LOCAL_7B}"
mkdir -p "${LOCAL_7B}"
rsync -a --info=progress2 "${NFS_7B}/" "${LOCAL_7B}/"
echo "[SLURM] 7B checkpoint cached locally ($(du -sh ${LOCAL_7B} | cut -f1))"

# ── Retrieval server ──────────────────────────────────────────────────────────
export RETRIEVER_ROOT=/data/user_data/jgibson2/condor/retriever
RETRIEVAL_LOG=${PROJECT_ROOT}/log/slurm_logs/${SLURM_JOB_ID}_retrieval_server.log

python ${PROJECT_ROOT}/scripts/start_retrieval_server.py > ${RETRIEVAL_LOG} 2>&1 &
RETRIEVAL_PID=$!
trap "kill ${RETRIEVAL_PID} ${EXECUTOR_PID} 2>/dev/null; rm -rf ${LOCAL_7B}" EXIT

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
# Executor: Qwen2.5-14B-Instruct (frozen); Planner: Qwen2.5-7B-Instruct (trained)
EXECUTOR_CKPT=/data/user_data/jgibson2/condor/checkpoints/sft/qwen2.5_14b_instruct
EXECUTOR_LOG=${PROJECT_ROOT}/log/slurm_logs/${SLURM_JOB_ID}_executor_vllm.log

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
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
for i in $(seq 1 240); do
    if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "[SLURM] Executor vLLM server ready (${i}s)"
        break
    fi
    if ! kill -0 ${EXECUTOR_PID} 2>/dev/null; then
        echo "[SLURM] ERROR: Executor vLLM server died. Check ${EXECUTOR_LOG}"
        exit 1
    fi
    sleep 5
done

# Point executor agents at local vLLM
export OPENAI_API_BASE="http://localhost:8001/v1"
export OPENAI_API_KEY="local"

# Restrict training to GPUs 0,1 only
export CUDA_VISIBLE_DEVICES=0,1

echo "[SLURM] ============================================================"
echo "[SLURM] SMOKE TEST — 3 training steps, batch_size=16"
echo "[SLURM] Job ID        : ${SLURM_JOB_ID}"
echo "[SLURM] Node          : $(hostname)"
echo "[SLURM] Training GPUs : ${CUDA_VISIBLE_DEVICES}"
echo "[SLURM] Config        : condor_nq  (+smoke overrides)"
echo "[SLURM] Start time    : $(date)"
echo "[SLURM] ============================================================"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="condor_qwen2.5_7b_nq_smoke_${SLURM_JOB_ID}_${TIMESTAMP}"

# ── Launch ────────────────────────────────────────────────────────────────────
cd ${PROJECT_ROOT}

# Smoke-test overrides (all passed as Hydra CLI args on top of condor_nq.yaml):
#   batch_size=16 so GRPO normalisation still has > 1 sample per group
#   total_training_steps=3 — run exactly 3 gradient steps and exit
#   val_before_train=false — skip the 50-min pre-train validation pass
#   test_freq=3 / save_freq=9999 — validate once at the end, no checkpoint saves
python -m verl.trainer.main_ppo_condor \
    --config-path "${PROJECT_ROOT}/configs" \
    --config-name  condor_nq \
    "wandb.project=verl_condor_nq_smoke" \
    "wandb.experiment=${EXPERIMENT_NAME}" \
    "paths.sft_checkpoint=${LOCAL_7B}" \
    "data.train_batch_size=16" \
    "data.val_batch_size=16" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=16" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.ref.ppo_mini_batch_size=16" \
    "actor_rollout_ref.ref.ppo_micro_batch_size_per_gpu=2" \
    "critic.ppo_mini_batch_size=16" \
    "critic.ppo_micro_batch_size_per_gpu=2" \
    "critic.forward_micro_batch_size_per_gpu=2" \
    "trainer.total_training_steps=3" \
    "trainer.val_before_train=false" \
    "trainer.test_freq=3" \
    "trainer.save_freq=9999" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GMU}" \
    "$@"

EXIT_CODE=$?
echo "[SLURM] End time: $(date)  Exit code: ${EXIT_CODE}"

# On failure, dump the last 150 lines of each Ray worker log so OOM/error details
# (including vLLM "Sleep mode freed X GiB" messages) appear in the SLURM output.
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
