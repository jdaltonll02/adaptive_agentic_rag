#!/bin/bash
#SBATCH --job-name=flash_attn_build
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_flash_attn_build.out
#SBATCH --error=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_flash_attn_build.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

export PYTHONUNBUFFERED=1
exec > >(stdbuf -oL tee /dev/stderr) 2>&1

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
cd ${PROJECT_ROOT}

source .venv/bin/activate

echo "[flash_attn] Start : $(date)"
echo "[flash_attn] Node  : $(hostname)"
echo "[flash_attn] Python: $(python3 --version)"
echo "[flash_attn] Torch : $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "[flash_attn] Disk  : $(df -h /data/user_data/jgibson2 | tail -1)"
echo ""

# Remove the stub so the real package installs cleanly
SITE=/data/user_data/jgibson2/venv_condor/lib/python3.12/site-packages
echo "[flash_attn] Removing any existing flash_attn (stub or broken build)..."
rm -rf ${SITE}/flash_attn ${SITE}/flash_attn-*.dist-info
rm -f ${SITE}/flash_attn_2_cuda*.so

echo "[flash_attn] Installing build prerequisites..."
pip install wheel packaging 2>&1

echo "[flash_attn] Clearing pip cache to force fresh build with correct ABI..."
pip cache remove flash_attn 2>/dev/null; true
pip cache remove flash-attn 2>/dev/null; true

echo "[flash_attn] Building flash-attn from source (uses ninja, ~15 min)..."
# Force old C++ ABI to match torch 2.6.0+cu124 which was built with _GLIBCXX_USE_CXX11_ABI=0
CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
MAX_JOBS=16 \
pip install flash-attn --no-build-isolation --no-cache-dir 2>&1
BUILD_CODE=$?

echo ""
echo "[flash_attn] Build exit code: ${BUILD_CODE}"

if [ ${BUILD_CODE} -eq 0 ]; then
    echo "[flash_attn] Verifying install..."
    python3 -c "
import flash_attn, importlib.metadata
v = importlib.metadata.version('flash_attn')
print('[flash_attn] Installed version:', v)
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from flash_attn import flash_attn_func, flash_attn_varlen_func
print('[flash_attn] All imports OK')
"
    echo "[flash_attn] Done : $(date)"
    echo "[flash_attn] SUCCESS — sbatch slurm/train_condor_nq.slurm"
else
    echo "[flash_attn] FAILED — check errors above"
fi
