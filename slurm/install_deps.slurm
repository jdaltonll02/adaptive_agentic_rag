#!/bin/bash
#SBATCH --job-name=condor_install
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_install.out
#SBATCH --error=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_install.err

# Force line-buffered output so every line appears in the log immediately
export PYTHONUNBUFFERED=1
exec > >(stdbuf -oL tee /dev/stderr) 2>&1

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
cd ${PROJECT_ROOT}

PIP=".venv/bin/pip"
PY="PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/baseline .venv/bin/python3"

echo "[install] Start : $(date)"
echo "[install] Node  : $(hostname)"
echo "[install] Python: $(.venv/bin/python3 --version)"
echo "[install] Disk  : $(df -h /data/user_data/jgibson2 | tail -1)"

pip_install() {
    echo ""
    echo "[install] >> pip install $*"
    .venv/bin/pip install --no-cache-dir "$@"
    local code=$?
    echo "[install] << exit code: ${code}"
    return ${code}
}

# ── Step 1: install packages one at a time ────────────────────────────────────
echo ""
echo "[install] === STEP 1: individual package installs ==="

pip_install setuptools   # provides pkg_resources (needed by verl/__init__.py)
pip_install matplotlib
pip_install pandas
pip_install pytest
pip_install datasets
pip_install hydra-core
pip_install omegaconf
pip_install codetiming
pip_install fastapi uvicorn
pip_install networkx
pip_install scikit-learn
pip_install wandb
pip_install transformers
pip_install requests
pip_install tensordict
# torchdata: install with --no-build-isolation to avoid compilation hang
pip_install --no-build-isolation torchdata || echo "[install] torchdata failed — will handle at runtime"

# ── Step 2: verl editable (no-deps — vllm/torch already installed) ─────────
echo ""
echo "[install] === STEP 2: verl editable install ==="
pip_install --no-deps -e baseline/

# ── Step 3: import chain ──────────────────────────────────────────────────────
echo ""
echo "[install] === STEP 3: import chain test ==="
PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/baseline" \
  .venv/bin/python3 -c "
import sys; print('Python:', sys.version)
import verl.trainer.main_ppo_condor as m
print('main_ppo_condor: OK')
import verl.trainer.main_eval_condor
print('main_eval_condor: OK')
"
IMPORT_CODE=$?

# ── Step 4: test suite ────────────────────────────────────────────────────────
echo ""
echo "[install] === STEP 4: test suite ==="
PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/baseline" \
  .venv/bin/python3 -m pytest tests/ -q --tb=short
TEST_CODE=$?

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "[install] Done: $(date)"
echo "[install] Import exit: ${IMPORT_CODE}  |  Tests exit: ${TEST_CODE}"

if [ ${IMPORT_CODE} -eq 0 ] && [ ${TEST_CODE} -eq 0 ]; then
    echo "[install] ✓ READY — sbatch slurm/train_condor_nq.slurm"
else
    echo "[install] ✗ FAILED — check errors above"
fi
