#!/bin/bash
#SBATCH --job-name=condor_bench_setup
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_bench_setup.out
#SBATCH --error=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_bench_setup.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

# Downloads and prepares six benchmark datasets for CONDOR evaluation
# (and future training).  CPU-only job — no GPU required.
#
# Datasets prepared:
#   1. FRAMES          → condor/data/frames/test_996.parquet
#   2. StrategyQA      → condor/data/strategy_qa/{train,test_996}.parquet
#   3. CWQ             → condor/data/cwq/{train,test_996}.parquet
#   4. TriviaQA        → condor/data/trivia_qa/{train,test_996}.parquet
#   5. KILT-NQ         → condor/data/kilt_nq/test_996.parquet
#   6. MuSiQue         → condor/data/musique/{train,test_996}.parquet
#
# Usage
#   sbatch slurm/setup_new_benchmarks.slurm

export PYTHONUNBUFFERED=1

source ~/.bashrc
source /home/jgibson2/projects/adaptive_rag/.venv/bin/activate

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/baseline:${PYTHONPATH:-}"

CONDOR_DATA=/data/user_data/jgibson2/condor/data

echo "[setup] Job ID : ${SLURM_JOB_ID}"
echo "[setup] Node   : $(hostname)"
echo "[setup] Start  : $(date)"

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p "${CONDOR_DATA}/frames"
mkdir -p "${CONDOR_DATA}/strategy_qa"
mkdir -p "${CONDOR_DATA}/cwq"
mkdir -p "${CONDOR_DATA}/trivia_qa"
mkdir -p "${CONDOR_DATA}/kilt_nq"
mkdir -p "${CONDOR_DATA}/musique"

cd "${PROJECT_ROOT}"

OVERALL_EXIT=0

# ── 1. FRAMES ─────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "[setup] 1/6  FRAMES (google/frames-benchmark)"
echo "========================================"
python -m data.frames --local-dir "${CONDOR_DATA}/frames"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "[setup] WARNING: frames exited with code ${STATUS}"
    OVERALL_EXIT=${STATUS}
else
    echo "[setup] frames OK"
fi

# ── 2. StrategyQA ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "[setup] 2/6  StrategyQA (ChilleD/StrategyQA)"
echo "========================================"
python -m data.strategy_qa --local-dir "${CONDOR_DATA}/strategy_qa"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "[setup] WARNING: strategy_qa exited with code ${STATUS}"
    OVERALL_EXIT=${STATUS}
else
    echo "[setup] strategy_qa OK"
fi

# ── 3. ComplexWebQuestions ────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "[setup] 3/6  ComplexWebQuestions (rmanluo/RoG-cwq)"
echo "========================================"
python -m data.complex_web_questions --local-dir "${CONDOR_DATA}/cwq"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "[setup] WARNING: cwq exited with code ${STATUS}"
    OVERALL_EXIT=${STATUS}
else
    echo "[setup] cwq OK"
fi

# ── 4. TriviaQA ───────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "[setup] 4/6  TriviaQA (mandarjoshi/trivia_qa, rc.nocontext)"
echo "========================================"
python -m data.trivia_qa --local-dir "${CONDOR_DATA}/trivia_qa"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "[setup] WARNING: trivia_qa exited with code ${STATUS}"
    OVERALL_EXIT=${STATUS}
else
    echo "[setup] trivia_qa OK"
fi

# ── 5. KILT-NQ ────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "[setup] 5/6  KILT-NQ (facebook/kilt_tasks, nq)"
echo "========================================"
python -m data.kilt --local-dir "${CONDOR_DATA}/kilt_nq"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "[setup] WARNING: kilt_nq exited with code ${STATUS}"
    OVERALL_EXIT=${STATUS}
else
    echo "[setup] kilt_nq OK"
fi

# ── 6. MuSiQue ────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "[setup] 6/6  MuSiQue (bdsaglam/musique, answerable)"
echo "========================================"
python -m data.musique --local-dir "${CONDOR_DATA}/musique"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "[setup] WARNING: musique exited with code ${STATUS}"
    OVERALL_EXIT=${STATUS}
else
    echo "[setup] musique OK"
fi

echo ""
echo "[setup] End: $(date)  Overall exit: ${OVERALL_EXIT}"
exit ${OVERALL_EXIT}
