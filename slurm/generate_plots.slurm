#!/bin/bash
#SBATCH --job-name=condor_plots
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/data/user_data/jgibson2/condor/logs/slurm/%j_generate_plots.out
#SBATCH --error=/data/user_data/jgibson2/condor/logs/slurm/%j_generate_plots.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

# Usage
#   sbatch slurm/generate_plots.slurm --experiment condor_nq_<jobid>_<timestamp>
#
# Required argument:
#   --experiment  <name>    prefix of the *_training.csv and *_eval_summary.json files
#
# Optional:
#   --dataset     NQ        label shown in figure titles (default empty)
#   --smooth      20        training curve smoothing window (default 20)

source ~/.bashrc
conda activate condor

PROJECT_ROOT=/home/jgibson2/projects/adaptive_rag
BASELINE_ROOT=${PROJECT_ROOT}/baseline
export PYTHONPATH="${PROJECT_ROOT}:${BASELINE_ROOT}:${PYTHONPATH:-}"

METRICS_DIR="${PROJECT_ROOT}/outputs/metrics"
PLOTS_DIR="${PROJECT_ROOT}/outputs/plots"

echo "[SLURM] Job ID     : ${SLURM_JOB_ID}"
echo "[SLURM] Node       : $(hostname)"
echo "[SLURM] Start time : $(date)"

cd ${PROJECT_ROOT}

python -m condor.plots.generate_all \
    --metrics-dir "${METRICS_DIR}" \
    --plots-dir   "${PLOTS_DIR}" \
    "$@"

EXIT_CODE=$?
echo "[SLURM] End time: $(date)  Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
