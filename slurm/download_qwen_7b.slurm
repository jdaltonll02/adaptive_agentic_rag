#!/bin/bash
#SBATCH --job-name=dl_qwen7b
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_dl_qwen7b.out
#SBATCH --error=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_dl_qwen7b.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

source ~/.bashrc
source /home/jgibson2/projects/adaptive_rag/.venv/bin/activate

export DEST=/data/user_data/jgibson2/condor/checkpoints/sft/qwen2.5_7b_instruct
mkdir -p "${DEST}"

export HF_HOME=/data/user_data/jgibson2/hf_cache
mkdir -p "${HF_HOME}"

echo "[DL] Downloading Qwen/Qwen2.5-7B-Instruct to ${DEST}"
echo "[DL] Start: $(date)"

python - <<'EOF'
import os
from huggingface_hub import snapshot_download

dest = os.environ["DEST"]
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir=dest,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.pt", "original/*"],
)
print(f"Download complete → {dest}")
EOF

echo "[DL] End: $(date)"
ls -lh "${DEST}"
