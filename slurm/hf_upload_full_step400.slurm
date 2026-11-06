#!/bin/bash
#SBATCH --job-name=hf_upload_step400
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_hf_upload_step400.out
#SBATCH --error=/home/jgibson2/projects/adaptive_rag/log/slurm_logs/%j_hf_upload_step400.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dalton.edu02@gmail.com

# Uploads the remaining parts of global_step_400 that the original
# hf_upload_checkpoint.py skipped (actor/optim_*, actor/extra_state_*,
# and the entire critic/).  After all uploads succeed, writes a
# .hf_full_uploaded marker.  DOES NOT delete anything — delete manually
# after verifying the marker exists.
#
# Target repo: jdaltonII02/condor-hotpot-gae
# Uploaded paths in repo:
#   condor_qwen2.5_7b_hotpot_gae_10369967/global_step_400/actor_optim/
#   condor_qwen2.5_7b_hotpot_gae_10369967/global_step_400/actor_extra/
#   condor_qwen2.5_7b_hotpot_gae_10369967/global_step_400/critic/

source ~/.bashrc
source .venv/bin/activate

CKPT=/data/user_data/jgibson2/condor/checkpoints/rl/verl_condor_hotpot_gae/condor_qwen2.5_7b_hotpot_gae_10369967/global_step_400
REPO=jdaltonII02/condor-hotpot-gae
EXPERIMENT=condor_qwen2.5_7b_hotpot_gae_10369967
STEP=global_step_400
MARKER=${CKPT}/.hf_full_uploaded

echo "[HF-UPLOAD] Job ID  : ${SLURM_JOB_ID}"
echo "[HF-UPLOAD] Node    : $(hostname)"
echo "[HF-UPLOAD] Start   : $(date)"
echo "[HF-UPLOAD] Checkpoint: ${CKPT}"
echo "[HF-UPLOAD] Repo    : ${REPO}"

if [ -f "${MARKER}" ]; then
    echo "[HF-UPLOAD] .hf_full_uploaded marker already exists — nothing to do."
    exit 0
fi

.venv/bin/python - <<'PYEOF'
import sys, os
from pathlib import Path
from huggingface_hub import HfApi

CKPT      = Path(os.environ.get("CKPT", "/data/user_data/jgibson2/condor/checkpoints/rl/verl_condor_hotpot_gae/condor_qwen2.5_7b_hotpot_gae_10369967/global_step_400"))
REPO      = "jdaltonII02/condor-hotpot-gae"
EXPERIMENT= "condor_qwen2.5_7b_hotpot_gae_10369967"
STEP      = "global_step_400"

api = HfApi()

try:
    api.create_repo(repo_id=REPO, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"[HF-UPLOAD] Warning: could not create repo: {e}", flush=True)

def upload(local_dir, path_in_repo, patterns=None):
    local_dir = Path(local_dir)
    if not local_dir.exists():
        print(f"[HF-UPLOAD] SKIP (not found): {local_dir}", flush=True)
        return
    files = list(local_dir.glob("*"))
    if patterns:
        import fnmatch
        files = [f for f in files if any(fnmatch.fnmatch(f.name, p) for p in patterns)]
    total = sum(f.stat().st_size for f in files if f.is_file()) / 1e9
    print(f"[HF-UPLOAD] Uploading {local_dir.name}/ → {REPO}/{path_in_repo}  ({total:.1f} GB)", flush=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=REPO,
        path_in_repo=path_in_repo,
        repo_type="model",
        ignore_patterns=["*.tmp"],
    )
    print(f"[HF-UPLOAD] Done: {path_in_repo}", flush=True)

# 1. Actor optimizer states (skipped by original script)
upload(
    CKPT / "actor",
    f"{EXPERIMENT}/{STEP}/actor_optim",
    patterns=["optim_world_size_*.pt", "extra_state_world_size_*.pt"],
)

# 2. Full critic directory
upload(
    CKPT / "critic",
    f"{EXPERIMENT}/{STEP}/critic",
)

# Write marker
marker = CKPT / ".hf_full_uploaded"
marker.touch()
print(f"[HF-UPLOAD] All uploads complete. Marker written: {marker}", flush=True)
print(f"[HF-UPLOAD] Repo URL: https://huggingface.co/{REPO}", flush=True)
PYEOF

UPLOAD_EXIT=$?
echo "[HF-UPLOAD] End: $(date)  Exit: ${UPLOAD_EXIT}"
exit ${UPLOAD_EXIT}
