"""Background watcher: upload actor model weights to HuggingFace Hub at milestone steps.

Runs alongside the training job. Polls the checkpoint directory every
--poll_interval seconds; when a milestone step's actor/model_*.pt files
appear, uploads them to the HF repo (skipping optim and extra_state files,
which are too large and only needed for resumption on NFS).

Usage (launched by SLURM script):
    python scripts/hf_milestone_backup.py \
        --ckpt_dir /path/to/experiment \
        --experiment condor_qwen2.5_7b_hotpot_gae_JOBID \
        --repo_id jdaltonII02/condor-hotpot-gae \
        --milestones 100 200 300 \
        --poll_interval 120
"""
import argparse
import sys
import time
from pathlib import Path


def upload_step(api, step: int, actor_dir: Path, repo_id: str, experiment: str):
    path_in_repo = f"{experiment}/global_step_{step}/actor"
    print(f"[HF-BACKUP] Uploading global_step_{step} → {repo_id}/{path_in_repo}", flush=True)
    api.upload_folder(
        folder_path=str(actor_dir),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type="model",
        ignore_patterns=["optim_*", "extra_state_*"],
    )
    print(f"[HF-BACKUP] global_step_{step} uploaded successfully.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="Experiment checkpoint base dir")
    parser.add_argument("--experiment", required=True, help="Experiment name (for HF path)")
    parser.add_argument("--repo_id", required=True, help="HuggingFace repo id (user/repo)")
    parser.add_argument("--milestones", nargs="+", type=int, default=[100, 200, 300])
    parser.add_argument("--poll_interval", type=int, default=120, help="Seconds between polls")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[HF-BACKUP] Warning: could not create repo: {e}", flush=True)

    ckpt_base = Path(args.ckpt_dir)
    remaining = set(args.milestones)
    print(f"[HF-BACKUP] Watching {ckpt_base} for steps {sorted(remaining)}", flush=True)

    while remaining:
        for step in sorted(remaining):
            actor_dir = ckpt_base / f"global_step_{step}" / "actor"
            marker = ckpt_base / f"global_step_{step}" / ".hf_uploaded"

            if marker.exists():
                remaining.discard(step)
                continue

            model_files = list(actor_dir.glob("model_world_size_*.pt")) if actor_dir.exists() else []
            if not model_files:
                continue

            try:
                upload_step(api, step, actor_dir, args.repo_id, args.experiment)
                marker.touch()
                remaining.discard(step)
            except Exception as e:
                print(f"[HF-BACKUP] ERROR uploading step {step}: {e}", flush=True)

        if remaining:
            time.sleep(args.poll_interval)

    print("[HF-BACKUP] All milestones uploaded. Exiting.", flush=True)


if __name__ == "__main__":
    main()
