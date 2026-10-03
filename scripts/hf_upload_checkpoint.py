"""Upload a single checkpoint's actor weights to HuggingFace Hub.

Usage:
    python scripts/hf_upload_checkpoint.py \
        --ckpt_path /path/to/global_step_260 \
        --experiment condor_qwen2.5_7b_hotpot_gae_10369967 \
        --repo_id jdaltonII02/condor-hotpot-gae

Uploads actor/model_*.pt files only (skips optim_* and extra_state_*).
Writes a .hf_uploaded marker on success so re-runs are idempotent.
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True, help="Path to global_step_N dir")
    parser.add_argument("--experiment", required=True, help="Experiment name (for HF path prefix)")
    parser.add_argument("--repo_id", required=True, help="HuggingFace repo id (user/repo)")
    parser.add_argument("--force", action="store_true", help="Re-upload even if marker exists")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path).resolve()
    if not ckpt_path.exists():
        print(f"[HF-UPLOAD] ERROR: checkpoint path does not exist: {ckpt_path}", flush=True)
        sys.exit(1)

    step_name = ckpt_path.name  # e.g. global_step_260
    actor_dir = ckpt_path / "actor"
    marker = ckpt_path / ".hf_uploaded"

    if marker.exists() and not args.force:
        print(f"[HF-UPLOAD] {step_name} already uploaded (marker exists). Use --force to re-upload.", flush=True)
        sys.exit(0)

    if not actor_dir.exists():
        print(f"[HF-UPLOAD] ERROR: actor dir not found: {actor_dir}", flush=True)
        sys.exit(1)

    model_files = list(actor_dir.glob("model_world_size_*.pt"))
    if not model_files:
        print(f"[HF-UPLOAD] ERROR: no model_world_size_*.pt files in {actor_dir}", flush=True)
        sys.exit(1)

    print(f"[HF-UPLOAD] Found {len(model_files)} model shard(s) in {actor_dir}", flush=True)

    from huggingface_hub import HfApi
    api = HfApi()

    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[HF-UPLOAD] Warning: could not create repo: {e}", flush=True)

    path_in_repo = f"{args.experiment}/{step_name}/actor"
    print(f"[HF-UPLOAD] Uploading {step_name} → {args.repo_id}/{path_in_repo}", flush=True)

    api.upload_folder(
        folder_path=str(actor_dir),
        repo_id=args.repo_id,
        path_in_repo=path_in_repo,
        repo_type="model",
        ignore_patterns=["optim_*", "extra_state_*"],
    )

    marker.touch()
    print(f"[HF-UPLOAD] Done. Marker written to {marker}", flush=True)


if __name__ == "__main__":
    main()
