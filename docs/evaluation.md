# Evaluation Guide

---

## Overview

CONDOR evaluation runs a trained checkpoint in pure-exploitation mode across any supported dataset.
No gradient updates occur. Each question is routed through the mechanism router and answered by the appropriate retrieval mechanism. Results are logged to stdout and optionally to W&B.

**Important:** Always use `baseline/verl/trainer/main_eval_condor.py` as the eval entry point — it loads the trained RL policy and runs the full CONDOR rollout. The project-root version (`verl/trainer/main_eval_condor.py`) uses a different eval loop without policy loading and is not for checkpoint evaluation.

---

## Supported Datasets

All test sets are capped at 996 questions (or fewer where the full split is smaller). Files live at `/data/user_data/jgibson2/condor/data/`.

| Dataset | Split used | Questions | Parquet path |
|---------|-----------|----------:|--------------|
| NQ | test (last 996) | 996 | `nq/test_996.parquet` |
| HotpotQA | validation[:996] | 996 | `hotpot_qa/test_996.parquet` |
| MuSiQue | validation[:996] | 996 | `musique/test_996.parquet` |
| CWQ | test[:996] | 996 | `cwq/test_996.parquet` |
| TriviaQA | validation[:996] | 996 | `trivia_qa/test_996.parquet` |
| FRAMES | test (full) | 824 | `frames/test_996.parquet` |
| KILT-NQ | validation[:996] | 996 | `kilt_nq/test_996.parquet` |
| StrategyQA | test (full) | 687 | `strategy_qa/test_996.parquet` |
| AmbigQA | — | — | *not yet set up* |
| 2WikiMultiHop | — | — | *not yet set up* |
| BamBoogle | — | — | *not yet set up* |
| PopQA | — | — | *not yet set up* |

HotpotQA and MuSiQue use the validation split because their public test splits withhold gold answers. The 996 cap is consistent with the CONDOR baseline methodology.

---

## Prerequisites

### Environment

```bash
source ~/.bashrc
source .venv/bin/activate
export PYTHONPATH="/home/jgibson2/projects/adaptive_rag:/home/jgibson2/projects/adaptive_rag/baseline:${PYTHONPATH:-}"
```

### API credentials (OpenRouter)

The eval agent uses `google/gemini-2.5-flash-lite` via OpenRouter. Ensure the following are set in `~/.bashrc`:

```bash
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="sk-or-v1-..."
export CONDOR_AGENT_MODEL="google/gemini-2.5-flash-lite"
```

### Retrieval server

The FAISS index takes ~25 minutes to load. All eval SLURM scripts start it automatically and wait up to 30 minutes before beginning eval.

---

## Running Evaluation

### One dataset at a time (recommended)

```bash
cd /home/jgibson2/projects/adaptive_rag

sbatch --job-name=condor_eval_<ds> slurm/eval_condor_dataset.slurm <ds>
```

Where `<ds>` is the config suffix matching `configs/eval_condor_<ds>.yaml`:

```bash
sbatch --job-name=condor_eval_nq         slurm/eval_condor_dataset.slurm nq
sbatch --job-name=condor_eval_hotpot     slurm/eval_condor_dataset.slurm hotpot
sbatch --job-name=condor_eval_musique    slurm/eval_condor_dataset.slurm musique
sbatch --job-name=condor_eval_cwq        slurm/eval_condor_dataset.slurm cwq
sbatch --job-name=condor_eval_triviaqa   slurm/eval_condor_dataset.slurm triviaqa
sbatch --job-name=condor_eval_frames     slurm/eval_condor_dataset.slurm frames
sbatch --job-name=condor_eval_kilt_nq    slurm/eval_condor_dataset.slurm kilt_nq
sbatch --job-name=condor_eval_strategy   slurm/eval_condor_dataset.slurm strategy_qa
```

Each job requests 1× L40S GPU, 192 GB RAM, up to 6 hours. The checkpoint path is hardcoded in `eval_condor_dataset.slurm`; override with an extra Hydra arg if needed:

```bash
sbatch --job-name=condor_eval_nq slurm/eval_condor_dataset.slurm nq \
    "eval.checkpoint_path=/data/user_data/jgibson2/condor/checkpoints/rl/.../global_step_<N>/actor_hf"
```

### All datasets sequentially (one job)

```bash
sbatch slurm/eval_condor_all_datasets.slurm
```

Runs all 12 datasets in sequence using one retrieval server instance. Wall time: up to 20 hours.

### Smoke test (4 questions, fast validation)

```bash
sbatch slurm/eval_condor_nq_smoke.slurm
```

Runs 4 NQ questions end-to-end. Use this to validate the pipeline after any infrastructure change before submitting a full eval.

---

## Eval Configuration

Configs live at `configs/eval_condor_<dataset>.yaml`. Key fields:

```yaml
eval:
  checkpoint_path: "${paths.rl_checkpoint_dir}/latest"   # override per run
  max_turn: 5

trainer:
  val_only: true
  val_before_train: true
```

The `val_only` + `val_before_train` combination causes `RayPPOTrainer` to run one validation pass and exit without training.

---

## Environment Variables (SLURM scripts)

All eval SLURM scripts set:

```bash
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://openrouter.ai/api/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export CONDOR_AGENT_MODEL="${CONDOR_AGENT_MODEL:-google/gemini-2.5-flash-lite}"
export CUDA_VISIBLE_DEVICES=0
export DISABLE_FLASH_ATTN_TRITON_CE=1          # bypasses broken Triton CE kernel on Python 3.12
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}     # avoids stale /tmp/ray/ on requeue
```

---

## Metrics Reported

| Metric | Definition |
|--------|-----------|
| `acc` | Accuracy (token-level F1 ≥ 0.5 threshold) |
| `em` | Exact match (normalised: lowercase, punctuation stripped) |
| `f1` | Token-level F1 between predicted and gold answer |
| `precision` / `recall` | Token-level precision and recall |
| `rouge1` / `rouge2` / `rougeL` | ROUGE scores |
| `turn_num` | Average number of reasoning turns per question |
| `multi_step_pct` | Fraction of questions using more than one turn |
| `mechanism/m{1-4}_f1` | F1 broken down by which mechanism the router selected |
| `mechanism/m{1-4}_count` | Number of questions routed to each mechanism |

Results are printed to the job's stdout log at the end of the run.

---

## Troubleshooting

**`PermissionError: /tmp/ray/session_...`**
→ Stale Ray temp dir from a previous job on the same node. Fixed by `RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}` (already in all SLURM scripts).

**`Error code: 400 - Please pass a valid API key`**
→ `OPENAI_API_KEY` is empty or wrong. Check `echo $OPENAI_API_KEY` after `source ~/.bashrc`. Ensure the OpenRouter key starts with `sk-or-v1-`.

**`IndexError: list index out of range` in rollout**
→ LLM API calls are failing silently, leaving `sub_query` empty. Root cause is always a bad API key or exhausted quota.

**`FileNotFoundError` on checkpoint path**
→ Specify the full path to `actor_hf/` explicitly: `.../global_step_400/actor_hf`.

**Retrieval server never becomes ready**
→ Check `${SLURM_JOB_ID}_retrieval_server.log` in `log/slurm_logs/`. The FAISS index load can take up to 25 minutes; the scripts wait 30 minutes (360 × 5 s).
