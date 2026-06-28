# Training Guide

This document covers everything needed to run a CONDOR training job on the Babel cluster.

---

## Prerequisites

### Environment

```bash
conda activate condor
pip install -r requirements_condor.txt   # if not already installed
```

Verify the environment:

```bash
python -c "import verl, ray, torch; print('OK')"
```

### PYTHONPATH

All SLURM scripts set this automatically. For interactive runs, set it manually:

```bash
export PYTHONPATH="/home/jgibson2/projects/adaptive_rag:/home/jgibson2/projects/adaptive_rag/baseline:${PYTHONPATH:-}"
```

---

## Data Preparation (one-time)

Most datasets are already available at `/data/user_data/jgibson2/aqawflow/data/`. Only the NQ training split needs to be generated:

```bash
bash scripts/prep_nq_train.sh
```

This reads `aqawflow/data/nq/nq_train_questions_and_answers.json` (87,925 questions) and writes `/data/user_data/jgibson2/condor/data/nq/train.parquet`.

Verify:

```bash
python -c "import pandas as pd; df = pd.read_parquet('/data/user_data/jgibson2/condor/data/nq/train.parquet'); print(len(df), df.columns.tolist())"
```

Expected: `87925 ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']`

---

## Running Tests

Always run the test suite before submitting a training job:

```bash
cd /home/jgibson2/projects/adaptive_rag
python3 -m pytest tests/ -v
```

All 133 tests must pass. If any fail, do not submit.

---

## Training on Babel (SLURM)

### NQ (primary experiment)

```bash
sbatch slurm/train_condor_nq.slurm
```

This requests 1 L40S GPU, 64 GB RAM, and up to 24 hours. Checkpoint and log paths are set automatically.

### HotpotQA

```bash
sbatch slurm/train_condor_hotpot.slurm
```

### Monitoring

```bash
# Watch the job log
tail -f /data/user_data/jgibson2/condor/logs/slurm/<jobid>_train_nq.out

# W&B: open the verl_condor project in your browser
# Experiment name format: condor_nq_<jobid>_<timestamp>
```

---

## Interactive / Development Run

For quick iteration without SLURM:

```bash
cd /home/jgibson2/projects/adaptive_rag
bash run_condor.sh
```

`run_condor.sh` uses a small batch size and reduced step count. Append Hydra overrides directly:

```bash
bash run_condor.sh "condor.router_lr=5e-4" "trainer.total_training_steps=200"
```

---

## Checkpoint Layout

Checkpoints are saved to `/data/user_data/jgibson2/condor/checkpoints/rl/`:

```
checkpoints/rl/
├── global_step_100/
│   ├── actor/          ← HuggingFace model weights
│   ├── critic/
│   ├── condor_router.pt
│   ├── condor_dual_alpha.pt
│   └── condor_phi_scorer.pt
├── global_step_200/
└── latest -> global_step_200/   ← symlink to most recent
```

The CONDOR component checkpoints (`condor_*.pt`) are saved alongside the standard PPO checkpoints.

### Resuming from a Checkpoint

```bash
sbatch slurm/train_condor_nq.slurm \
    "trainer.resume_from_checkpoint=/data/user_data/jgibson2/condor/checkpoints/rl/global_step_500"
```

---

## Key Hyperparameters

See [configuration.md](configuration.md) for the full reference. The most commonly tuned fields:

| Parameter | Config key | Effect |
|-----------|-----------|--------|
| Router exploration | `condor.epsilon_start` / `condor.epsilon_end` | Higher → more random mechanism selection early |
| L0 cost constraint | `condor.budget_hi` | Lower → router penalised more for expensive mechanisms |
| L1 cost constraint | `condor.budget_lo` | Lower → planner penalised more for long workflows |
| Lagrange step size | `condor.dual_alpha_lr` | Larger → faster λ adaptation but less stable |
| Phi warm-up | `condor.phi_t_warmup` | Delay before blending to PPO critic |
| PPO learning rate | `ppo.lr` | Actor LR; the most impactful single LR |

---

## Baseline Reproduction

To reproduce MAO-ARAG exactly (no CONDOR components active):

```bash
bash run_condor.sh "condor.use_condor=false"
# or
sbatch slurm/train_condor_nq.slurm "condor.use_condor=false"
```

The `use_condor=false` flag causes `CONDORRayPPOTrainer.fit()` to immediately delegate to `super().fit()`. No CONDOR objects are created; training is identical to running the baseline directly.

---

## Output Files

After training completes:

| Location | Contents |
|----------|---------|
| `/data/user_data/jgibson2/condor/checkpoints/rl/` | Model + CONDOR component checkpoints |
| `/data/user_data/jgibson2/condor/logs/slurm/` | SLURM stdout/stderr |
| `outputs/metrics/<exp>_training.csv` | Per-step metrics (F1, cost, λ, α, cluster fractions) |
| `outputs/metrics/<exp>_eval.jsonl` | Per-question eval records |
| `outputs/metrics/<exp>_eval_summary.json` | Aggregated eval metrics per dataset |
| W&B dashboard | Real-time training curves |

---

## Generating Plots After Training

```bash
# Via SLURM (CPU only, 30 min)
sbatch slurm/generate_plots.slurm --experiment condor_nq_<jobid>_<timestamp>

# Interactively
python -m condor.plots.generate_all \
    --metrics-dir outputs/metrics \
    --plots-dir   outputs/plots \
    --experiment  condor_nq_<jobid>_<timestamp>
```

See [metrics_and_plots.md](metrics_and_plots.md) for all figures and tables produced.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'verl'`**
→ PYTHONPATH is not set. Run `export PYTHONPATH=...` or use a SLURM script.

**`ModuleNotFoundError: No module named 'condor'`**
→ PROJECT_ROOT must precede BASELINE_ROOT in PYTHONPATH.

**OOM on GPU**
→ Reduce `data.train_batch_size` or `ppo.mini_batch_size`.

**Ray initialisation errors**
→ Ensure `RAY_DEDUP_LOGS=0` is set (done in SLURM scripts). If Ray is already running from a previous crashed job, run `ray stop` first.

**W&B authentication**
→ Run `wandb login` interactively once. The token is cached in `~/.netrc`.
