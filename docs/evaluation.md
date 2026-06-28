# Evaluation Guide

---

## Overview

CONDOR evaluation runs a trained checkpoint in pure-exploitation mode (`ε = 0`) across any of the 7 supported datasets. No gradient updates are made. Each question is routed through the mechanism router, planned by the goal-conditioned agent, and answered by the appropriate retrieval backend. Results are written to structured metric files and optional plots are generated afterwards.

---

## Supported Datasets

| Dataset | Split | Path |
|---------|-------|------|
| NQ (Natural Questions) | test (996) | `aqawflow/data/nq/test_996.parquet` |
| HotpotQA | test (996) | `aqawflow/data/hotpot_qa/test_996.parquet` |
| AmbigQA | test (996) | `aqawflow/data/ambig_qa/test_996.parquet` |
| MuSiQue | test (996) | `aqawflow/data/musique/test_996.parquet` |
| 2WikiMultiHop | test (996) | `aqawflow/data/2wikimultihop_qa/test_996.parquet` |
| Bamboogle | test | `aqawflow/data/bamboogle/test.parquet` |
| PopQA | test (996) | `aqawflow/data/pop_qa/test_996.parquet` |

All parquet files share the same schema: `['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']`.

---

## Running Evaluation

### All 7 datasets in one job (recommended)

```bash
sbatch slurm/eval_condor_all_datasets.slurm \
    "eval.checkpoint_path=/data/user_data/jgibson2/condor/checkpoints/rl/global_step_<N>"
```

This evaluates each dataset sequentially then generates per-dataset plots. Wall time: up to 12 hours. Uses 1 L40S GPU.

### Single dataset

```bash
# NQ
sbatch slurm/eval_condor_nq.slurm \
    "eval.checkpoint_path=/data/user_data/jgibson2/condor/checkpoints/rl/global_step_<N>"

# HotpotQA
sbatch slurm/eval_condor_hotpot.slurm \
    "eval.checkpoint_path=..."

# 2Wiki, AmbigQA, MuSiQue, Bamboogle, PopQA — same pattern
sbatch slurm/eval_condor_2wiki.slurm    "eval.checkpoint_path=..."
sbatch slurm/eval_condor_ambigqa.slurm  "eval.checkpoint_path=..."
sbatch slurm/eval_condor_musique.slurm  "eval.checkpoint_path=..."
sbatch slurm/eval_condor_bamboogle.slurm "eval.checkpoint_path=..."
sbatch slurm/eval_condor_popqa.slurm    "eval.checkpoint_path=..."
```

### Interactive (no SLURM)

```bash
export PYTHONPATH="/home/jgibson2/projects/adaptive_rag:/home/jgibson2/projects/adaptive_rag/baseline:${PYTHONPATH:-}"

python -m verl.trainer.main_eval_condor \
    --config-path configs \
    --config-name eval_condor_nq \
    "eval.checkpoint_path=/data/user_data/jgibson2/condor/checkpoints/rl/global_step_<N>" \
    "trainer.experiment_name=my_eval_run"
```

---

## Checkpoint Path

The default checkpoint path in all eval configs is `"${paths.rl_checkpoint_dir}/latest"`, which resolves to `/data/user_data/jgibson2/condor/checkpoints/rl/latest`. This symlink is updated automatically at the end of each training job.

To evaluate a specific step, override at the command line as shown above.

---

## Eval Configuration

Eval configs live at `configs/eval_condor_<dataset>.yaml`. Key settings that differ from training configs:

```yaml
condor:
  use_condor: true
  epsilon_start: 0.0    # pure exploitation — no random mechanism selection
  epsilon_end:   0.0
  epsilon_decay: 1.0
  l0_reward_threshold: 0.4

eval:
  checkpoint_path: "${paths.rl_checkpoint_dir}/latest"
  router_checkpoint: null    # null = load from main checkpoint bundle
  max_turn: 5
  eval_batch_size: 8
```

See [configuration.md](configuration.md) for the full field reference.

---

## Output Files

After evaluation, the following files are written under `outputs/metrics/`:

| File | Contents |
|------|---------|
| `<exp>_eval.jsonl` | One JSON object per question: `{question, answer, pred, f1, em, mechanism_id, n_turns, token_cost}` |
| `<exp>_eval_summary.json` | Aggregated metrics per dataset: `{avg_f1, avg_em, avg_cost, avg_turns, mechanism_distribution}` |

If plots are generated (either by the eval script or `slurm/generate_plots.slurm`), figures land in `outputs/plots/`.

---

## Metrics Reported

| Metric | Definition |
|--------|-----------|
| `avg_f1` | Token-level F1 between predicted and gold answers |
| `avg_em` | Exact match (normalised: lowercase, punctuation stripped) |
| `avg_cost` | Average token cost per question (scaled 0–1) |
| `avg_turns` | Average number of planning turns per question |
| `mechanism_distribution` | Fraction of questions assigned to each mechanism (m0–m4) |
| `l0_correct_rate` | Fraction of questions where the router's mechanism choice resulted in above-threshold reward |

---

## Comparing Against Baseline

Run baseline evaluation with:

```bash
sbatch slurm/eval_condor_nq.slurm \
    "condor.use_condor=false" \
    "eval.checkpoint_path=/data/user_data/jgibson2/condor/checkpoints/rl/global_step_<N>"
```

Or compare experiment names in the `TablesGenerator`:

```python
from condor.metrics.tables import TablesGenerator
gen = TablesGenerator(metrics_dir='outputs/metrics', output_dir='outputs/plots')
gen.generate_all(experiment='condor_nq_<id>', baseline_experiment='baseline_nq_<id>')
```

---

## Troubleshooting

**`FileNotFoundError` on checkpoint path**
→ The `latest` symlink may not exist. Run training first or specify an explicit step path.

**`ValueError: checkpoint has no router state`**
→ The checkpoint was saved with `condor.use_condor=false`. Re-train with CONDOR enabled, or specify `eval.router_checkpoint` to load a router from a different checkpoint.

**All questions assigned to m1**
→ Expected behaviour for early-stopped or untrained router. The router needs sufficient training steps to learn a diverse policy.

**Low F1 on Bamboogle / PopQA**
→ These datasets have different knowledge requirements. m4 (web search) should be preferred; check mechanism distribution.
