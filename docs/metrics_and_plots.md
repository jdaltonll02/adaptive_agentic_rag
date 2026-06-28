# Metrics and Plots

This document describes every metric collected during training and evaluation, and every figure and table produced by `condor.plots.generate_all`.

---

## Metrics Collected During Training

`MetricsCollector` streams two files per experiment:

### `<exp>_training.csv`

One row per training step. Columns:

| Column | Description |
|--------|-------------|
| `step` | Global gradient update step |
| `f1` | Average F1 over the training batch |
| `em` | Average exact match |
| `token_cost` | Average token cost (scaled 0–1) |
| `n_turns` | Average number of planning turns |
| `l0_reward` | Average L0 reward (`ΔF1 − λ_hi × C_hi`) |
| `lambda_hi` | Current `λ_hi` (L0 Lagrange multiplier) |
| `lambda_lo` | Current `λ_lo` (L1 Lagrange multiplier) |
| `alpha_blend` | Current `α_t` (PhiScorer blend weight) |
| `phi_mse` | PhiScorer regression loss |
| `router_loss` | REINFORCE loss for the mechanism router |
| `ppo_policy_loss` | L1 PPO policy loss |
| `ppo_value_loss` | L1 PPO value function loss |
| `m0_frac` – `m4_frac` | Fraction of batch assigned to each mechanism |
| `cluster_efficient_correct` | Fraction of trajectories in the `efficient_correct` cluster |
| `cluster_expensive_correct` | Fraction in `expensive_correct` |
| `cluster_efficient_incorrect` | Fraction in `efficient_incorrect` |
| `cluster_expensive_incorrect` | Fraction in `expensive_incorrect` |

### `<exp>_eval.jsonl`

One JSON object per evaluated question:

```json
{
  "step": 1000,
  "dataset": "nq",
  "question": "Who invented the telephone?",
  "gold_answer": "Alexander Graham Bell",
  "pred_answer": "Alexander Graham Bell",
  "f1": 1.0,
  "em": 1,
  "mechanism_id": 1,
  "n_turns": 2,
  "token_cost": 0.12
}
```

### `<exp>_eval_summary.json`

Aggregated statistics per dataset per step:

```json
{
  "nq": {
    "step": 1000,
    "avg_f1": 0.612,
    "avg_em": 0.541,
    "avg_cost": 0.183,
    "avg_turns": 2.4,
    "l0_correct_rate": 0.71,
    "mechanism_distribution": {"0": 0.05, "1": 0.42, "2": 0.31, "3": 0.12, "4": 0.10}
  }
}
```

---

## Generating All Plots and Tables

```bash
python -m condor.plots.generate_all \
    --metrics-dir outputs/metrics \
    --plots-dir   outputs/plots \
    --experiment  condor_nq_<jobid>_<timestamp>
```

Or via SLURM:

```bash
sbatch slurm/generate_plots.slurm --experiment condor_nq_<jobid>_<timestamp>
```

---

## Figures

### Figure 3 — Pareto Frontier (`pareto.py`)

**File:** `pareto_<experiment>.pdf` / `.png`

Scatter plot of F1 vs token cost, one point per evaluated question. The Pareto frontier (lowest cost for each F1 level) is overlaid as a step-post curve. Reference anchor points for three baseline systems are included:

| System | Approximate F1 | Approximate Cost |
|--------|---------------|-----------------|
| LLM w/o RAG (m0 only) | 0.31 | 0.05 |
| Vanilla RAG (m1 only) | 0.52 | 0.25 |
| MAO-ARAG (baseline) | 0.61 | 0.45 |

Points are coloured by selected mechanism (m0–m4).

---

### Figure 4 — Ablation (`ablation.py`)

**File:** `ablation_<experiment>.pdf` / `.png`

Bar chart showing F1 and cost as a function of one swept hyperparameter (e.g. `budget_hi`, `epsilon_decay`, `phi_t_warmup`). If `sweep_results` is not directly available, `sweep_from_training_csv()` bins the training CSV by the requested parameter column.

---

### Training Curves (`training_curves.py`)

**File:** `training_curves_<experiment>.pdf` / `.png`

12-panel grid covering:

| Panel | Metric |
|-------|--------|
| 1 | F1 (smoothed) |
| 2 | Exact match |
| 3 | Token cost |
| 4 | Average turns |
| 5 | L0 reward |
| 6 | λ_hi and λ_lo |
| 7 | α blend weight |
| 8 | PhiScorer MSE |
| 9 | Router REINFORCE loss |
| 10 | PPO policy loss |
| 11 | PPO value loss |
| 12 | Cluster fractions (4 lines) |

Missing columns are silently skipped (no crash).

---

### Mechanism Distribution (`condor_specific.py`)

**File:** `mechanism_dist_<experiment>.pdf` / `.png`

Stacked bar chart of m0–m4 fractions over training steps. Shows how the router's policy evolves from exploration (uniform) to specialised routing.

---

### Cluster Heatmap (`condor_specific.py`)

**File:** `cluster_heatmap_<experiment>.pdf` / `.png`

Heatmap of cluster centroids (4 clusters × 4 axes: F1, cost, turns, l0_correct). Colour-coded by cluster label (`efficient_correct`, `expensive_correct`, `efficient_incorrect`, `expensive_incorrect`). Used to diagnose failure modes.

---

### Mechanism F1 Bar Chart (`condor_specific.py`)

**File:** `mechanism_f1_<experiment>.pdf` / `.png`

Grouped bar chart: one group per dataset, one bar per mechanism (m0–m4). Shows which mechanism performs best on each dataset type.

---

### Phi-Blend Curve (`condor_specific.py`)

**File:** `phi_blend_<experiment>.pdf` / `.png`

Dual-axis plot showing `α_t` (blend weight, left axis) and PhiScorer MSE loss (right axis) over training steps. Illustrates the warm-up-then-blend schedule.

---

## Tables

All tables are saved in three formats: `.csv`, `.md` (pipe-delimited), and `.tex` (LaTeX `tabular`). Bold formatting marks the best value per column.

### Table 1 — F1 Comparison

**File:** `table1_f1_<experiment>.(csv|md|tex)`

F1 scores for CONDOR and three baseline systems across all 7 datasets.

| System | NQ | HotpotQA | AmbigQA | MuSiQue | 2Wiki | Bamboogle | PopQA |
|--------|----|---------|---------|---------|-------|-----------|-------|
| LLM w/o RAG | 0.310 | 0.211 | 0.298 | 0.182 | 0.198 | 0.254 | 0.412 |
| Standard RAG | 0.524 | 0.388 | 0.441 | 0.301 | 0.334 | 0.387 | 0.562 |
| MAO-ARAG | 0.612 | 0.498 | 0.531 | 0.423 | 0.461 | 0.503 | 0.634 |
| **CONDOR** | *from eval* | … | … | … | … | … | … |

### Table 5 — Token Cost

**File:** `table5_token_cost_<experiment>.(csv|md|tex)`

Average normalised token cost per dataset and system.

### Table 6 — Retrieval Cost

**File:** `table6_retrieval_<experiment>.(csv|md|tex)`

Average retrieval API calls per question.

### Table 7 — Average Turns

**File:** `table7_turns_<experiment>.(csv|md|tex)`

Average planning turns per dataset and system.

### Mechanism Breakdown

**File:** `table_mechanism_breakdown_<experiment>.(csv|md|tex)`

Per-dataset mechanism selection fractions (m0–m4 percentages).

### Pareto Summary

**File:** `table_pareto_summary_<experiment>.(csv|md|tex)`

F1 and cost at the Pareto-optimal operating point for each system.

---

## Adding Custom Metrics

To add a new metric to the training CSV:

1. Compute the value inside `CONDORRayPPOTrainer._update_condor_components()`.
2. Add it to the `merged_dict` passed to `self.metrics_collector.record_training_step()`.
3. Add a new `_PANEL_SPECS` entry in `condor/plots/training_curves.py` with `col=<column_name>`.

No other changes are needed — `MetricsCollector` writes whatever keys are present in the dict.
