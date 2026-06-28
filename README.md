# CONDOR

**CONDOR** (Constrained Online Dual-level Optimisation for RAG) is a two-level Hierarchical Goal-Conditioned Reinforcement Learning (H-GCRL) extension of the [MAO-ARAG](baseline/) agentic-RAG baseline. It adds a trainable *Mechanism Router* at Level 0 that selects the most appropriate retrieval strategy for each query before the Level 1 planning agent constructs the workflow.

---

## Key Idea

Standard agentic-RAG systems apply the same retrieval pipeline to every question. CONDOR learns *when* to use which retrieval mechanism by training a lightweight router alongside the existing PPO-trained planner:

| Level | Component | What it does |
|-------|-----------|--------------|
| **L0** | `MechanismRouter` | Selects one of 5 retrieval mechanisms (ε-greedy + REINFORCE) |
| **L1** | `PlanningAgent` | Plans a workflow restricted to the chosen mechanism's tool set |
| **Dual** | `DualAlphaManager` | Adam-stabilised Lagrange multipliers control cost–quality trade-offs |
| **Baseline** | `PhiScorer` | MLP counterfactual baseline blended with the PPO critic |

**Five mechanisms:**

| ID | Name | Action space |
|----|------|-------------|
| m0 | Pure LLM | `[AG]` |
| m1 | Standard RAG | `[QR, R, DS, AG]` |
| m2 | Advanced Iterative RAG | `[QR, QDP, QDS, R, DS, AG]` |
| m3 | Graph RAG | `[QR, R, DS, AG]` |
| m4 | Web Search RAG | `[QR, R, DS, AG]` |

See [docs/architecture.md](docs/architecture.md) for a full technical description.

---

## Repository Layout

```
adaptive_rag/
├── baseline/               # MAO-ARAG source — NEVER MODIFIED
├── condor/                 # All novel CONDOR components
│   ├── mechanism_router.py
│   ├── dual_alpha_manager.py
│   ├── phi_scorer.py
│   ├── process_reward.py
│   ├── trajectory_logger.py
│   ├── analysis/           # K-Means failure taxonomy
│   ├── metrics/            # CSV / JSON streaming metrics
│   └── plots/              # Matplotlib figure generators
├── configs/                # Hydra YAML configs (all hyperparameters)
├── data/                   # Dataset preprocessing scripts
├── docs/                   # Extended documentation (this file links here)
├── qa_manager/             # Goal-conditioned prompt / agent wrappers
├── retriever/              # Graph and web retrieval backends
├── scripts/                # One-time data preparation helpers
├── slurm/                  # SLURM job scripts for Babel cluster
├── tests/                  # 133-test pytest suite
├── outputs/                # Light outputs: metrics/, plots/, logs/
└── verl/trainer/           # CONDOR trainer & eval entry points
```

Heavy outputs (checkpoints, parquet data) live in `/data/user_data/jgibson2/condor/`.

---

## Quick Start

### 1. Prerequisites

```bash
conda activate condor          # your env with verl, torch, ray
pip install -r requirements_condor.txt
```

### 2. Data preparation (one-time)

All datasets except NQ train are already available at `/data/user_data/jgibson2/aqawflow/data/`. Generate the NQ training parquet:

```bash
bash scripts/prep_nq_train.sh
```

This reads `aqawflow/data/nq/nq_train_questions_and_answers.json` (87 925 examples) and writes `/data/user_data/jgibson2/condor/data/nq/train.parquet`.

### 3. Run tests

```bash
python3 -m pytest tests/ -v
```

All 133 tests should pass before submitting any training job.

### 4. Training

```bash
# NQ (primary experiment)
sbatch slurm/train_condor_nq.slurm

# HotpotQA
sbatch slurm/train_condor_hotpot.slurm

# Development run (no SLURM, small scale)
bash run_condor.sh
```

### 5. Evaluation

```bash
# All 7 datasets in one job
sbatch slurm/eval_condor_all_datasets.slurm

# Single dataset
sbatch slurm/eval_condor_nq.slurm "eval.checkpoint_path=/data/user_data/jgibson2/condor/checkpoints/rl/<id>"
```

### 6. Generate plots and tables

```bash
sbatch slurm/generate_plots.slurm --experiment condor_nq_<jobid>_<timestamp>
# or
python -m condor.plots.generate_all \
    --metrics-dir outputs/metrics \
    --plots-dir   outputs/plots \
    --experiment  condor_nq_<jobid>_<timestamp>
```

Outputs land in `outputs/plots/` (PDF + PNG + CSV + LaTeX for every figure and table).

---

## Reproducing the Baseline

Set `condor.use_condor=false` to reproduce exact MAO-ARAG behaviour:

```bash
bash run_condor.sh "condor.use_condor=false"
```

This bypasses all CONDOR components and delegates directly to `RayPPOTrainer.fit()`.

---

## Documentation

| File | Contents |
|------|----------|
| [docs/architecture.md](docs/architecture.md) | H-GCRL design, reward functions, all algorithms |
| [docs/codebase.md](docs/codebase.md) | Module-by-module code guide; baseline reuse table |
| [docs/configuration.md](docs/configuration.md) | Every YAML field explained |
| [docs/training.md](docs/training.md) | Step-by-step training guide |
| [docs/evaluation.md](docs/evaluation.md) | Evaluation workflow and dataset coverage |
| [docs/metrics_and_plots.md](docs/metrics_and_plots.md) | All metrics collected; all figures produced |

---

## Citation

If you use CONDOR in your work, please also cite the MAO-ARAG baseline:

```
@article{mao-arag-2024,
  title   = {MAO-ARAG: Multi-Agent Orchestration for Agentic Retrieval-Augmented Generation},
  ...
}
```
