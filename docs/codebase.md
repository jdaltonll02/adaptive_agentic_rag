# Codebase Guide

This document walks through every module in the CONDOR tree. For the high-level algorithm, see [architecture.md](architecture.md).

---

## Directory Map

```
adaptive_rag/
├── baseline/                   ← MAO-ARAG (READ-ONLY — never modify)
├── condor/                     ← CONDOR-specific components
│   ├── __init__.py
│   ├── mechanism_router.py
│   ├── dual_alpha_manager.py
│   ├── phi_scorer.py
│   ├── process_reward.py
│   ├── trajectory_logger.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── cluster_report.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── collector.py
│   └── plots/
│       ├── __init__.py
│       ├── pareto.py
│       ├── ablation.py
│       ├── training_curves.py
│       ├── condor_specific.py
│       └── generate_all.py
├── configs/
│   ├── paths.yaml
│   ├── condor_nq.yaml
│   ├── condor_hotpot.yaml
│   ├── eval_condor_*.yaml      ← one per dataset
│   └── ...
├── data/
│   └── nq_open.py
├── docs/
├── qa_manager/
│   ├── __init__.py
│   ├── config.py
│   ├── PlanningAgent.py
│   ├── qa.py                   ← imports baseline + verl
│   └── AgentPool.py
├── retriever/
│   ├── __init__.py
│   ├── graph_retriever.py
│   └── web_retriever.py
├── scripts/
│   └── prep_nq_train.sh
├── slurm/
│   ├── train_condor_nq.slurm
│   ├── train_condor_hotpot.slurm
│   ├── eval_condor_nq.slurm
│   ├── eval_condor_hotpot.slurm
│   ├── eval_condor_all_datasets.slurm
│   └── generate_plots.slurm
├── tests/
│   ├── test_mechanism_router.py
│   ├── test_dual_alpha.py
│   ├── test_phi_scorer.py
│   ├── test_process_reward.py
│   ├── test_trajectory_logger.py
│   ├── test_metrics_collector.py
│   ├── test_planning_agent.py
│   ├── test_cluster_report.py
│   └── test_integration.py
└── verl/trainer/
    ├── ppo/
    │   └── ray_trainer_condor.py
    ├── main_condor.py
    └── main_eval_condor.py
```

---

## Baseline Reuse

CONDOR does **not** copy baseline files. It imports them at runtime via `sys.path.insert`.

| Baseline module | How CONDOR uses it |
|----------------|-------------------|
| `baseline/verl/trainer/ppo/ray_trainer.py` | `CONDORRayPPOTrainer` extends `RayPPOTrainer`; overrides `fit()` and `_update_condor_components()` |
| `baseline/qa_manager/PlanningAgent.py` | `qa_manager/PlanningAgent.py` subclasses it and overrides `build_system_message()` to inject goal |
| `baseline/qa_manager/qa.py` | `qa_manager/qa.py` subclasses `Agentic_RAG_Manager`; overrides `init_context_1turn_list()` to call mechanism router |
| `baseline/qa_manager/AgentPool.py` | `qa_manager/AgentPool.py` subclasses `AgentPool`; overrides `get_retrieval_agent()` to route m3/m4 to new backends |
| `baseline/verl/utils/reward_score/qa.py` | `condor/process_reward.py` imports `compute_f1_score` directly — no subclassing |

The PYTHONPATH ordering `${PROJECT_ROOT}:${BASELINE_ROOT}` ensures Python resolves `qa_manager` to CONDOR's version first.

---

## Module-by-Module Reference

### `condor/mechanism_router.py`

**Class:** `MechanismRouter`

| Method | Description |
|--------|-------------|
| `__init__(config)` | Builds MLP policy (`32→32→5`), Adam optimiser, loads epsilon schedule from config |
| `route(question, step)` | Returns `(mechanism_id, qtype_id, query_features)` |
| `_query_type(question)` | Heuristic classifier → 0=factoid, 1=multi-hop, 2=comparison, 3=complex |
| `update(features, m_ids, advantages)` | REINFORCE gradient step |
| `state_dict()` / `load_state_dict()` | Checkpoint serialisation |

**Module-level:** `query_to_features(question) → np.ndarray[32]` — pure-Python feature extractor; no GPU required.

---

### `condor/dual_alpha_manager.py`

**Class:** `DualAlphaManager`

| Method | Description |
|--------|-------------|
| `__init__(config)` | Sets `lambda_hi`, `lambda_lo` from config; initialises Adam moments |
| `update(avg_mech_cost, avg_token_cost)` | Adam-stabilised ascent step; clamps to `[0, lambda_max]` |
| `get_lambda_hi()` / `get_lambda_lo()` | Read multipliers |
| `state_dict()` / `load_state_dict()` | Key names: `lambda_hi`, `lambda_lo`, `step`, `m_hi`, `v_hi`, `m_lo`, `v_lo` |

---

### `condor/phi_scorer.py`

**Class:** `PhiScorer`

| Method | Description |
|--------|-------------|
| `__init__(config)` | Builds MLP (`41→64→32→1`); sets `t_warmup`, `tau_trans` |
| `predict(features, m_ids, qtype_ids)` | Returns per-sample predicted F1 |
| `blend(phi_values, v_hi_values)` | Returns `(1−α)*phi + α*V_hi` using current `_t` |
| `update(features, m_ids, qtype_ids, targets)` | MSE regression step; increments `_t` |
| `state_dict()` / `load_state_dict()` | Includes `_t` as `'step'` |

**Blend schedule:** `α_t = sigmoid((_t − t_warmup) / τ_trans)`. Access via `scorer._t`.

---

### `condor/process_reward.py`

**Class:** `CONDORRewardManager`

Extends the baseline reward manager. Key difference: `coeff_cost` is no longer hardcoded to `0.0` — it is set dynamically from `DualAlphaManager.get_lambda_lo()` before each batch.

**Function:** `compute_l0_reward(f1, baseline_f1, mechanism_id, lambda_hi) → float`

```python
delta_f1 = f1 - baseline_f1
cost     = MECHANISM_SELECTION_COST[mechanism_id]
return delta_f1 - lambda_hi * cost
```

---

### `condor/trajectory_logger.py`

**Class:** `TrajectoryLogger`

Maintains a rolling buffer of 36-dim trajectory feature vectors.

| Attribute | Value |
|-----------|-------|
| `FEATURE_DIM` | 36 (4 scalars + 32-dim query embedding) |
| `buffer_size` | 2000 (configurable) |

| Method | Description |
|--------|-------------|
| `log(question, l0_correct, f1, cost, n_turns)` | Appends one row to the buffer |
| `get_matrix()` | Returns `np.ndarray` shape `(N, 36)` |
| `clear()` | Empties buffer |

---

### `condor/analysis/cluster_report.py`

**Class:** `ClusterReport`

| Method | Description |
|--------|-------------|
| `analyze(matrix)` | K-Means (k=4) on normalised matrix; returns `{'cluster_0': {...}, 'cluster_1': {...}, ...}` |
| `label_cluster(stats)` | Maps `(high/low F1) × (low/high cost)` → taxonomy string |

Each cluster dict contains: `size`, `avg_l0_correct`, `avg_f1`, `avg_cost`, `avg_turns`, `label`.

---

### `condor/metrics/collector.py`

**Class:** `MetricsCollector`

Streams metrics to disk without holding state in memory.

| Method | Description |
|--------|-------------|
| `record_training_step(step, metrics)` | Appends one row to `<exp>_training.csv` |
| `record_eval_batch(step, dataset, records)` | Appends records to `<exp>_eval.jsonl`; updates `<exp>_eval_summary.json` |
| `get_training_df()` | Reads and returns training CSV as a DataFrame |
| `close()` | Flushes and closes open file handles |

---

### `condor/plots/`

| Module | What it produces |
|--------|-----------------|
| `pareto.py` | Figure 3: F1 vs token cost scatter + Pareto frontier |
| `ablation.py` | Figure 4: bar chart sweeping one hyperparameter |
| `training_curves.py` | 12-panel training progress (F1, cost, λ, α, cluster fractions…) |
| `condor_specific.py` | Mechanism distribution stacked bar; cluster heatmap; mechanism F1 bar; φ-blend curve |
| `generate_all.py` | CLI orchestrator — reads CSV/JSON, generates all figures and tables |

---

### `qa_manager/`

| File | Description |
|------|-------------|
| `config.py` | `MECHANISM_ACTION_SPACES` dict and `MECHANISM_GOAL_DESCRIPTIONS` strings |
| `PlanningAgent.py` | Subclasses baseline `PlanningAgent`; injects goal into system message |
| `qa.py` | Subclasses `Agentic_RAG_Manager`; calls `MechanismRouter.route()` per question |
| `AgentPool.py` | Subclasses baseline `AgentPool`; dispatches m3→`GraphRetriever`, m4→`WebRetriever` |
| `__init__.py` | Exports only pure-Python symbols (no verl/Ray imports at package level) |

---

### `retriever/`

| File | Description |
|------|-------------|
| `graph_retriever.py` | Builds a NetworkX entity co-occurrence graph from retrieved passages; re-ranks by neighbourhood overlap |
| `web_retriever.py` | DuckDuckGo Lite as primary; Google Custom Search API as fallback |

---

### `verl/trainer/ppo/ray_trainer_condor.py`

**Class:** `CONDORRayPPOTrainer(RayPPOTrainer)`

Key overrides:

| Method | Change from baseline |
|--------|---------------------|
| `__init__()` | Instantiates `MechanismRouter`, `DualAlphaManager`, `PhiScorer`, `TrajectoryLogger`, `ClusterReport`, `MetricsCollector` |
| `fit()` | Calls `_route_batch()` before rollout; calls `_update_condor_components()` after PPO step; logs to `MetricsCollector` |
| `_route_batch(batch)` | Iterates questions; calls `router.route()`; injects `mechanism_id` into `extra_info` |
| `_update_condor_components(step, metrics)` | REINFORCE update → DualAlpha update → PhiScorer update → TrajectoryLogger → optional ClusterReport |

---

### `verl/trainer/main_condor.py`

Entry point for training. Resolves Hydra config → instantiates `CONDORRayPPOTrainer` → calls `fit()`. Handles checkpoint restore.

### `verl/trainer/main_eval_condor.py`

Standalone evaluation script. Sets `epsilon_start=epsilon_end=0.0` (pure exploitation), iterates validation set, calls `metrics_collector.record_eval_batch()`, then generates all figures. No PPO update is performed.

---

## Test Suite

```
tests/
├── test_mechanism_router.py    (20 tests) — init, route, update, checkpointing
├── test_dual_alpha.py          (15 tests) — update, clamping, Adam moments
├── test_phi_scorer.py          (16 tests) — predict, blend schedule, update
├── test_process_reward.py      (16 tests) — L0 reward, CONDORRewardManager
├── test_trajectory_logger.py   (12 tests) — log, matrix shape, buffer rolling
├── test_metrics_collector.py   (13 tests) — CSV writes, JSONL writes, summary
├── test_planning_agent.py      (17 tests) — goal injection, tool restriction
├── test_cluster_report.py      (12 tests) — K-Means, labelling, dict structure
└── test_integration.py         ( 8 tests) — smoke tests across module boundaries
```

Run with: `python3 -m pytest tests/ -v`
