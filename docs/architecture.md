# CONDOR Architecture

CONDOR extends MAO-ARAG with a two-level Hierarchical Goal-Conditioned RL (H-GCRL) framework. The two levels are trained jointly: Level 0 selects a retrieval mechanism; Level 1 plans a workflow conditioned on that goal.

---

## Overview

```
Query q
  │
  ▼
┌─────────────────────────────────┐
│  Level 0 — MechanismRouter      │
│  ε-greedy + REINFORCE           │
│  Input : 32-dim query features  │
│  Output: m* ∈ {0,1,2,3,4}       │
└──────────────┬──────────────────┘
               │  goal = (m*, qtype_id)
               ▼
┌─────────────────────────────────┐
│  Level 1 — PlanningAgent        │
│  PPO (baseline RayPPOTrainer)   │
│  Prompt restricted to           │
│  MECHANISM_ACTION_SPACES[m*]    │
└──────────────┬──────────────────┘
               │  workflow string
               ▼
┌─────────────────────────────────┐
│  Agent execution                │
│  (RetrievalAgent dispatches     │
│   to FAISS / Graph / Web)       │
└──────────────┬──────────────────┘
               │  answer + costs
               ▼
┌─────────────────────────────────┐
│  Reward & update                │
│  L0: REINFORCE + PhiScorer      │
│  L1: PPO (unchanged baseline)   │
│  λ : DualAlphaManager           │
└─────────────────────────────────┘
```

---

## Level 0 — Mechanism Router

**File:** `condor/mechanism_router.py`

### 5 Retrieval Mechanisms

| ID | Name | Description | Action Space |
|----|------|-------------|-------------|
| m0 | Pure LLM | No retrieval; answer from parametric knowledge | `[AG]` |
| m1 | Standard RAG | Single-hop FAISS retrieval | `[QR, R, DS, AG]` |
| m2 | Advanced Iterative RAG | Decomposition + iterative retrieval | `[QR, QDP, QDS, R, DS, AG]` |
| m3 | Graph RAG | Entity co-occurrence graph re-ranking | `[QR, R, DS, AG]` |
| m4 | Web Search RAG | Live DuckDuckGo / Google CSE retrieval | `[QR, R, DS, AG]` |

### Query Feature Extraction

Each question is encoded into a 32-dimensional float vector via `query_to_features(question)`. The encoding captures:
- Bag-of-character n-gram features (TF-style)
- Question length, word count, punctuation density
- Presence of WH-words, comparative/superlative words, named-entity-like patterns

### Policy Architecture

A three-layer MLP (`32 → 32 → 5`) maps query features to mechanism logits. During exploitation (ε = 0), the argmax is selected. During exploration, a rule-based prior `P(m | qtype)` is sampled.

**Rule-based prior** (used during exploration):

| Query type | Preferred mechanism |
|------------|-------------------|
| Factoid (who, what year…) | m1 (Standard RAG) |
| Multi-hop (before/after, first/then…) | m2 (Adv. Iterative) |
| Comparison (compare, versus…) | m3 (Graph RAG) |
| Complex / analytical | m4 (Web Search) |

### Exploration Schedule

ε decays exponentially:

```
ε(t) = ε_end + (ε_start − ε_end) × exp(−t / ε_decay)
```

Default: `ε_start=0.9`, `ε_end=0.1`, `ε_decay=1000` steps.

### REINFORCE Update

After each batch the router is updated via policy gradient:

```
∇J(θ) = Σ_i (R_i − b_i) × ∇log π_θ(m*_i | q_i)
```

where `b_i` is the blended PhiScorer baseline (see below) and `R_i` is the L0 reward.

---

## Level 0 Reward Function

**File:** `condor/process_reward.py`

```
R_hi(i) = ΔF1(i) − λ_hi × C_hi(m*_i)
ΔF1(i)  = F1(i) − baseline_F1(i)
C_hi(m) = MECHANISM_SELECTION_COST[m]   # ∈ [0, 1]
```

**Mechanism selection costs:**

| m | Cost | Rationale |
|---|------|-----------|
| m0 | 0.0 | Free (no retrieval) |
| m1 | 0.1 | Cheap (one API call) |
| m2 | 0.3 | Moderate (multi-turn) |
| m3 | 0.4 | Higher (graph construction) |
| m4 | 0.5 | Highest (live web query) |

---

## Level 1 — Goal-Conditioned Planning Agent

**File:** `qa_manager/PlanningAgent.py`

The planning prompt is modified in two ways when `goal=(m*, qtype_id)` is provided:

1. **Tool restriction** — the system message lists only agents in `MECHANISM_ACTION_SPACES[m*]`. The LLM cannot plan a workflow that uses unavailable agents.
2. **Goal prefix** — the system message begins with `[Goal] Selected mechanism: <description>` so the planner understands its operational context.

When `goal=None` (baseline mode or `use_condor=False`), the prompt is bit-for-bit identical to the baseline `PlanningAgent.py`, including the intentional `"Qustion"` typo in the final user message that the SFT checkpoint was trained on.

---

## Dual Gradient Descent — DualAlphaManager

**File:** `condor/dual_alpha_manager.py`

Two Lagrange multipliers control constraint satisfaction:

| Multiplier | Constraint | Controls |
|------------|-----------|---------|
| `λ_hi` | L0 budget: avg mechanism cost ≤ `budget_hi` | How aggressively the router avoids expensive mechanisms |
| `λ_lo` | L1 cost: avg token cost ≤ `budget_lo` | How much the PPO reward penalises token/retrieval spend |

**Update rule (Adam-stabilised gradient ascent):**

```
g_t    = avg_cost_t − budget           # positive when constraint violated
m_t    = β₁m_{t-1} + (1−β₁)g_t
v_t    = β₂v_{t-1} + (1−β₂)g_t²
m̂_t   = m_t / (1−β₁ᵗ)
v̂_t   = v_t / (1−β₂ᵗ)
λ_{t+1} = clip(λ_t + α × m̂_t / (√v̂_t + ε), λ_min, λ_max)
```

Default: `α=1e-3`, `β₁=0.9`, `β₂=0.999`, `ε=1e-8`, `λ_max=10`.

`λ_lo` is injected into `CONDORRewardManager` before each batch, replacing the baseline's hardcoded `coeff_cost=0.0`.

---

## PhiScorer — Counterfactual Baseline

**File:** `condor/phi_scorer.py`

A 41→64→32→1 MLP that predicts expected F1 given a query and the chosen mechanism. Used as a control-variate baseline in the REINFORCE update.

**Input encoding (41-dim):**
```
x = concat(query_emb[:32], m_onehot[5], qtype_onehot[4])
```

**Blend schedule:**

The baseline transitions from the MLP (which converges fast) to the PPO critic (which becomes more accurate as L1 training progresses):

```
α_t = sigmoid((t − t_warmup) / τ_trans)
b_t = (1 − α_t) × φ(s) + α_t × V_hi(s)
```

- Early training (`t << t_warmup`): `α ≈ 0`, pure MLP baseline.
- Late training (`t >> t_warmup`): `α ≈ 1`, pure critic baseline.

Default: `t_warmup=500`, `τ_trans=100`.

---

## Trajectory Logger & Cluster Report

**Files:** `condor/trajectory_logger.py`, `condor/analysis/cluster_report.py`

Every completed question is logged as a 36-dimensional feature vector:

```
[l0_correct(1), f1(1), cost(1), n_turns(1), query_emb[:32](32)]
```

Every `cluster_report_freq` steps (default 50), K-Means (k=4) is run on the normalised buffer. Clusters are auto-labelled by two axes:

| | Low Cost | High Cost |
|--|----------|-----------|
| **High F1** | `efficient_correct` | `expensive_correct` |
| **Low F1** | `efficient_incorrect` | `expensive_incorrect` |

The cluster taxonomy is logged to W&B and used to diagnose failure modes during training.

---

## Retrieval Backends

**Files:** `retriever/graph_retriever.py`, `retriever/web_retriever.py`

| Mechanism | Backend | Implementation |
|-----------|---------|----------------|
| m1, m2 | FAISS HTTP API | Handled by baseline `RetrievalAgent`; unchanged |
| m3 | `GraphRetriever` | NetworkX entity co-occurrence graph; re-ranks by neighbourhood overlap |
| m4 | `WebRetriever` | DuckDuckGo Lite (primary) + Google Custom Search API (fallback) |

---

## Training Loop (CONDORRayPPOTrainer.fit)

```
for each training batch:
  1. L0: route each question → m*, qtype_id, features
  2. Inject m* into extra_info and context dicts
  3. L1: multi-turn rollout (goal-conditioned)
         − PlanningAgent restricted to MECHANISM_ACTION_SPACES[m*]
         − RetrievalAgent dispatches to correct backend
  4. Compute F1 / EM per question
  5. L1 update: assign rewards → compute advantages → PPO update
  6. CONDOR update:
       a. Compute L0 rewards (ΔF1 − λ_hi × C_hi)
       b. REINFORCE update for MechanismRouter
       c. DualAlphaManager.update(avg_mech_cost, avg_token_cost)
       d. Push new λ_lo into CONDORRewardManager
       e. PhiScorer.update(features, m_ids, qtype_ids, f1_targets)
       f. TrajectoryLogger.log(...)
       g. Every cluster_report_freq steps: ClusterReport.analyze(...)
  7. Log all metrics to W&B + MetricsCollector CSV/JSON
```

---

## Backward Compatibility

Setting `condor.use_condor=false` in any config causes `CONDORRayPPOTrainer.fit()` to delegate immediately to `super().fit()` (the baseline `RayPPOTrainer`). No CONDOR components are initialised. Output is bit-for-bit identical to the baseline.
