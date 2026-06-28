# Configuration Reference

CONDOR uses [Hydra](https://hydra.cc/) for configuration. Every training and evaluation run is controlled by YAML files under `configs/`. Fields can be overridden on the command line using Hydra dot-notation.

---

## File Structure

```
configs/
├── paths.yaml              ← Dataset and output paths (shared by all configs)
├── condor_nq.yaml          ← Training on NQ
├── condor_hotpot.yaml      ← Training on HotpotQA
├── eval_condor_nq.yaml     ← Evaluation on NQ
├── eval_condor_hotpot.yaml ← Evaluation on HotpotQA
├── eval_condor_2wiki.yaml
├── eval_condor_ambigqa.yaml
├── eval_condor_musique.yaml
├── eval_condor_bamboogle.yaml
└── eval_condor_popqa.yaml
```

All training and eval configs start with:

```yaml
defaults:
  - paths
  - _self_
```

This merges `paths.yaml` first, then the local file overrides.

---

## `paths.yaml` — Shared Paths

```yaml
paths:
  # Root data directories
  aqawflow_data: /data/user_data/jgibson2/aqawflow/data
  condor_data:   /data/user_data/jgibson2/condor/data

  # Dataset parquet files
  nq:
    train: /data/user_data/jgibson2/condor/data/nq/train.parquet
    test:  /data/user_data/jgibson2/aqawflow/data/nq/test_996.parquet
  hotpot_qa:
    train: /data/user_data/jgibson2/aqawflow/data/hotpot_qa/train.parquet
    test:  /data/user_data/jgibson2/aqawflow/data/hotpot_qa/test_996.parquet
  ambig_qa:
    train: /data/user_data/jgibson2/aqawflow/data/ambig_qa/train.parquet
    test:  /data/user_data/jgibson2/aqawflow/data/ambig_qa/test_996.parquet
  musique:
    train: /data/user_data/jgibson2/aqawflow/data/musique/train.parquet
    test:  /data/user_data/jgibson2/aqawflow/data/musique/test_996.parquet
  two_wiki:
    train: /data/user_data/jgibson2/aqawflow/data/2wikimultihop_qa/train.parquet
    test:  /data/user_data/jgibson2/aqawflow/data/2wikimultihop_qa/test_996.parquet
  bamboogle:
    test:  /data/user_data/jgibson2/aqawflow/data/bamboogle/test.parquet
  pop_qa:
    test:  /data/user_data/jgibson2/aqawflow/data/pop_qa/test_996.parquet

  # Model checkpoints
  sft_checkpoint: /data/user_data/jgibson2/aqawflow/checkpoints/sft/qwen2.5_1.5b/global_step_11
  rl_checkpoint_dir: /data/user_data/jgibson2/condor/checkpoints/rl

  # Output directories
  logs_dir:    /data/user_data/jgibson2/condor/logs
  metrics_dir: outputs/metrics
  plots_dir:   outputs/plots
```

---

## Training Config Fields

### `wandb`

| Field | Description |
|-------|-------------|
| `project` | W&B project name |
| `experiment` | Run name; also used as the experiment prefix for metrics files |

### `data`

| Field | Description |
|-------|-------------|
| `train_files` | List of parquet paths for training (supports multiple) |
| `val_files` | List of parquet paths for validation |
| `val_dataset` | Short dataset name used in metric logging (e.g. `nq`, `hotpot`) |
| `train_batch_size` | Number of questions per training batch |
| `val_batch_size` | Number of questions per validation batch |
| `max_prompt_length` | Token limit for question + system message |
| `max_response_length` | Token limit for planner output per turn |
| `prompt_key` | Column name in parquet containing the question (always `prompt`) |
| `trust_remote_code` | Passed to HuggingFace tokenizer loader |

### `model`

| Field | Description |
|-------|-------------|
| `path` | Path to SFT checkpoint |
| `tokenizer_path` | Defaults to `model.path` if omitted |

### `trainer`

| Field | Description |
|-------|-------------|
| `project_name` | W&B project (mirrors `wandb.project`) |
| `experiment_name` | Run name (mirrors `wandb.experiment`) |
| `logger` | List: `[wandb]`, `[console]`, or `[wandb, console]` |
| `n_gpus_per_node` | GPUs per node (1 for dev, 4 for full training) |
| `nnodes` | Number of nodes |
| `ray_num_cpus` | CPU cores for Ray; `null` = auto-detect |
| `log_dir` | SLURM stdout/stderr base directory |
| `metrics_dir` | Where `MetricsCollector` writes CSV and JSON files |
| `plots_dir` | Where figures are written |
| `checkpoint_dir` | Where model checkpoints are saved |
| `save_freq` | Save checkpoint every N steps |
| `val_freq` | Run validation every N steps |
| `total_training_steps` | Total gradient update steps |
| `cluster_report_freq` | Run K-Means cluster analysis every N steps (default 50) |

### `ppo`

Standard PPO hyperparameters inherited from `RayPPOTrainer`. Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `gamma` | 1.0 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `clip_ratio` | 0.2 | PPO clip coefficient |
| `entropy_coeff` | 0.01 | Entropy bonus weight |
| `vf_coeff` | 0.5 | Value function loss weight |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `lr` | 1e-6 | Actor learning rate |
| `vf_lr` | 1e-5 | Critic learning rate |
| `ppo_epochs` | 2 | Number of PPO update epochs per batch |
| `mini_batch_size` | 8 | Mini-batch size within each PPO epoch |

### `condor`

All CONDOR-specific hyperparameters. Setting `use_condor: false` makes this entire block a no-op.

| Field | Default | Description |
|-------|---------|-------------|
| `use_condor` | `true` | Master switch — `false` reproduces exact baseline |
| `epsilon_start` | `0.9` | Initial exploration rate for mechanism router |
| `epsilon_end` | `0.1` | Final exploration rate |
| `epsilon_decay` | `1000` | Exponential decay constant (steps) |
| `router_lr` | `1e-3` | MechanismRouter Adam learning rate |
| `budget_hi` | `0.3` | L0 mechanism cost constraint target |
| `budget_lo` | `0.5` | L1 token cost constraint target |
| `lambda_hi_init` | `0.1` | Initial value of `λ_hi` |
| `lambda_lo_init` | `0.1` | Initial value of `λ_lo` |
| `lambda_max` | `10.0` | Maximum Lagrange multiplier |
| `dual_alpha_lr` | `1e-3` | Adam learning rate for multiplier updates |
| `phi_lr` | `1e-3` | PhiScorer MLP learning rate |
| `phi_t_warmup` | `500` | Step at which blending starts (`α_t = 0` before this) |
| `phi_tau_trans` | `100` | Blend transition timescale (steps) |
| `trajectory_buffer_size` | `2000` | Rolling trajectory buffer size |
| `cluster_report_freq` | `50` | K-Means analysis frequency (steps) |
| `l0_reward_threshold` | `0.0` | Minimum L0 reward for router update (use `0.0` to update always) |

---

## Evaluation Config Fields

Eval configs use the same `paths`, `wandb`, `data`, and `trainer` sections, but `data` has only `val_files` (no `train_files`), and `condor` is simplified:

```yaml
condor:
  use_condor: true
  l0_reward_threshold: 0.4   # lower bound for counting a mechanism selection as "rewarded"
  epsilon_start: 0.0          # no exploration during eval
  epsilon_end:   0.0
  epsilon_decay: 1.0
```

The `eval` section is unique to eval configs:

| Field | Description |
|-------|-------------|
| `checkpoint_path` | Path to the RL checkpoint to evaluate; defaults to `latest` |
| `router_checkpoint` | Optional separate checkpoint for the MechanismRouter; `null` uses the one bundled in the main checkpoint |
| `max_turn` | Maximum planning turns per question |
| `eval_batch_size` | Questions per eval batch (lower than training to reduce GPU memory) |

---

## Command-Line Overrides

Any YAML field can be overridden using Hydra's dot-notation:

```bash
# Override learning rate
python -m verl.trainer.main_condor \
    --config-name condor_nq \
    condor.router_lr=5e-4

# Override checkpoint path at eval time
sbatch slurm/eval_condor_nq.slurm \
    "eval.checkpoint_path=/data/user_data/jgibson2/condor/checkpoints/rl/step_1000"

# Disable CONDOR (reproduce baseline)
python -m verl.trainer.main_condor \
    --config-name condor_nq \
    condor.use_condor=false
```

---

## Adding a New Dataset Config

1. Copy an existing eval config: `cp configs/eval_condor_nq.yaml configs/eval_condor_<name>.yaml`
2. Update `data.val_files` to point to the new parquet.
3. Update `data.val_dataset` to a short name (used in metric file prefixes).
4. Update `wandb.experiment`.
5. Create a SLURM script: `cp slurm/eval_condor_nq.slurm slurm/eval_condor_<name>.slurm` and update the config name.
