# Experiment Results

Results from CONDOR H-GCRL training and evaluation runs on the Babel cluster.
All evaluations use 996-question caps from standard test/validation splits (see [evaluation.md](evaluation.md)).

---

## Checkpoint: `global_step_400` — HotpotQA GAE

**Training run:** `verl_condor_hotpot_gae / condor_qwen2.5_7b_hotpot_gae_10369967`
**SLURM job:** `10369967`
**Base model:** Qwen2.5-7B
**Training dataset:** HotpotQA (distractor, fullwiki)
**Algorithm:** PPO with GAE advantage estimation
**Checkpoint path:** `/data/user_data/jgibson2/condor/checkpoints/rl/verl_condor_hotpot_gae/condor_qwen2.5_7b_hotpot_gae_10369967/global_step_400/actor_hf`
**HuggingFace backup:** `jdaltonII02/condor-hotpot-gae`

### Eval setup

| Setting | Value |
|---------|-------|
| Eval script | `baseline/verl/trainer/main_eval_condor.py` |
| LLM agent | `google/gemini-2.5-flash-lite` via OpenRouter |
| Max turns | 5 |
| Retrieval | FAISS dense index (localhost:8000) |
| SLURM template | `slurm/eval_condor_dataset.slurm` |

### Results by dataset

| Dataset | Split | Questions | Accuracy | EM | F1 | Precision | Recall | Avg Turns | Multi-step % | SLURM Job |
|---------|-------|----------:|--------:|---:|---:|----------:|-------:|----------:|-------------:|-----------|
| NQ | test[-996:] | 996 | 37.65% | 23.90% | 36.03% | 34.95% | 45.37% | 1.98 | 47.2% | 11059310 |
| HotpotQA | val[:996] | 996 | 47.09% | 38.96% | 51.29% | 52.49% | 54.97% | 3.29 | 99.0% | 11065181 |
| MuSiQue | val[:996] | 996 | — | — | — | — | — | — | — | 11080713 |
| CWQ | test[:996] | 996 | — | — | — | — | — | — | — | — |
| TriviaQA | val[:996] | 996 | — | — | — | — | — | — | — | — |
| FRAMES | test (full) | 824 | — | — | — | — | — | — | — | — |
| KILT-NQ | val[:996] | 996 | — | — | — | — | — | — | — | — |
| StrategyQA | test (full) | 687 | — | — | — | — | — | — | — | — |
| AmbigQA | — | — | — | — | — | — | — | — | — | no data |
| 2WikiMultiHop | — | — | — | — | — | — | — | — | — | no data |
| BamBoogle | — | — | — | — | — | — | — | — | — | no data |
| PopQA | — | — | — | — | — | — | — | — | — | no data |

### Mechanism breakdown

#### NQ (996 questions)

| Mechanism | Count | F1 |
|-----------|------:|---:|
| M1 — single-step retrieval | 880 | 35.6% |
| M2 — multi-step retrieval | 99 | 43.3% |
| M4 | 9 | 27.0% |
| M3 | 8 | 7.5% |

#### HotpotQA (996 questions)

| Mechanism | Count | F1 |
|-----------|------:|---:|
| M1 — single-step retrieval | 706 | 48.9% |
| M2 — multi-step retrieval | 198 | 50.2% |
| M3 | 86 | 74.9% |
| M4 | 6 | 29.8% |

### Notes

- The checkpoint was trained on HotpotQA, which explains the higher accuracy on that dataset (47.1%) vs. NQ (37.7%).
- Multi-step routing is near-universal on HotpotQA (99%) and selective on NQ (47%), consistent with the router learning the training distribution.
- M2 outperforms M1 on NQ (43.3% vs. 35.6% F1), suggesting the multi-step mechanism generalises to single-hop questions as well.
- M3 on HotpotQA achieves the highest F1 of any mechanism (74.9%) on a relatively small slice (86 questions).

---

*Results appended as each eval job completes. See [evaluation.md](evaluation.md) for how to run evals and [training.md](training.md) for training details.*
