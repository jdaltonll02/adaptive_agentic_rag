"""CONDOR main_ppo_condor.py

Entry point for CONDOR training.  Mirrors baseline main_ppo_agentic_rag.py
but instantiates CONDORRayPPOTrainer and CONDORRewardManager.

Default config is loaded from:
    adaptive_rag/configs/condor_nq.yaml   (or condor_hotpot.yaml)

Usage via SLURM (preferred):
    sbatch slurm/train_condor_nq.slurm

Usage via shell (development / quick test):
    bash run_condor.sh [extra Hydra overrides]

Usage direct (advanced):
    python -m verl.trainer.main_ppo_condor \
        --config-path /home/jgibson2/projects/adaptive_rag/configs \
        --config-name condor_nq \
        "wandb.experiment=my_run"
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
)
_BASELINE_ROOT = os.path.join(_PROJECT_ROOT, 'baseline')

for _p in [_PROJECT_ROOT, _BASELINE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Standard imports (from baseline infrastructure)
# ---------------------------------------------------------------------------
import re
import string
from collections import Counter

import hydra
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

# Baseline verl infrastructure
from verl import DataProto
from verl.trainer.ppo.ray_trainer_agentic_rag_2 import RewardManager as BaseRewardManager
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from verl.utils.ray_utils import RoleEnum as Role
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer_agentic_rag_2 import ResourcePoolManager

# CONDOR trainer
from verl.trainer.ppo.ray_trainer_condor import CONDORRayPPOTrainer, CONDORRewardManager  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset / metric helpers (copied from baseline main_ppo_agentic_rag.py)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_trailing_marker(text: str) -> str:
    marker = "<|im_end|>"
    return text[:-len(marker)] if text.endswith(marker) else text


def create_rl_dataset(data_files, tokenizer, config):
    from verl.utils.dataset.rl_dataset import RLHFDataset
    return RLHFDataset(
        parquet_files=data_files,
        tokenizer=tokenizer,
        prompt_key=config.data.get('prompt_key', 'prompt'),
        max_prompt_length=config.data.max_prompt_length,
        filter_prompts=config.data.get('filter_overlong_prompts', True),
        return_raw_chat=config.data.get('return_raw_chat', True),
        truncation=config.data.get('truncation', 'error'),
    )


def create_rl_sampler(dataset, config):
    from torch.utils.data import RandomSampler
    return RandomSampler(dataset)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path='../../../configs', config_name='condor_nq')
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    print(OmegaConf.to_yaml(config))

    # Ensure output directories exist (light outputs in repo, heavy on /data)
    import os as _os
    for _d in [
        _os.path.expandvars(str(config.trainer.get('log_dir', ''))),
        _os.path.expandvars(str(config.trainer.get('metrics_dir', ''))),
        _os.path.expandvars(str(config.trainer.get('plots_dir', ''))),
        _os.path.expandvars(str(config.trainer.get('checkpoint_dir', ''))),
    ]:
        if _d:
            _os.makedirs(_d, exist_ok=True)

    # ---- Tokenizer ----
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    # ---- Ray init ----
    if not ray.is_initialized():
        ray.init(
            runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true'}},
            num_cpus=config.trainer.get('ray_num_cpus', None),
        )

    # ---- Resource pool ----
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    if config.trainer.get('use_reference_policy', False):
        mapping[Role.RefPolicy] = global_pool_id

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )

    role_worker_mapping = {
        Role.ActorRollout: RayClassWithInitArgs(
            cls=RayWorkerGroup,
            role=Role.ActorRollout,
            worker_cls=ActorRolloutRefWorker,
        ),
        Role.Critic: RayClassWithInitArgs(
            cls=RayWorkerGroup,
            role=Role.Critic,
            worker_cls=CriticWorker,
        ),
    }

    # ---- Datasets ----
    train_files = OmegaConf.to_container(config.data.train_files)
    val_files = OmegaConf.to_container(config.data.val_files)
    train_dataset = create_rl_dataset(train_files, tokenizer, config)
    val_dataset = create_rl_dataset(val_files, tokenizer, config)
    train_sampler = create_rl_sampler(train_dataset, config)

    from torch.utils.data import DataLoader
    def collate_fn(batch):
        import numpy as np
        keys = batch[0].keys()
        out = {}
        for k in keys:
            vals = [b[k] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals)
            else:
                out[k] = np.array(vals, dtype=object)
        return out

    # ---- Reward manager ----
    reward_manager = CONDORRewardManager(tokenizer)

    reward_fn = None
    val_reward_fn = reward_manager

    # ---- Trainer ----
    trainer = CONDORRayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=None,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_manager=reward_manager,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        device_name='cuda',
    )

    trainer.fit()


if __name__ == '__main__':
    main()
