"""CONDOR standalone evaluation entry point.

Runs the full multi-turn rollout + metric computation on a held-out split
using a trained RL checkpoint.  Reuses RayPPOTrainer in val_only mode.

Usage (via slurm/eval_condor_nq.slurm):
    python -m verl.trainer.main_eval_condor \
        --config-path /path/to/configs \
        --config-name eval_condor_nq \
        "eval.checkpoint_path=/data/.../condor/checkpoints/<id>"
"""

import hydra
import ray
from omegaconf import OmegaConf, DictConfig

from verl.trainer.main_ppo_condor import (
    CondorRewardManager,
    _create_rl_dataset,
    _create_rl_sampler,
)
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):
    run_eval(config)


def run_eval(config: DictConfig) -> None:
    # Force val_only so the trainer skips training and runs one eval pass
    OmegaConf.update(config, "trainer.val_only", True, merge=True)
    OmegaConf.update(config, "trainer.val_before_train", True, merge=True)

    # Map eval-specific keys onto trainer keys the PPO trainer expects
    if OmegaConf.select(config, "eval.checkpoint_path") is not None:
        OmegaConf.update(
            config,
            "actor_rollout_ref.model.path",
            config.eval.checkpoint_path,
            merge=True,
        )
        OmegaConf.update(
            config, "critic.model.path", config.eval.checkpoint_path, merge=True
        )

    if not ray.is_initialized():
        num_cpus = OmegaConf.select(config, "trainer.ray_num_cpus")
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            },
            num_cpus=num_cpus,
        )

    runner = EvalRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class EvalRunner:
    def run(self, config: DictConfig):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Worker classes (eval only needs actor, no critic training)
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.trainer.ppo.ray_trainer_agentic_rag_2 import (
            ResourcePoolManager,
            Role,
            RayPPOTrainer,
        )

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        from verl.utils.dataset.rl_dataset import collate_fn

        reward_manager = CondorRewardManager(tokenizer, condor_cfg=config.condor)

        # Eval uses val_files only; supply a trivial train dataset
        val_dataset = _create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor
        )
        train_dataset = val_dataset  # unused in val_only mode

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_manager=reward_manager,
            reward_fn=reward_fn,
            val_reward_fn=reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=_create_rl_sampler(config.data, train_dataset),
            device_name=config.trainer.device if hasattr(config.trainer, "device") else "cuda",
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
