"""Training entry point for TASPO."""

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_taspo(config)


def run_taspo(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env = OmegaConf.merge(
            default_runtime_env,
            ray_init_kwargs.get("runtime_env", {}),
        )
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(TASPOTaskRunner.remote().run.remote(config))


@ray.remote(num_cpus=1)
class TASPOTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        OmegaConf.resolve(config)
        printable_config = OmegaConf.to_container(config, resolve=True)
        analyzer_config = printable_config.get("algorithm", {}).get("taspo", {}).get("analyzer", {})
        if analyzer_config.get("api_key"):
            analyzer_config["api_key"] = "<redacted>"
        pprint(printable_config)

        if config.actor_rollout_ref.actor.strategy not in ["fsdp", "fsdp2"]:
            raise NotImplementedError("TASPO currently supports FSDP/FSDP2 actor workers")
        with open_dict(config):
            # Always pass the row multiplier so copied divisibility padding has
            # zero mass, even in the no-trajectory-balancing ablation.
            config.actor_rollout_ref.actor.use_taspo_trajectory_balance = True

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        if config.actor_rollout_ref.rollout.name == "vllm":
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0 and not is_version_ge(pkg="vllm", minver="0.7.3"):
                raise NotImplementedError("PPO LoRA is not supported before vLLM 0.7.3")

        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import (
            ActorRolloutRefWorker,
            AsyncActorRolloutRefWorker,
            CriticWorker,
        )

        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

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

        if config.reward_model.enable:
            from verl.workers.fsdp_workers import RewardModelWorker

            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        if config.reward_model.get("reward_manager", "episode") != "episode":
            raise NotImplementedError("TASPO agent training currently uses EpisodeRewardManager")
        from agent_system.reward_manager import EpisodeRewardManager

        reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        assert config.actor_rollout_ref.rollout.n == 1, "In verl+env, keep actor_rollout_ref.rollout.n=1 and set GRPO groups with env.rollout.n"

        from agent_system.multi_turn_rollout import TrajectoryCollector
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from agent_system.taspo import TASPOAnalyzer

        taspo_analyzer = None
        if bool(config.algorithm.taspo.enabled) and not bool(config.trainer.get("val_only", False)):
            taspo_analyzer = TASPOAnalyzer(config.algorithm.taspo.analyzer)

        print("[TASPO] Trajectory-aligned process supervision")
        print(f"[TASPO] enabled={config.algorithm.taspo.enabled}")
        print(f"[TASPO] analyzer_model={config.algorithm.taspo.analyzer.model}")
        print(f"[TASPO] credit=epsilon:{config.algorithm.taspo.credit.epsilon}, temperature:{config.algorithm.taspo.credit.temperature}, token_gap_clip:{config.algorithm.taspo.credit.token_gap_clip}")

        from verl.trainer.ppo.taspo_ray_trainer import TASPORayTrainer

        trainer = TASPORayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            taspo_analyzer=taspo_analyzer,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
