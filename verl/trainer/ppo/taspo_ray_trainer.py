"""Ray trainer for TASPO."""

from __future__ import annotations

from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from agent_system.taspo.action_mask import build_action_token_mask
from agent_system.taspo.credit import build_taspo_advantages
from agent_system.taspo.teacher import build_taspo_teacher_batch
from agent_system.multi_turn_rollout import adjust_batch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.metric import reduce_metrics


class TASPORayTrainer(RayPPOTrainer):
    """GRPO with trajectory-aligned PI and conservative action allocation."""

    def __init__(self, *args, taspo_analyzer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.taspo_analyzer = taspo_analyzer
        config = self.config.algorithm.get("taspo", {})
        credit = config.get("credit", {})
        self.taspo_enabled = bool(config.get("enabled", True))
        self.credit_epsilon = float(credit.get("epsilon", 0.4))
        self.credit_temperature = float(credit.get("temperature", 0.5))
        self.token_gap_clip = float(credit.get("token_gap_clip", 2.0))
        self.trajectory_balance = bool(credit.get("trajectory_balance", True))
        self.pi_token_budget = int(config.get("teacher_pi_token_budget", 256))

        if str(self.config.algorithm.adv_estimator).lower() != "grpo":
            raise ValueError("TASPO currently requires algorithm.adv_estimator=grpo")
        if self.config.algorithm.use_kl_in_reward:
            raise ValueError("TASPO anchors credit to outcome GRPO and does not support KL-in-reward")
        if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", False):
            raise ValueError("TASPO determines the trajectory direction from outcome reward; disable the local invalid-action penalty")
        if self.config.actor_rollout_ref.actor.loss_agg_mode != "seq-mean-token-mean":
            raise ValueError("TASPO requires actor.loss_agg_mode=seq-mean-token-mean for action-level token normalization")

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="TASPO Training",
        )
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                for key in ["multi_modal_data", "raw_prompt", "tools_kwargs", "env_kwargs"]:
                    if key in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append(key)
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        batch = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    action_mask, action_span_found = build_action_token_mask(batch, self.tokenizer)
                    batch.batch["taspo_action_mask"] = action_mask
                    batch.non_tensor_batch["taspo_action_span_found"] = action_span_found.cpu().numpy()
                    metrics["taspo/action_span_parse_rate"] = action_span_found.float().mean().item()

                    with _timer("pi_alignment", timing_raw):
                        if self.taspo_enabled and self.taspo_analyzer is not None:
                            batch, analyzer_metrics = self.taspo_analyzer.annotate_batch(
                                batch,
                                global_step=self.global_steps,
                            )
                            metrics.update(analyzer_metrics)
                        else:
                            size = len(batch)
                            batch.non_tensor_batch["taspo_teacher_pi"] = np.array([""] * size, dtype=object)
                            batch.non_tensor_batch["taspo_pi_available"] = np.zeros(size, dtype=bool)
                            batch.non_tensor_batch["taspo_pi_reason"] = np.array(
                                ["taspo_disabled"] * size,
                                dtype=object,
                            )
                            batch.non_tensor_batch["taspo_guidance_count"] = np.zeros(size, dtype=np.int64)
                            metrics["taspo/pi_trajectory_coverage"] = 0.0
                            metrics["taspo/pi_row_coverage"] = 0.0
                    self._drop_analyzer_trace(batch)

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("adv", timing_raw):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({key: np.array(value) for key, value in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo",
                                True,
                            ),
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            compute_mean_std_cross_steps=False,
                        )

                    # Model workers require divisible batches. Copies are marked
                    # and receive zero objective mass after all forwards.
                    batch = adjust_batch(self.config, batch, mark_padding=True)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropies = old_log_prob.batch.pop("entropys")
                        entropy_loss = agg_loss(
                            loss_mat=entropies,
                            loss_mask=batch.batch["response_mask"],
                            loss_agg_mode="seq-mean-token-mean",
                        )
                        metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                        batch = batch.union(old_log_prob)

                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs, effective_pi = self._compute_teacher_log_probs(batch)
                        batch.batch["teacher_log_probs"] = teacher_log_probs
                        batch.non_tensor_batch["taspo_pi_effective"] = effective_pi.cpu().numpy()
                        real_rows = ~batch.non_tensor_batch["is_padding_row"]
                        metrics["taspo/teacher_row_ratio"] = float(effective_pi.cpu().numpy()[real_rows].mean())

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with _timer("credit", timing_raw):
                        shaped_advantages, credit_tensors, credit_metrics = build_taspo_advantages(
                            seq_advantages=batch.batch["advantages"],
                            student_log_probs=batch.batch["old_log_probs"],
                            teacher_log_probs=batch.batch["teacher_log_probs"],
                            response_mask=batch.batch["response_mask"],
                            action_mask=batch.batch["taspo_action_mask"],
                            traj_uids=batch.non_tensor_batch["traj_uid"],
                            turn_steps=batch.non_tensor_batch["turn_step"],
                            pi_available=batch.non_tensor_batch["taspo_pi_effective"],
                            padding_rows=batch.non_tensor_batch["is_padding_row"],
                            epsilon=self.credit_epsilon,
                            temperature=self.credit_temperature,
                            token_gap_clip=self.token_gap_clip,
                            trajectory_balance=self.trajectory_balance,
                        )
                        batch.batch["advantages"] = shaped_advantages
                        batch.batch["taspo_step_score"] = credit_tensors["step_score"]
                        batch.batch["taspo_action_weight"] = credit_tensors["action_weight"]
                        batch.batch["taspo_trajectory_balance"] = credit_tensors["trajectory_balance"]
                        metrics.update(credit_metrics)
                        self._drop_training_sidecars(batch)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            dump_batch = self._without_padding(batch)
                            inputs = self.tokenizer.batch_decode(dump_batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(dump_batch.batch["responses"], skip_special_tokens=True)
                            scores = dump_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            dump_reward_infos = {key: dump_batch.non_tensor_batch[key].tolist() for key in reward_extra_infos_dict if key in dump_batch.non_tensor_batch}
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=dump_reward_infos,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metric_batch = self._without_padding(batch)
                metrics.update(compute_data_metrics(batch=metric_batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=metric_batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=metric_batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _compute_teacher_log_probs(self, batch: DataProto) -> tuple[torch.Tensor, torch.Tensor]:
        requested = torch.as_tensor(
            batch.non_tensor_batch["taspo_pi_available"],
            dtype=torch.bool,
            device=batch.batch["old_log_probs"].device,
        )
        padding = torch.as_tensor(
            batch.non_tensor_batch["is_padding_row"],
            dtype=torch.bool,
            device=requested.device,
        )
        requested = requested & ~padding
        if not self.taspo_enabled or not torch.any(requested):
            return batch.batch["old_log_probs"].detach().clone(), torch.zeros_like(requested)

        requested_indices = requested.nonzero(as_tuple=True)[0]
        requested_batch = batch.select_idxs(requested_indices.cpu().numpy())
        teacher_batch, effective = build_taspo_teacher_batch(
            batch=requested_batch,
            tokenizer=self.tokenizer,
            pi_token_budget=self.pi_token_budget,
        )
        divisor = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * self.actor_rollout_wg.world_size
        padded_teacher_batch, pad_size = pad_dataproto_to_divisor(
            teacher_batch,
            divisor,
        )
        output = self.actor_rollout_wg.compute_log_prob(padded_teacher_batch)
        output = unpad_dataproto(output, pad_size=pad_size)

        teacher_log_probs = batch.batch["old_log_probs"].detach().clone()
        teacher_log_probs[requested_indices] = output.batch["old_log_probs"]
        full_effective = torch.zeros_like(requested)
        full_effective[requested_indices] = effective
        return teacher_log_probs, full_effective

    @staticmethod
    def _without_padding(batch: DataProto) -> DataProto:
        padding = batch.non_tensor_batch.get("is_padding_row")
        if padding is None or not np.any(padding):
            return batch
        result = batch.select_idxs(np.flatnonzero(~padding))
        result.meta_info = dict(batch.meta_info)
        result.meta_info["global_token_num"] = torch.sum(result.batch["attention_mask"], dim=-1).tolist()
        return result

    @staticmethod
    def _drop_training_sidecars(batch: DataProto) -> None:
        """Avoid shipping analyzer text through Ray during the actor update."""
        TASPORayTrainer._drop_analyzer_trace(batch)
        for key in [
            "teacher_log_probs",
            "taspo_action_mask",
            "taspo_step_score",
            "taspo_action_weight",
        ]:
            if key in batch.batch:
                batch.batch.pop(key)
        for key in [
            "taspo_teacher_pi",
            "taspo_pi_available",
            "taspo_pi_effective",
            "taspo_pi_reason",
            "taspo_guidance_count",
            "taspo_action_span_found",
            "turn_step",
            "uid",
            "gamefile",
        ]:
            batch.non_tensor_batch.pop(key, None)

    @staticmethod
    def _drop_analyzer_trace(batch: DataProto) -> None:
        """Release large text traces once PI alignment has been audited."""
        for key in [
            "task_text",
            "pre_observation",
            "post_observation",
            "model_response_text",
            "action_text",
            "env_done",
            "verified_success",
        ]:
            batch.non_tensor_batch.pop(key, None)
