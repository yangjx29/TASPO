#!/usr/bin/env bash

set -euo pipefail

TASPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASPO_ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TASPO_ANALYZER_MODEL="${TASPO_ANALYZER_MODEL:-glm-5.2}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-150}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-10}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-16}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
TASPO_ENABLED="${TASPO_ENABLED:-true}"
TASPO_ANALYZER_REQUIRED="${TASPO_ANALYZER_REQUIRED:-false}"
TASPO_EPSILON="${TASPO_EPSILON:-0.4}"
TASPO_TEMPERATURE="${TASPO_TEMPERATURE:-0.5}"
TASPO_TOKEN_GAP_CLIP="${TASPO_TOKEN_GAP_CLIP:-2.0}"
TASPO_PI_TOKEN_BUDGET="${TASPO_PI_TOKEN_BUDGET:-256}"
TASPO_ANALYZER_WORKERS="${TASPO_ANALYZER_WORKERS:-4}"
SEED="${SEED:-0}"
LOGGER="${LOGGER:-console}"

taspo_require_analyzer() {
  if [[ "$TASPO_ENABLED" == "true" ]]; then
    : "${TASPO_ANALYZER_BASE_URL:?Set TASPO_ANALYZER_BASE_URL to an OpenAI-compatible chat-completions endpoint}"
  fi
}

taspo_logger_override() {
  if [[ "$LOGGER" == "wandb" ]]; then
    printf "%s" "['console','wandb']"
  else
    printf "%s" "['console']"
  fi
}

taspo_common_overrides() {
  local project_name="$1"
  local experiment_name="$2"
  local audit_dir="${TASPO_AUDIT_DIR:-outputs/taspo_audit/${experiment_name}}"
  local cache_dir="${TASPO_CACHE_DIR:-outputs/taspo_cache}"

  cat <<EOF
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
algorithm.taspo.enabled=${TASPO_ENABLED}
algorithm.taspo.teacher_pi_token_budget=${TASPO_PI_TOKEN_BUDGET}
algorithm.taspo.credit.epsilon=${TASPO_EPSILON}
algorithm.taspo.credit.temperature=${TASPO_TEMPERATURE}
algorithm.taspo.credit.token_gap_clip=${TASPO_TOKEN_GAP_CLIP}
algorithm.taspo.credit.trajectory_balance=True
algorithm.taspo.analyzer.enabled=${TASPO_ENABLED}
algorithm.taspo.analyzer.required=${TASPO_ANALYZER_REQUIRED}
algorithm.taspo.analyzer.model=${TASPO_ANALYZER_MODEL}
algorithm.taspo.analyzer.max_workers=${TASPO_ANALYZER_WORKERS}
algorithm.taspo.analyzer.audit_dir=${audit_dir}
algorithm.taspo.analyzer.cache_dir=${cache_dir}
actor_rollout_ref.model.path=${MODEL_PATH}
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE}
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE}
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}
actor_rollout_ref.rollout.enable_chunked_prefill=False
actor_rollout_ref.rollout.enforce_eager=False
actor_rollout_ref.rollout.free_cache_engine=False
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
actor_rollout_ref.ref.fsdp_config.param_offload=False
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.actor.use_invalid_action_penalty=False
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.enable_gradient_checkpointing=True
env.seed=${SEED}
env.rollout.n=${GROUP_SIZE}
trainer.project_name=${project_name}
trainer.experiment_name=${experiment_name}
trainer.n_gpus_per_node=${NUM_GPUS}
trainer.nnodes=1
trainer.total_training_steps=${TOTAL_STEPS}
trainer.total_epochs=${TOTAL_STEPS}
trainer.save_freq=${SAVE_FREQ}
trainer.test_freq=${TEST_FREQ}
trainer.val_before_train=False
trainer.logger=$(taspo_logger_override)
EOF
}
