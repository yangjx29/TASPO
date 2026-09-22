#!/usr/bin/env bash

WEBSHOP_PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
WEBSHOP_PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
WEBSHOP_LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
source "$(dirname "$0")/common.sh"
taspo_require_analyzer

MAX_STEPS="${MAX_STEPS:-15}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-taspo_webshop_qwen2.5_3b_seed${SEED}}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/verl-agent/text/train.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/verl-agent/text/test.parquet}"
PPO_MINI_BATCH_SIZE="$WEBSHOP_PPO_MINI_BATCH_SIZE"
PPO_MICRO_BATCH_SIZE="$WEBSHOP_PPO_MICRO_BATCH_SIZE"
LOG_PROB_MICRO_BATCH_SIZE="$WEBSHOP_LOG_PROB_MICRO_BATCH_SIZE"

python3 -m examples.data_preprocess.prepare \
  --mode text \
  --train_data_size "$TRAIN_BATCH_SIZE" \
  --val_data_size "$VAL_BATCH_SIZE"

COMMON=()
while IFS= read -r line; do COMMON+=("$line"); done < <(
  taspo_common_overrides "taspo_webshop" "$EXPERIMENT_NAME"
)

python3 -m verl.trainer.main_taspo \
  "${COMMON[@]}" \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.use_invalid_action_penalty=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  env.env_name=Webshop \
  env.max_steps="$MAX_STEPS" \
  env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV_WORKER:-0.1}" \
  trainer.ray_wait_register_center_timeout=600 \
  "$@"
