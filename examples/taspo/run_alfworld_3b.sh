#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"
taspo_require_analyzer

export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-taspo_alfworld_qwen2.5_3b_seed${SEED}}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/verl-agent/text/train.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/verl-agent/text/test.parquet}"

python3 -m examples.data_preprocess.prepare \
  --mode text \
  --train_data_size "$TRAIN_BATCH_SIZE" \
  --val_data_size "$VAL_BATCH_SIZE"

COMMON=()
while IFS= read -r line; do COMMON+=("$line"); done < <(
  taspo_common_overrides "taspo_alfworld" "$EXPERIMENT_NAME"
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
  env.env_name=alfworld/AlfredTWEnv \
  env.max_steps="$MAX_STEPS" \
  env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV_WORKER:-0.1}" \
  trainer.ray_wait_register_center_timeout=600 \
  "$@"
