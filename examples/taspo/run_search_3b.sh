#!/usr/bin/env bash

SEARCH_TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
SEARCH_VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
source "$(dirname "$0")/common.sh"
taspo_require_analyzer

TRAIN_BATCH_SIZE="$SEARCH_TRAIN_BATCH_SIZE"
VAL_BATCH_SIZE="$SEARCH_VAL_BATCH_SIZE"
MAX_STEPS="${MAX_STEPS:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-taspo_search_qwen2.5_3b_seed${SEED}}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/searchR1_processed_direct/train.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/searchR1_processed_direct/test.parquet}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"

[[ -f "$TRAIN_DATA" ]] || { echo "Missing $TRAIN_DATA; run examples/data_preprocess/preprocess_search_r1_dataset.py"; exit 1; }
[[ -f "$VAL_DATA" ]] || { echo "Missing $VAL_DATA"; exit 1; }

export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'
COMMON=()
while IFS= read -r line; do COMMON+=("$line"); done < <(
  taspo_common_overrides "taspo_search" "$EXPERIMENT_NAME"
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
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.use_invalid_action_penalty=False \
  env.env_name=search \
  env.max_steps="$MAX_STEPS" \
  env.history_length=4 \
  env.search.search_url="$SEARCH_URL" \
  trainer.ray_wait_register_center_timeout=600 \
  "$@"
