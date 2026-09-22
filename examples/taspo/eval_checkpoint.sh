#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

BENCHMARK="${1:?Usage: eval_checkpoint.sh <alfworld|search|webshop> <global_step_checkpoint>}"
CHECKPOINT="${2:?Pass the global_step_N checkpoint directory}"
shift 2
[[ -d "$CHECKPOINT" ]] || { echo "Checkpoint directory does not exist: $CHECKPOINT"; exit 1; }

TASPO_ENABLED=false
TOTAL_STEPS=1
SAVE_FREQ=-1
TEST_FREQ=-1
EXPERIMENT_NAME="${EXPERIMENT_NAME:-taspo_eval_${BENCHMARK}}"

case "$BENCHMARK" in
  alfworld)
    export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
    TRAIN_DATA="${TRAIN_DATA:-$HOME/data/verl-agent/text/train.parquet}"
    VAL_DATA="${VAL_DATA:-$HOME/data/verl-agent/text/test.parquet}"
    ENV_NAME="alfworld/AlfredTWEnv"
    MAX_STEPS="${MAX_STEPS:-50}"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
    ;;
  search)
    TRAIN_DATA="${TRAIN_DATA:-$HOME/data/searchR1_processed_direct/train.parquet}"
    VAL_DATA="${VAL_DATA:-$HOME/data/searchR1_processed_direct/test.parquet}"
    ENV_NAME="search"
    MAX_STEPS="${MAX_STEPS:-4}"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    ;;
  webshop)
    TRAIN_DATA="${TRAIN_DATA:-$HOME/data/verl-agent/text/train.parquet}"
    VAL_DATA="${VAL_DATA:-$HOME/data/verl-agent/text/test.parquet}"
    ENV_NAME="Webshop"
    MAX_STEPS="${MAX_STEPS:-15}"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    ;;
  *) echo "Unknown benchmark: $BENCHMARK"; exit 1 ;;
esac

COMMON=()
while IFS= read -r line; do COMMON+=("$line"); done < <(
  taspo_common_overrides "taspo_eval" "$EXPERIMENT_NAME"
)

python3 -m verl.trainer.main_taspo \
  "${COMMON[@]}" \
  algorithm.taspo.enabled=False \
  algorithm.taspo.analyzer.enabled=False \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-512}" \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  data.return_raw_chat=True \
  env.env_name="$ENV_NAME" \
  env.max_steps="$MAX_STEPS" \
  env.search.search_url="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CHECKPOINT" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  "$@"
