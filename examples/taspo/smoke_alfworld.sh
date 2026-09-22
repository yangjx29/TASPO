#!/usr/bin/env bash

set -euo pipefail

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
export GROUP_SIZE="${GROUP_SIZE:-2}"
export TOTAL_STEPS="${TOTAL_STEPS:-2}"
export MAX_STEPS="${MAX_STEPS:-5}"
export SAVE_FREQ=-1
export TEST_FREQ=-1
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-1}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-taspo_alfworld_smoke}"

exec bash "$(dirname "$0")/run_alfworld_3b.sh" "$@"
