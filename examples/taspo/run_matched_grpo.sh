#!/usr/bin/env bash

set -euo pipefail

BENCHMARK="${1:?Usage: run_matched_grpo.sh <alfworld|search|webshop> [Hydra overrides...]}"
shift
export TASPO_ENABLED=false
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-matched_grpo_${BENCHMARK}_seed${SEED:-0}}"
exec bash "$(dirname "$0")/run_${BENCHMARK}_3b.sh" "$@"
