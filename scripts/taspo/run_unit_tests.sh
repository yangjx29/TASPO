#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 -m compileall -q agent_system/taspo verl/trainer/main_taspo.py verl/trainer/ppo/taspo_ray_trainer.py
python3 -m unittest -v \
  tests.taspo.test_client \
  tests.taspo.test_analyzer \
  tests.taspo.test_action_mask \
  tests.taspo.test_teacher \
  tests.taspo.test_credit \
  tests.taspo.test_pipeline
