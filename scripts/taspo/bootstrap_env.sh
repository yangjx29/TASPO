#!/usr/bin/env bash

set -euo pipefail

BENCHMARK="${1:?Usage: bootstrap_env.sh <alfworld|search|webshop>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 -m pip install --upgrade pip

case "$BENCHMARK" in
  alfworld)
    python3 -m pip install vllm==0.11.0
    python3 -m pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
    alfworld-download -f
    ;;
  search)
    python3 -m pip install vllm==0.11.0 gym==0.26.2
    python3 -m pip install -e agent_system/environments/env_package/search/third_party
    ;;
  webshop)
    python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    python3 -m pip install vllm==0.8.2
    (
      cd agent_system/environments/env_package/webshop/webshop
      bash setup.sh -d all
    )
    ;;
  *) echo "Unknown benchmark: $BENCHMARK"; exit 1 ;;
esac

python3 -m pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
python3 -m pip install -e ".[test]"

echo "Environment installed. Run scripts/taspo/preflight.py before training."
