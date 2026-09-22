#!/usr/bin/env python3
"""Check an TASPO server before launching an expensive training run."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REQUIRED_MODULES = [
    "torch",
    "ray",
    "vllm",
    "transformers",
    "hydra",
    "numpy",
    "openai",
]


def check_modules() -> list[str]:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    for name in REQUIRED_MODULES:
        state = "MISSING" if name in missing else "ok"
        print(f"[module] {name}: {state}")
    if "torch" not in missing:
        import torch

        print(f"[gpu] cuda_available={torch.cuda.is_available()}")
        print(f"[gpu] visible_device_count={torch.cuda.device_count()}")
    return missing


def check_benchmark(name: str) -> list[str]:
    problems = []
    home = Path.home()
    if name == "alfworld":
        root = Path(os.getenv("ALFWORLD_DATA", home / ".cache/alfworld")).expanduser()
        print(f"[benchmark] ALFWORLD_DATA={root}")
        if not root.exists():
            problems.append(f"ALFWorld data directory is missing: {root}")
        for relative in ["logic/alfred.pddl", "logic/alfred.twl2"]:
            path = root / relative
            if not path.is_file():
                problems.append(f"ALFWorld logic file is missing: {path}")
    elif name == "search":
        train = Path(os.getenv("TRAIN_DATA", home / "data/searchR1_processed_direct/train.parquet")).expanduser()
        val = Path(os.getenv("VAL_DATA", home / "data/searchR1_processed_direct/test.parquet")).expanduser()
        print(f"[benchmark] train={train}")
        print(f"[benchmark] val={val}")
        if not train.is_file():
            problems.append(f"Search train parquet is missing: {train}")
        if not val.is_file():
            problems.append(f"Search validation parquet is missing: {val}")
        print(f"[benchmark] SEARCH_URL={os.getenv('SEARCH_URL', 'http://127.0.0.1:8000/retrieve')}")
    elif name == "webshop":
        package = Path("agent_system/environments/env_package/webshop/webshop").resolve()
        print(f"[benchmark] WebShop package={package}")
        if not package.exists():
            problems.append(f"WebShop source directory is missing: {package}")
    return problems


def check_analyzer(call_api: bool) -> list[str]:
    problems = []
    key = os.getenv("TASPO_ANALYZER_API_KEY", "")
    base_url = os.getenv("TASPO_ANALYZER_BASE_URL", "")
    model = os.getenv("TASPO_ANALYZER_MODEL", "your-analyzer-model")
    print(f"[analyzer] model={model}")
    print(f"[analyzer] base_url={base_url or 'MISSING'}")
    print(f"[analyzer] api_key={'set' if key else 'not set (allowed for local endpoints)'}")
    if not base_url:
        problems.append("TASPO_ANALYZER_BASE_URL is not set")
    if call_api and not problems:
        from agent_system.taspo.client import AnalyzerClientConfig, OpenAIJSONClient

        client = OpenAIJSONClient(
            AnalyzerClientConfig(
                model=model,
                base_url=base_url,
                api_key=key,
                max_retries=1,
                max_tokens=64,
                cache_dir="outputs/taspo_cache/preflight",
            )
        )
        result = client.complete_json(
            "Return exactly one JSON object with the key ok and value true.",
            {"request": "TASPO preflight"},
            "ping",
        )
        if result.get("ok") is not True:
            problems.append(f"Analyzer returned an unexpected response: {result}")
        else:
            print("[analyzer] live request: ok")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["alfworld", "search", "webshop"], required=True)
    parser.add_argument("--skip-analyzer", action="store_true")
    parser.add_argument("--check-api", action="store_true")
    args = parser.parse_args()

    problems = check_modules()
    problems.extend(check_benchmark(args.benchmark))
    if not args.skip_analyzer:
        problems.extend(check_analyzer(args.check_api))
    if problems:
        print("\nPreflight failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("\nTASPO preflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
