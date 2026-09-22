# Training and evaluation

[Back to README](../README.md) · [Installation](installation.md) · [Results](results.md)

## First run

After installing the environment and configuring the analyzer:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m scripts.taspo.preflight --benchmark alfworld --check-api
bash examples/taspo/smoke_alfworld.sh
```

The smoke test runs two policy updates under a separate experiment name, with checkpoint saving disabled.

## Train

Select the launcher for the environment:

```bash
bash examples/taspo/run_alfworld_3b.sh
bash examples/taspo/run_search_3b.sh
bash examples/taspo/run_webshop_3b.sh
```

Run each command in its corresponding environment. Search-QA requires the retriever to be running.

For example, save an ALFWorld run's console output:

```bash
mkdir -p logs
EXPERIMENT_NAME=taspo_alfworld_seed0 \
  bash examples/taspo/run_alfworld_3b.sh \
  2>&1 | tee logs/taspo_alfworld_seed0.log
```

## Configuration

These environment variables override the launcher defaults:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `MODEL_PATH` | `Qwen/Qwen2.5-3B-Instruct` | Hugging Face model ID or local model directory |
| `NUM_GPUS` | `4` | GPUs on the training node |
| `TRAIN_BATCH_SIZE` | `16` | Training task batch size |
| `VAL_BATCH_SIZE` | `128` | Validation batch size |
| `GROUP_SIZE` | `8` | Rollouts per task |
| `TOTAL_STEPS` | `150` | Policy updates |
| `SAVE_FREQ` | `20` | Checkpoint interval |
| `TEST_FREQ` | `10` | Validation interval |
| `SEED` | `0` | Environment seed |
| `TASPO_EPSILON` | `0.4` | Credit redistribution amplitude |
| `TASPO_TEMPERATURE` | `0.5` | Credit redistribution temperature |
| `TASPO_PI_TOKEN_BUDGET` | `256` | Privileged-guidance token budget |
| `TASPO_ANALYZER_WORKERS` | `4` | Concurrent analyzer workers |
| `TASPO_ANALYZER_REQUIRED` | `false` | Stop on analyzer request failure when `true` |
| `PPO_MICRO_BATCH_SIZE` | `16` | Actor micro-batch per GPU |
| `LOG_PROB_MICRO_BATCH_SIZE` | `16` | Scoring micro-batch per GPU |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.5` | vLLM memory allocation |
| `LOGGER` | `console` | Set to `wandb` for W&B and console logging |

WebShop defaults to actor micro-batch `8` and PPO mini-batch `64`; the shared PPO mini-batch default is `256`. Environment horizons default to 50 turns for ALFWorld, 4 for Search-QA, and 15 for WebShop.

For a different backbone or batch size:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct SEED=1 \
  bash examples/taspo/run_alfworld_3b.sh

TRAIN_BATCH_SIZE=128 \
  bash examples/taspo/run_search_3b.sh
```

Additional Hydra arguments can be appended to a launcher. The complete training configuration is in [`ppo_trainer.yaml`](../verl/trainer/config/ppo_trainer.yaml).

## Logging

```bash
wandb login
LOGGER=wandb bash examples/taspo/run_alfworld_3b.sh
```

| Output | Location |
| :--- | :--- |
| Model checkpoints | `checkpoints/<project>/<experiment>/global_step_*` |
| Analyzer response cache | `outputs/taspo_cache/` |
| Guidance audit records | `outputs/taspo_audit/<experiment>/step_*.jsonl` |

Credit and guidance diagnostics use the `taspo/` metric prefix. Audit records contain the extracted evidence and accepted guidance for inspecting a run.

## Resume

The trainer defaults to `trainer.resume_mode=auto`. Relaunch with the same experiment name to resume from its checkpoint directory, or select a checkpoint explicitly:

```bash
bash examples/taspo/run_alfworld_3b.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/path/to/global_step_100
```

## Outcome-only baseline

```bash
bash examples/taspo/run_matched_grpo.sh alfworld
bash examples/taspo/run_matched_grpo.sh search
bash examples/taspo/run_matched_grpo.sh webshop
```

This baseline disables privileged guidance while retaining TASPO's trajectory-level advantage calculation and loss aggregation. It accepts the same environment-variable overrides as the training launchers.

## Evaluate

```bash
bash examples/taspo/eval_checkpoint.sh \
  alfworld /path/to/global_step_150
```

The first argument can be `alfworld`, `search`, or `webshop`. Evaluation loads the training checkpoint, runs validation without model updates, and does not call the analyzer. Search-QA still requires its retriever. `VAL_BATCH_SIZE` and `VAL_DATA` select the evaluation batch size and input data.

## Common issues

**Out of memory.** Reduce the actor and scoring micro-batches, then adjust the vLLM memory fraction to fit the model and context length:

```bash
export PPO_MICRO_BATCH_SIZE=8
export LOG_PROB_MICRO_BATCH_SIZE=8
export VLLM_GPU_MEMORY_UTILIZATION=0.4
```

**Missing analyzer URL.** Export `TASPO_ANALYZER_BASE_URL` and `TASPO_ANALYZER_MODEL` in the shell that starts training. Use the preflight command with `--check-api` to test the connection.

**Missing ALFWorld data.** Run `alfworld-download -f` and check `ALFWORLD_DATA`.

**Search cannot connect.** Check that the retrieval service is running and that `SEARCH_URL` matches its address.

**No privileged guidance.** Groups without a usable successful sibling or accepted guidance use outcome-only credit. If analyzer requests are failing, inspect the error log; setting `TASPO_ANALYZER_REQUIRED=true` makes those failures stop the run.
