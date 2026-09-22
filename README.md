<p align="center">
  <img src="docs/assets/taspo.svg" alt="TASPO — Process supervision, grounded in outcomes." width="880">
</p>

<h2 align="center">Reconciling Process Supervision with<br>Outcome-Based Credit in Agentic Policy Optimization</h2>

<p align="center">
  <a href="docs/paper.pdf"><b>Paper</b></a> &nbsp; · &nbsp;
  <a href="#results"><b>Results</b></a> &nbsp; · &nbsp;
  <a href="#getting-started"><b>Getting Started</b></a> &nbsp; · &nbsp;
  <a href="docs/usage.md"><b>Documentation</b></a> &nbsp; · &nbsp;
  <a href="README_zh-CN.md">中文</a>
</p>

---

We introduce **TASPO**, a method for bringing process supervision into outcome-based agent reinforcement learning. TASPO turns successful sibling trajectories into guidance for the trajectory being trained, then uses that guidance to redistribute credit across its actions.

**The outcome sets the advantage sign; process supervision refines the allocation.** Within a failed trajectory, better-supported actions receive less negative credit; within a successful one, they receive more positive credit.

- **Trajectory-aligned guidance.** Extract useful evidence from successful rollouts and match it to the target's own interaction history.
- **Action-level credit.** Score executable actions with and without privileged guidance, then redistribute the original advantage with positive, bounded, mean-one weights.
- **Reuse the rollouts.** Guidance construction and rescoring happen during training, with no additional environment interaction. Evaluation uses the policy alone.

<p align="center">
  <a href="docs/assets/method.png"><img src="docs/assets/method.png" alt="TASPO overview: trajectory-aligned process supervision and outcome-consistent credit redistribution." width="100%"></a>
</p>

<p align="center"><sub>Successful siblings provide the evidence. The target trajectory determines where it applies. Outcome-based advantages remain the basis of the update.</sub></p>

## Results

TASPO improves over GRPO on **ALFWorld, Search-QA, and WebShop** across all three backbones in our experiments. With Qwen2.5-3B-Instruct, the gains are **14.4 points on ALFWorld**, **10.0 points on Search-QA**, and **14.8 points in WebShop success rate**.

| Backbone | Method | ALFWorld ↑ | Search-QA ↑ | WebShop ↑ |
| :--- | :--- | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | GRPO | 71.8 | 36.4 | 63.3 |
| | **TASPO** | **86.2** | **46.4** | **78.1** |
| Qwen2.5-7B-Instruct | GRPO | 77.0 | 48.1 | 75.0 |
| | **TASPO** | **90.1** | **49.8** | **76.4** |
| Qwen3-1.7B-Instruct | GRPO | 40.1 | 43.6 | 43.0 |
| | **TASPO** | **69.6** | **46.6** | **67.1** |

Results from Table 1 of our [paper](docs/paper.pdf), in percent. ALFWorld reports macro-averaged task-family success, Search-QA reports macro-averaged exact-match accuracy, and WebShop reports success rate. See [full benchmark results](docs/results.md) for additional baselines and WebShop scores.

## Getting started

Start with ALFWorld. The default launcher uses Qwen2.5-3B-Instruct on **4 NVIDIA GPUs**, with Linux, Conda, and Python 3.12. Run the commands below from the repository root. Search-QA and WebShop have separate [setup instructions](docs/installation.md).

### 1. Install

```bash
conda create -n taspo python=3.12 -y
conda activate taspo
bash scripts/taspo/bootstrap_env.sh alfworld
```

### 2. Set up the analyzer

TASPO uses a training-time analyzer to construct privileged guidance. Set an OpenAI-compatible chat-completions endpoint and the model served by it:

```bash
export TASPO_ANALYZER_BASE_URL="https://your-endpoint/v1/chat/completions"
export TASPO_ANALYZER_MODEL="your-analyzer-model"
export TASPO_ANALYZER_API_KEY="your-api-key"
export TASPO_ANALYZER_REQUIRED=true
```

The API key can be omitted for a local endpoint without authentication. `TASPO_ANALYZER_REQUIRED=true` stops the run if an analyzer request fails.

### 3. Run

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Check the environment and analyzer connection.
python -m scripts.taspo.preflight --benchmark alfworld --check-api

# Run a two-step smoke test.
bash examples/taspo/smoke_alfworld.sh

# Train with the default configuration.
bash examples/taspo/run_alfworld_3b.sh
```

Checkpoints are written to `checkpoints/<project>/<experiment>/`. Launchers accept environment-variable overrides and additional Hydra arguments; see the [training guide](docs/usage.md) for logging, evaluation, and resuming a run.

## Training and evaluation

| Task | Setup | Training entry |
| :--- | :--- | :--- |
| ALFWorld | [Environment](docs/installation.md#alfworld) | [`run_alfworld_3b.sh`](examples/taspo/run_alfworld_3b.sh) |
| Search-QA | [Data and retriever](docs/installation.md#search-qa) | [`run_search_3b.sh`](examples/taspo/run_search_3b.sh) |
| WebShop | [Environment](docs/installation.md#webshop) | [`run_webshop_3b.sh`](examples/taspo/run_webshop_3b.sh) |

The launchers default to Qwen2.5-3B-Instruct. Use `MODEL_PATH` to select another backbone and `SEED` to select the training seed:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct SEED=1 \
  bash examples/taspo/run_alfworld_3b.sh
```

Run the matched outcome-only baseline or evaluate a saved checkpoint:

```bash
bash examples/taspo/run_matched_grpo.sh alfworld

bash examples/taspo/eval_checkpoint.sh \
  alfworld /path/to/global_step_150
```

Evaluation does not call the analyzer. For the benchmark settings and metrics used in the paper, see [results and evaluation](docs/results.md).

## Code

The TASPO components are in [`agent_system/taspo/`](agent_system/taspo), with training built on the verl ecosystem.

| Component | Source |
| :--- | :--- |
| Guidance construction and alignment | [`analyzer.py`](agent_system/taspo/analyzer.py), [`prompts.py`](agent_system/taspo/prompts.py) |
| Executable-action masks | [`action_mask.py`](agent_system/taspo/action_mask.py) |
| Privileged-context scoring inputs | [`teacher.py`](agent_system/taspo/teacher.py) |
| Advantage redistribution | [`credit.py`](agent_system/taspo/credit.py) |
| Training entry and orchestration | [`main_taspo.py`](verl/trainer/main_taspo.py), [`taspo_ray_trainer.py`](verl/trainer/ppo/taspo_ray_trainer.py) |

To run the core tests:

```bash
bash scripts/taspo/run_unit_tests.sh
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for bug reports and changes.

## Citation

If you use TASPO in your research, please cite our paper:

> **Reconciling Process Supervision with Outcome-Based Credit in Agentic Policy Optimization.** [Paper](docs/paper.pdf)

## Acknowledgments

Our implementation builds on [verl](https://github.com/volcengine/verl), [verl-agent](https://github.com/langfengQ/verl-agent), and [SDAR](https://github.com/ZJU-REAL/SDAR). We thank their authors and the developers of ALFWorld, Search-R1, WebShop, and SkyRL for making their work available.

## License

TASPO is released under [Apache-2.0](LICENSE). Third-party components retain their respective licenses and copyright notices; see [NOTICE](NOTICE).
