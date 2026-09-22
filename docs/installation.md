# Installation

[Back to README](../README.md) · [Training guide](usage.md)

Run these commands from the repository root on a Linux machine with NVIDIA GPUs and Conda. The training launchers default to four GPUs. ALFWorld and Search-QA use Python 3.12; WebShop uses a separate Python 3.10 environment.

## ALFWorld

```bash
conda create -n taspo python=3.12 -y
conda activate taspo
bash scripts/taspo/bootstrap_env.sh alfworld
```

The installer downloads ALFWorld data. To download it again or set its location:

```bash
alfworld-download -f
export ALFWORLD_DATA="$HOME/.cache/alfworld"
```

The training launcher prepares its input parquet files before starting the trainer.

## Search-QA

Install the training environment and prepare the datasets, retrieval corpus, and index:

```bash
conda create -n taspo python=3.12 -y
conda activate taspo
bash scripts/taspo/bootstrap_env.sh search
bash scripts/taspo/prepare_search_assets.sh
```

Create a separate environment for the retriever:

```bash
conda create -n taspo-retriever python=3.10 -y
conda activate taspo-retriever
conda install numpy==1.26.4 -y
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets pyserini huggingface_hub uvicorn fastapi
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
```

Start retrieval in another terminal, selecting a GPU available for the retriever:

```bash
conda activate taspo-retriever
CUDA_VISIBLE_DEVICES=4 bash examples/search/retriever/retrieval_launch.sh
```

The training process connects to `http://127.0.0.1:8000/retrieve` by default. To use another address:

```bash
export SEARCH_URL="http://your-host:8000/retrieve"
```

Return to the `taspo` environment before launching training. `TRAIN_DATA` and `VAL_DATA` can override the default parquet paths under `$HOME/data/searchR1_processed_direct/`.

## WebShop

```bash
conda create -n taspo-webshop python=3.10 -y
conda activate taspo-webshop
bash scripts/taspo/bootstrap_env.sh webshop
```

The installer uses WebShop's setup script to prepare its dependencies and data. Run WebShop training from this environment.

## Analyzer

Set these variables in the shell that launches training so that Ray workers inherit them:

```bash
export TASPO_ANALYZER_BASE_URL="https://your-endpoint/v1/chat/completions"
export TASPO_ANALYZER_MODEL="your-analyzer-model"
export TASPO_ANALYZER_API_KEY="your-api-key"
export TASPO_ANALYZER_REQUIRED=true
```

The endpoint must support OpenAI-compatible Chat Completions and JSON responses. A local endpoint without authentication does not need `TASPO_ANALYZER_API_KEY`.

Check the setup before training:

```bash
python -m scripts.taspo.preflight --benchmark alfworld --check-api
```

Replace `alfworld` with `search` or `webshop` for the other environments. Continue with the [training guide](usage.md).
