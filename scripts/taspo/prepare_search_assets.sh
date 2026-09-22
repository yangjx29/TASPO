#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROCESSED_DIR="${SEARCH_PROCESSED_DIR:-$HOME/data/searchR1_processed_direct}"
RETRIEVAL_DIR="${SEARCH_RETRIEVAL_DIR:-$HOME/data/searchR1}"

cd "$ROOT"
mkdir -p "$PROCESSED_DIR" "$RETRIEVAL_DIR"

python3 examples/data_preprocess/preprocess_search_r1_dataset.py \
  --local_dir "$PROCESSED_DIR"

if [[ ! -s "$RETRIEVAL_DIR/part_aa" || ! -s "$RETRIEVAL_DIR/part_ab" || ! -s "$RETRIEVAL_DIR/wiki-18.jsonl.gz" ]]; then
  python3 examples/search/searchr1_download.py --local_dir "$RETRIEVAL_DIR"
fi

if [[ ! -s "$RETRIEVAL_DIR/e5_Flat.index" ]]; then
  index_tmp="$RETRIEVAL_DIR/e5_Flat.index.tmp"
  cat "$RETRIEVAL_DIR/part_aa" "$RETRIEVAL_DIR/part_ab" > "$index_tmp"
  mv "$index_tmp" "$RETRIEVAL_DIR/e5_Flat.index"
fi

if [[ ! -s "$RETRIEVAL_DIR/wiki-18.jsonl" ]]; then
  corpus_tmp="$RETRIEVAL_DIR/wiki-18.jsonl.tmp"
  gzip -dc "$RETRIEVAL_DIR/wiki-18.jsonl.gz" > "$corpus_tmp"
  mv "$corpus_tmp" "$RETRIEVAL_DIR/wiki-18.jsonl"
fi

echo "Search assets are ready:"
echo "  processed data: $PROCESSED_DIR"
echo "  retrieval data: $RETRIEVAL_DIR"
