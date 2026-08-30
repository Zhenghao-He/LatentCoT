#!/usr/bin/env bash
set -euo pipefail

# Matched tokenwise target: baseline 62.0%, anti-steer 12.1%, max 512 new tokens.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B}
SAE_PATH=${SAE_PATH:-/p/realai/guangzhi/sae-faithful-rag/checkpoints/sae-qwen3-4b/26624/layers.29}
FEATURES_FILE=${FEATURES_FILE:-src/anti_steer/reason_top25.json}
OUTPUT_DIR=${OUTPUT_DIR:-results/anti_steer/gsm8k}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}

SAMPLES=1000
MAX_NEW_TOKENS=512
STRENGTH=8.9
BATCH_SIZE=25
SHARD_SIZE=125

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -ne 8 ]]; then
  echo "GPU_IDS must contain exactly 8 comma-separated GPU ids" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

run_condition() {
  local condition=$1
  local engine=$2
  local prefix=$3
  local strength=$4
  local pids=()
  local failed=0

  for shard in {0..7}; do
    local start=$((shard * SHARD_SIZE))
    local end=$((start + SHARD_SIZE))
    CUDA_VISIBLE_DEVICES=${GPUS[$shard]} python -m src.anti_steer.evaluate \
      --dataset gsm8k \
      --condition "$condition" \
      --engine "$engine" \
      --model "$MODEL_NAME" \
      --sae-path "$SAE_PATH" \
      --features "$FEATURES_FILE" \
      --layer layers.29 \
      --device cuda:0 \
      --dtype float16 \
      --samples "$SAMPLES" \
      --start-index "$start" \
      --end-index "$end" \
      --batch-size "$BATCH_SIZE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --strength "$strength" \
      --seed 42 \
      --output "$OUTPUT_DIR/${prefix}_shard${shard}.json" \
      --overwrite &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ $failed -ne 0 ]]; then
    echo "$condition evaluation failed" >&2
    exit 1
  fi

  python -m src.anti_steer.merge \
    --output "$OUTPUT_DIR/${prefix}.json" \
    "$OUTPUT_DIR/${prefix}_shard"{0..7}.json
}

run_condition baseline tokenwise baseline 1.0
run_condition anti-steer tokenwise anti_steer_a8.9 "$STRENGTH"
