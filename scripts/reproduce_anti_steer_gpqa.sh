#!/usr/bin/env bash
set -euo pipefail

# GPQA Diamond (all 198 examples).
# Matched tokenwise result: CoT 18.2%, anti-steer 21.7%, 511 -> 476 mean tokens.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B}
SAE_PATH=${SAE_PATH:-/p/realai/guangzhi/sae-faithful-rag/checkpoints/sae-qwen3-4b/26624/layers.29}
FEATURES_FILE=${FEATURES_FILE:-src/anti_steer/reason_top25.json}
OUTPUT_DIR=${OUTPUT_DIR:-results/anti_steer/gpqa}
GPU_IDS=${GPU_IDS:-0,1,2,3}

SAMPLES=198
MAX_NEW_TOKENS=512
STRENGTH=8.7
STARTS=(0 50 100 149)
ENDS=(50 100 149 198)

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly 4 comma-separated GPU ids" >&2
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

  for shard in {0..3}; do
    local start=${STARTS[$shard]}
    local end=${ENDS[$shard]}
    local batch_size=$((end - start))
    CUDA_VISIBLE_DEVICES=${GPUS[$shard]} python -m src.anti_steer.evaluate \
      --dataset gpqa \
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
      --batch-size "$batch_size" \
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
    "$OUTPUT_DIR/${prefix}_shard"{0..3}.json
}

run_condition baseline tokenwise baseline 1.0
run_condition anti-steer tokenwise anti_steer_a8.7 "$STRENGTH"
