#!/usr/bin/env bash
set -euo pipefail

# BBH subset: logical_deduction_three_objects (all 250 examples).
# Verified target: strict CoT 81.6%, anti-steer 62.8%, max 512 new tokens.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B}
SAE_PATH=${SAE_PATH:-/p/realai/guangzhi/sae-faithful-rag/checkpoints/sae-qwen3-4b/26624/layers.29}
FEATURES_FILE=${FEATURES_FILE:-src/anti_steer/reason_top25.json}
OUTPUT_DIR=${OUTPUT_DIR:-results/anti_steer/bbh}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}

SAMPLES=250
MAX_NEW_TOKENS=512
STRENGTH=9.24
STARTS=(0 32 64 96 128 159 190 220)
ENDS=(32 64 96 128 159 190 220 250)

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -ne 8 ]]; then
  echo "GPU_IDS must contain exactly 8 comma-separated GPU ids" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

run_condition() {
  local condition=$1
  local prefix=$2
  local strength=$3
  local pids=()
  local failed=0

  for shard in {0..7}; do
    local start=${STARTS[$shard]}
    local end=${ENDS[$shard]}
    local batch_size=$((end - start))
    CUDA_VISIBLE_DEVICES=${GPUS[$shard]} python -m src.anti_steer.evaluate \
      --dataset bbh \
      --condition "$condition" \
      --engine tokenwise \
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
    "$OUTPUT_DIR/${prefix}_shard"{0..7}.json
}

run_condition baseline baseline 1.0
run_condition anti-steer anti_steer_a9.24 "$STRENGTH"
