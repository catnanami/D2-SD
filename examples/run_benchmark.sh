#!/usr/bin/env bash
# Run the DFlash (single-draft) baseline across a suite of datasets.
#
# Usage:
#   bash examples/run_benchmark.sh
#
# Override paths and resources via environment variables:
#   TARGET_MODEL=/path/to/qwen3-8b \
#   DRAFT_MODEL=/path/to/qwen3-8b-dflash \
#   GPUS=0,1,2,3 \
#   bash examples/run_benchmark.sh
set -euo pipefail

# ---- Configuration --------------------------------------------------------
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"$GPUS")}"
MASTER_PORT="${MASTER_PORT:-29600}"

TARGET_MODEL="${TARGET_MODEL:-../qwen3-8b}"
DRAFT_MODEL="${DRAFT_MODEL:-../qwen3-8b-dflash}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DEFAULT_MAX_SAMPLES="${MAX_SAMPLES:-128}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

# Datasets to evaluate. To override, set TASKS as a comma-separated list of
# "dataset[:max_samples]" pairs, e.g. TASKS="gsm8k:64,humaneval:32".
DEFAULT_TASKS=(
  "gsm8k"
  "math500"
  "aime24"
  "aime25"
  "humaneval"
  "mbpp"
  "livecodebench"
  "swe-bench"
  "mt-bench"
  "alpaca"
)

if [[ -n "${TASKS:-}" ]]; then
  IFS=',' read -r -a TASK_LIST <<<"$TASKS"
else
  TASK_LIST=("${DEFAULT_TASKS[@]}")
fi

# ---- Run ------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPUS"

for spec in "${TASK_LIST[@]}"; do
  if [[ "$spec" == *:* ]]; then
    DATASET_NAME="${spec%%:*}"
    SAMPLES="${spec##*:}"
  else
    DATASET_NAME="$spec"
    SAMPLES="$DEFAULT_MAX_SAMPLES"
  fi

  echo "========================================================"
  echo "Running DFlash benchmark: ${DATASET_NAME} (${SAMPLES} samples)"
  echo "========================================================"

  torchrun \
    --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" \
    benchmark.py \
    --mode dflash \
    --dataset "$DATASET_NAME" \
    --max-samples "$SAMPLES" \
    --model-name-or-path "$TARGET_MODEL" \
    --draft-name-or-path "$DRAFT_MODEL" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    2>&1 | tee "${LOG_DIR}/${DATASET_NAME}.log"
done
