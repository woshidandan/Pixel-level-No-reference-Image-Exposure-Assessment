#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLIT="${SPLIT:-test}"
IMAGE_ROOT="${IMAGE_ROOT:?Set IMAGE_ROOT to Dataset Part1_IEA40K/IEA_img}"
mkdir -p "$ROOT/runs/eval_epoch10"
cd "$ROOT/code"
export LLAMA_MODEL_NAME="$ROOT/weights/llama8b_base"
python train_reasoning_grounding.py evaluate \
  --checkpoint "$ROOT/weights/best_checkpoint/reasoning_grounding_model_best_epoch_010.pt" \
  --dataset-format dataset_example \
  --dataset-root "$ROOT/data/Data Partitioning/$SPLIT" \
  --image-root "$IMAGE_ROOT" \
  --output-json "$ROOT/runs/eval_epoch10/${SPLIT}_metrics.json" \
  --batch-size 2 \
  --num-workers 2 \
  --image-size 128 \
  --max-text-len 96 \
  --max-question-len 96 \
  --threshold 0.5 \
  --thresholds 0.2,0.3,0.4,0.5,0.6 \
  --device cuda
