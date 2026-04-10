#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/prepare_bottle_finetune_data.py \
  --source-root own_datasets \
  --output-json prepared_data/bottle_positive_only/split.json

CUDA_VISIBLE_DEVICES=15 python train_bottle_positive_only.py \
  --metadata-path prepared_data/bottle_positive_only/split.json \
  --checkpoint-path weight/train_on_mvtec/CLIP.pth \
  --output-dir runs/bottle_positive_only \
  --requested-device cuda:0 \
  --batch-size 4
