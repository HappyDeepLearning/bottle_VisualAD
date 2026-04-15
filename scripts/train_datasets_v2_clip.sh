#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYTHON="python"
if [ -x "${PROJECT_ROOT}/.venv/bin/python" ] && "${PROJECT_ROOT}/.venv/bin/python" -c "import torch" >/dev/null 2>&1; then
    DEFAULT_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
GPU_ID="${GPU_ID:-15}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_PATH="${DATASET_PATH:-/Users/majingzhe/Desktop/瓶盖缺陷检测论文整理/数据集/datasets_v2_visualad}"
TRAIN_DATASET="${TRAIN_DATASET:-datasets_v2}"
BACKBONE="${BACKBONE:-ViT-L/14@336px}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-15}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SEED="${SEED:-111}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/experiments/datasets_v2_clip_gpu15}"
CLASSIFICATION_ONLY="${CLASSIFICATION_ONLY:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
STRICT_DETERMINISM="${STRICT_DETERMINISM:-0}"

echo "================================================================"
echo "VisualAD Training - datasets_v2"
echo "================================================================"
echo "Python: ${PYTHON_BIN}"
echo "Visible GPU ID: ${GPU_ID}"
echo "Torch Device: ${DEVICE}"
echo "Dataset Path: ${DATASET_PATH}"
echo "Train Dataset Name: ${TRAIN_DATASET}"
echo "Backbone: ${BACKBONE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Image Size: ${IMAGE_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Seed: ${SEED}"
echo "Classification Only: ${CLASSIFICATION_ONLY}"
echo "Accumulation Steps: ${ACCUMULATION_STEPS}"
echo "Strict Determinism: ${STRICT_DETERMINISM}"
echo "Save Root: ${SAVE_ROOT}"
echo "================================================================"
echo ""

if [ ! -d "${DATASET_PATH}" ]; then
    echo "Dataset path not found: ${DATASET_PATH}"
    exit 1
fi

if [ ! -f "${DATASET_PATH}/meta.json" ]; then
    echo "meta.json not found under dataset path: ${DATASET_PATH}"
    exit 1
fi

mkdir -p "${SAVE_ROOT}"

cd "${PROJECT_ROOT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start training..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py \
    --train_data_path "${DATASET_PATH}" \
    --save_path "${SAVE_ROOT}" \
    --train_dataset "${TRAIN_DATASET}" \
    --backbone "${BACKBONE}" \
    --epoch "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --image_size "${IMAGE_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --accumulation_steps "${ACCUMULATION_STEPS}" \
    $([ "${CLASSIFICATION_ONLY}" = "1" ] && printf '%s ' "--classification_only") \
    $([ "${STRICT_DETERMINISM}" = "1" ] && printf '%s' "--strict_determinism")

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed."
echo "Checkpoints saved in: ${SAVE_ROOT}"
