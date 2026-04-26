#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../common.sh
source "${SCRIPT_DIR}/../../common.sh"

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

DEEPSPEED_BIN="$(resolve_deepspeed_bin)"
OUTPUT_DIR="runs/cpt/resume_validation/deepspeed/qwen3_1_7b/zero2_recommended"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-10"
SNAPSHOT_DIR="${OUTPUT_DIR}/resume_validation"

if [ -d "${OUTPUT_DIR}" ] && [ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
  echo "[ERROR] output dir ${OUTPUT_DIR} is not empty. remove it manually before running resume validation."
  exit 1
fi

echo "[RESUME] zero2 stage1"
"${DEEPSPEED_BIN}" src/train/train_cpt.py \
  --model_config configs/model/qwen3_1_7b_base.yaml \
  --train_config configs/train/resume/deepspeed/cpt_resume_zero2_stage1.yaml \
  --deepspeed configs/train/ds/ds_zero2_bench.json \
  --dataset_config configs/dataset_manifest.json \
  "$@"

mkdir -p "${SNAPSHOT_DIR}"
cp "${OUTPUT_DIR}/train_summary.json" "${SNAPSHOT_DIR}/stage1_train_summary.json"
cp "${OUTPUT_DIR}/train_log_history.json" "${SNAPSHOT_DIR}/stage1_train_log_history.json"

if [ ! -d "${CHECKPOINT_DIR}" ]; then
  echo "[ERROR] expected checkpoint not found at ${CHECKPOINT_DIR}"
  exit 1
fi

echo "[RESUME] zero2 stage2"
"${DEEPSPEED_BIN}" src/train/train_cpt.py \
  --model_config configs/model/qwen3_1_7b_base.yaml \
  --train_config configs/train/resume/deepspeed/cpt_resume_zero2_stage2.yaml \
  --deepspeed configs/train/ds/ds_zero2_bench.json \
  --dataset_config configs/dataset_manifest.json \
  --resume_from_checkpoint "${CHECKPOINT_DIR}" \
  "$@"

cp "${OUTPUT_DIR}/train_summary.json" "${SNAPSHOT_DIR}/stage2_train_summary.json"
cp "${OUTPUT_DIR}/train_log_history.json" "${SNAPSHOT_DIR}/stage2_train_log_history.json"

PYTHON_BIN="$(resolve_python_bin)"
"${PYTHON_BIN}" scripts/16_report_cpt_resume_validation.py --output reports/cpt_resume_validation.md
