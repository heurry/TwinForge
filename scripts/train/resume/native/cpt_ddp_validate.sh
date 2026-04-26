#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../common.sh
source "${SCRIPT_DIR}/../../common.sh"

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

TORCHRUN_BIN="$(resolve_torchrun_bin)"
OUTPUT_DIR="runs/cpt/resume_validation/native/qwen3_1_7b/ddp_recommended"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-10"
SNAPSHOT_DIR="${OUTPUT_DIR}/resume_validation"

if [ -d "${OUTPUT_DIR}" ] && [ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
  echo "[ERROR] output dir ${OUTPUT_DIR} is not empty. remove it manually before running resume validation."
  exit 1
fi

echo "[RESUME] native_ddp stage1"
"${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE:-2}" src/train/train_cpt.py \
  --model_config configs/model/qwen3_1_7b_base.yaml \
  --train_config configs/train/resume/native/cpt_resume_native_ddp_stage1.yaml \
  --dataset_config configs/dataset_manifest.json \
  "$@"

mkdir -p "${SNAPSHOT_DIR}"
cp "${OUTPUT_DIR}/train_summary.json" "${SNAPSHOT_DIR}/stage1_train_summary.json"
cp "${OUTPUT_DIR}/train_log_history.json" "${SNAPSHOT_DIR}/stage1_train_log_history.json"

if [ ! -d "${CHECKPOINT_DIR}" ]; then
  echo "[ERROR] expected checkpoint not found at ${CHECKPOINT_DIR}"
  exit 1
fi

echo "[RESUME] native_ddp stage2"
"${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE:-2}" src/train/train_cpt.py \
  --model_config configs/model/qwen3_1_7b_base.yaml \
  --train_config configs/train/resume/native/cpt_resume_native_ddp_stage2.yaml \
  --dataset_config configs/dataset_manifest.json \
  --resume_from_checkpoint "${CHECKPOINT_DIR}" \
  "$@"

cp "${OUTPUT_DIR}/train_summary.json" "${SNAPSHOT_DIR}/stage2_train_summary.json"
cp "${OUTPUT_DIR}/train_log_history.json" "${SNAPSHOT_DIR}/stage2_train_log_history.json"

PYTHON_BIN="$(resolve_python_bin)"
"${PYTHON_BIN}" scripts/16_report_cpt_resume_validation.py --output reports/cpt_resume_validation.md
