#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=train/common.sh
source "${SCRIPT_DIR}/train/common.sh"

cd "${ROOT_DIR}"
PYTHON_BIN="$(resolve_python_bin)"

MODEL_PATH="runs/sft/qwen3_1_7b_miniv2_sft_merged"
TASKS="gsm8k,mmlu_mini"
OUTPUT_DIR=""
MAX_SAMPLES=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  MODEL_TAG="$(basename "${MODEL_PATH}")"
  OUTPUT_DIR="runs/eval/${MODEL_TAG}"
fi
mkdir -p "${OUTPUT_DIR}"

IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
for task in "${TASK_ARRAY[@]}"; do
  case "${task}" in
    gsm8k)
      "${PYTHON_BIN}" -m src.eval.eval_gsm8k \
        --model_path "${MODEL_PATH}" \
        --input_path data/raw/eval/gsm8k.jsonl \
        --output_path "${OUTPUT_DIR}/gsm8k.json" \
        --max_samples "${MAX_SAMPLES}"
      ;;
    mmlu_mini)
      "${PYTHON_BIN}" -m src.eval.eval_mmlu \
        --model_path "${MODEL_PATH}" \
        --input_path data/raw/eval/mmlu_mini.jsonl \
        --output_path "${OUTPUT_DIR}/mmlu_mini.json" \
        --max_samples "${MAX_SAMPLES}"
      ;;
    *)
      echo "[ERROR] unsupported task: ${task}"
      exit 1
      ;;
  esac
done

"${PYTHON_BIN}" -m src.eval.aggregate \
  --input_dir "${OUTPUT_DIR}" \
  --output runs/eval/summary.json
