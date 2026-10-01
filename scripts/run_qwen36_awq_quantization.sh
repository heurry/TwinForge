#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv-awq/bin/python"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/quantization/qwen36_27b_awq_w4a16.yaml}"
MODE="${MODE:-smoke}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] missing AWQ environment: ${PYTHON_BIN}" >&2
  exit 1
fi

case "${MODE}" in
  smoke)
    SAMPLES="${SAMPLES:-8}"
    MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
    CALIBRATION_PATH="${ROOT_DIR}/data/quantization/dianjin_csc_awq_smoke.jsonl"
    OUTPUT_PATH="${OUTPUT_PATH:-/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-AWQ-INT4-smoke}"
    ;;
  full)
    # 512 samples exhaust activation-cache headroom on dual 24 GiB RTX 3090s.
    SAMPLES="${SAMPLES:-128}"
    MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
    CALIBRATION_PATH="${ROOT_DIR}/data/quantization/dianjin_csc_awq_calibration.jsonl"
    OUTPUT_PATH="${OUTPUT_PATH:-/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-AWQ-INT4}"
    ;;
  *)
    echo "[ERROR] MODE must be smoke or full" >&2
    exit 1
    ;;
esac

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_awq_calibration.py" \
  --source "${ROOT_DIR}/data/cleaned/dianjin_csc_sft_train.jsonl" \
  --output "${CALIBRATION_PATH}" \
  --model "/mnt/nvme-data/models/LLM_model/Qwen3.6-27B" \
  --samples "${SAMPLES}" \
  --seed 42

if docker ps --format '{{.Names}}' | grep -qx 'twinforge-vllm-qwen36'; then
  echo "[INFO] stopping twinforge-vllm-qwen36 to release both RTX 3090 GPUs"
  docker stop twinforge-vllm-qwen36
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/quantize_qwen36_awq.py" \
  --config "${CONFIG_PATH}" \
  --samples "${SAMPLES}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --calibration "${CALIBRATION_PATH}" \
  --output "${OUTPUT_PATH}"
