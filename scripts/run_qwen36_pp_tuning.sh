#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/inference/pp-tuning}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-32}"
MAX_TOKENS="${MAX_TOKENS:-128}"

mkdir -p "${OUT_DIR}"

run_candidate() {
  local name="$1" partition="$2" dbo="$3" decode_threshold="$4"
  OUT_DIR="${OUT_DIR}" \
  EXPERIMENT_NAME="${name}" \
  TENSOR_PARALLEL_SIZE=1 \
  PIPELINE_PARALLEL_SIZE=2 \
  PIPELINE_LAYER_PARTITION="${partition}" \
  ENABLE_DBO="${dbo}" \
  DBO_DECODE_TOKEN_THRESHOLD="${decode_threshold}" \
  DBO_PREFILL_TOKEN_THRESHOLD=512 \
  ASYNC_SCHEDULING=false \
  CONTEXT_LENGTHS_JSON='[1024,2048]' \
  CONCURRENCY_LEVELS_JSON='[8,16]' \
  REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL}" \
  MAX_TOKENS="${MAX_TOKENS}" \
    "${ROOT_DIR}/scripts/run_qwen36_scheduler_sweep.sh" 8
}

case "${1:-partitions}" in
  partitions)
    run_candidate pp-l32-32-dbo-off 32,32 false 32
    run_candidate pp-l34-30-dbo-off 34,30 false 32
    run_candidate pp-l35-29-dbo-off 35,29 false 32
    run_candidate pp-l36-28-dbo-off 36,28 false 32
    ;;
  dbo)
    partition="${BEST_PARTITION:-34,30}"
    run_candidate pp-l${partition/,/-}-dbo-d4 "${partition}" true 4
    run_candidate pp-l${partition/,/-}-dbo-d8 "${partition}" true 8
    ;;
  *)
    echo "usage: $0 [partitions|dbo]" >&2
    exit 2
    ;;
esac
