#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/inference/parallel-ab}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-32}"
MAX_TOKENS="${MAX_TOKENS:-128}"
CONTEXT_LENGTHS_JSON="${CONTEXT_LENGTHS_JSON:-[1024,2048]}"
CONCURRENCY_LEVELS_JSON="${CONCURRENCY_LEVELS_JSON:-[1,2,4,8,16]}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"

mkdir -p "${OUT_DIR}"

run_topology() {
  local label="$1" tensor_parallel="$2" pipeline_parallel="$3"
  OUT_DIR="${OUT_DIR}" \
  EXPERIMENT_NAME="${label}-full-32" \
  TENSOR_PARALLEL_SIZE="${tensor_parallel}" \
  PIPELINE_PARALLEL_SIZE="${pipeline_parallel}" \
  REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL}" \
  MAX_TOKENS="${MAX_TOKENS}" \
  CONTEXT_LENGTHS_JSON="${CONTEXT_LENGTHS_JSON}" \
  CONCURRENCY_LEVELS_JSON="${CONCURRENCY_LEVELS_JSON}" \
  ASYNC_SCHEDULING="${ASYNC_SCHEDULING}" \
    "${ROOT_DIR}/scripts/run_qwen36_scheduler_sweep.sh" 8
}

run_topology tp2-pp1 2 1
run_topology tp1-pp2 1 2
