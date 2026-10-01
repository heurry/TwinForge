#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VLLM_DOCKER_IMAGE:-vllm/vllm-openai:v0.26.0}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/inference/dense-tp-overlap-$(date +%Y%m%d-%H%M%S)}"
TOKENS="${MICROBATCH_TOKENS:-1,2,4,8,16,64,256,512}"

mkdir -p "${OUT_DIR}"

run_profile() {
  local profile="$1"
  shift
  docker run --rm --gpus all --ipc=host --network=host \
    -v "${ROOT_DIR}:/workspace" -w /workspace \
    --entrypoint torchrun \
    "$@" "${IMAGE}" \
    --standalone --nproc-per-node=2 \
    scripts/benchmark_dense_tp_overlap.py \
    --microbatch-tokens "${TOKENS}" \
    >"${OUT_DIR}/${profile}.jsonl"
}

run_profile auto
run_profile shm-ring4 \
  -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=0 \
  -e NCCL_ALGO=Ring -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4

printf 'profile,microbatch_tokens,total_tokens,message_bytes,monolithic_ms,split_serial_ms,overlap_ms,end_to_end_gain_percent,split_serial_gain_percent,hidden_ms,max_abs_error,mean_abs_error,relative_max_error,correct\n' \
  >"${OUT_DIR}/summary.csv"
for file in "${OUT_DIR}"/*.jsonl; do
  profile="$(basename "${file}" .jsonl)"
  while IFS= read -r line; do
    [[ "${line}" == \{* ]] || continue
    jq -r --arg profile "${profile}" \
      '[$profile,.microbatch_tokens,.total_tokens,.message_bytes_per_collective,.monolithic.median_ms,.sequential.median_ms,.overlap.median_ms,.end_to_end_gain_percent,.split_serial_gain_percent,.hidden_time_ms,.max_abs_error,.mean_abs_error,.relative_max_error,.correct] | @csv' \
      <<<"${line}" >>"${OUT_DIR}/summary.csv"
  done <"${file}"
done

printf '[overlap] results=%s\n' "${OUT_DIR}/summary.csv"
