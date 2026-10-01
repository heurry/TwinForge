#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NCCL_TESTS_COMMIT="${NCCL_TESTS_COMMIT:-717b68318278e93f371d8ffb46b076069d7c7851}"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-${ROOT_DIR}/.cache/nccl-tests/${NCCL_TESTS_COMMIT}}"
VLLM_DOCKER_IMAGE="${VLLM_DOCKER_IMAGE:-local/vllm-openai:qwen35-v0.19.1}"
SERVICE_CONTAINER="${SERVICE_CONTAINER:-twinforge-vllm-qwen36-awq}"
SERVICE_CONFIG="${SERVICE_CONFIG:-configs/serve/qwen36_27b_awq_vllm_tp2_optimized.yaml}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/inference/nccl-pcie-$(date +%Y%m%d-%H%M%S)}"
MIN_BYTES="${MIN_BYTES:-8K}"
MAX_BYTES="${MAX_BYTES:-64M}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
ITERS="${ITERS:-50}"

mkdir -p "${NCCL_TESTS_DIR%/*}" "${OUT_DIR}"
if [[ ! -d "${NCCL_TESTS_DIR}/.git" ]]; then
  git clone https://github.com/NVIDIA/nccl-tests.git "${NCCL_TESTS_DIR}"
  git -C "${NCCL_TESTS_DIR}" checkout --detach "${NCCL_TESTS_COMMIT}"
fi
if [[ ! -x "${NCCL_TESTS_DIR}/build/all_reduce_perf" ]]; then
  docker run --rm \
    -v "${NCCL_TESTS_DIR}:/workspace" -w /workspace \
    --entrypoint make "${VLLM_DOCKER_IMAGE}" \
    -j8 MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr
fi

service_was_running=false
if [[ "$(docker inspect -f '{{.State.Running}}' "${SERVICE_CONTAINER}" 2>/dev/null || true)" == "true" ]]; then
  service_was_running=true
  docker stop -t 30 "${SERVICE_CONTAINER}" >/dev/null
fi

restore_service() {
  if [[ "${service_was_running}" == "true" ]]; then
    CONFIG_PATH="${SERVICE_CONFIG}" \
    VLLM_DOCKER_IMAGE="${VLLM_DOCKER_IMAGE}" \
    VLLM_CONTAINER_NAME="${SERVICE_CONTAINER}" \
    VLLM_DOCKER_DETACH=true \
      "${ROOT_DIR}/scripts/serve_vllm_replica.sh" >/dev/null
  fi
}
trap restore_service EXIT

run_profile() {
  local profile="$1"
  shift
  printf '[NCCL] profile=%s\n' "${profile}"
  docker run --rm --gpus all --ipc=host --network=host \
    -v "${NCCL_TESTS_DIR}:/workspace" -w /workspace \
    --entrypoint /workspace/build/all_reduce_perf \
    "$@" "${VLLM_DOCKER_IMAGE}" \
    -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f 2 -g 2 \
    -w "${WARMUP_ITERS}" -n "${ITERS}" \
    >"${OUT_DIR}/allreduce-${profile}.log"
}

run_profile auto
run_profile ring-1ch \
  -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=0 \
  -e NCCL_ALGO=Ring -e NCCL_MIN_NCHANNELS=1 -e NCCL_MAX_NCHANNELS=1
run_profile ring-2ch \
  -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=0 \
  -e NCCL_ALGO=Ring -e NCCL_MIN_NCHANNELS=2 -e NCCL_MAX_NCHANNELS=2
run_profile ring-4ch \
  -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=0 \
  -e NCCL_ALGO=Ring -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4
run_profile tree-2ch \
  -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=0 \
  -e NCCL_ALGO=Tree -e NCCL_MIN_NCHANNELS=2 -e NCCL_MAX_NCHANNELS=2
run_profile ring-2ch-threads512 \
  -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=0 \
  -e NCCL_ALGO=Ring -e NCCL_MIN_NCHANNELS=2 -e NCCL_MAX_NCHANNELS=2 \
  -e NCCL_NTHREADS=512

summary="${OUT_DIR}/summary.csv"
printf 'profile,size_bytes,time_us,busbw_gbps,wrong\n' >"${summary}"
for file in "${OUT_DIR}"/allreduce-*.log; do
  profile="$(basename "${file}" .log)"
  profile="${profile#allreduce-}"
  awk -v profile="${profile}" \
    '$1 ~ /^[0-9]+$/ {printf "%s,%s,%s,%s,%s\n", profile, $1, $6, $8, $9}' \
    "${file}" >>"${summary}"
done

printf '[NCCL] results=%s\n' "${summary}"
