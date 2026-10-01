#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-configs/serve/qwen36_27b_awq_vllm.yaml}"
export VLLM_DOCKER_IMAGE="${VLLM_DOCKER_IMAGE:-local/vllm-openai:qwen35-v0.19.1}"
export VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-twinforge-vllm-qwen36-awq}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${SCRIPT_DIR}/../.cache/vllm/qwen36-27b-awq}"
export VLLM_GPUS="${VLLM_GPUS:-all}"

exec "${SCRIPT_DIR}/serve_vllm_replica.sh"
