#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-configs/serve/qwen38_27b_fp8_vllm.yaml}"
export VLLM_DOCKER_IMAGE="${VLLM_DOCKER_IMAGE:-local/vllm-openai:qwen35-v0.19.1}"
export VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-twinforge-vllm-qwen38}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${SCRIPT_DIR}/../.cache/vllm/qwen38-27b-fp8}"

exec "${SCRIPT_DIR}/serve_vllm_replica.sh"
