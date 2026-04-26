#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=train/common.sh
source "${SCRIPT_DIR}/train/common.sh"

cd "${ROOT_DIR}"
PYTHON_BIN="$(resolve_python_bin)"
CONFIG_PATH="${CONFIG_PATH:-configs/serve/vllm.yaml}"
MODEL_PATH="${MODEL_PATH:-}"
EXTRA_ARGS=()

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

readarray -t CFG_LINES < <(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import yaml
import os

with open(os.environ["CONFIG_PATH"], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
for key in [
    "model",
    "tensor_parallel_size",
    "dtype",
    "max_model_len",
    "gpu_memory_utilization",
    "trust_remote_code",
    "served_model_name",
    "host",
    "port",
    "api_key",
]:
    print(f"{key}={cfg[key]}")
PY
)

for line in "${CFG_LINES[@]}"; do
  key="${line%%=*}"
  value="${line#*=}"
  case "${key}" in
    model) DEFAULT_MODEL="${value}" ;;
    tensor_parallel_size) TP_SIZE="${value}" ;;
    dtype) DTYPE="${value}" ;;
    max_model_len) MAX_MODEL_LEN="${value}" ;;
    gpu_memory_utilization) GPU_MEMORY_UTILIZATION="${value}" ;;
    trust_remote_code) TRUST_REMOTE_CODE="${value}" ;;
    served_model_name) SERVED_MODEL_NAME="${value}" ;;
    host) HOST="${value}" ;;
    port) PORT="${value}" ;;
    api_key) API_KEY="${value}" ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="${DEFAULT_MODEL}"
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  MODEL_PATH="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

TRUST_REMOTE_CODE_FLAG=()
if [[ "${TRUST_REMOTE_CODE}" == "True" || "${TRUST_REMOTE_CODE}" == "true" ]]; then
  TRUST_REMOTE_CODE_FLAG=(--trust-remote-code)
fi

if ! HOST="${HOST}" PORT="${PORT}" "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import os
import socket
import sys

host = os.environ["HOST"]
port = int(os.environ["PORT"])

bind_host = None if host in ("0.0.0.0", "::") else host
infos = socket.getaddrinfo(bind_host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
ok = False
for family, socktype, proto, _, sockaddr in infos:
    sock = socket.socket(family, socktype, proto)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(sockaddr)
        ok = True
        break
    except OSError:
        continue
    finally:
        sock.close()
sys.exit(0 if ok else 1)
PY
then
  NEXT_PORT=$((PORT + 1))
  echo "[ERROR] ${HOST}:${PORT} is already in use." >&2
  echo "[ERROR] A vLLM server may already be running on this port. Benchmark the existing server, or start a new one on another port:" >&2
  echo "  bash scripts/10_serve_vllm.sh ${MODEL_PATH} --port ${NEXT_PORT}" >&2
  exit 98
fi

ACCESS_HOST="${HOST}"
if [[ "${ACCESS_HOST}" == "0.0.0.0" || "${ACCESS_HOST}" == "::" ]]; then
  ACCESS_HOST="127.0.0.1"
fi

echo "[INFO] serving model path: ${MODEL_PATH}"
echo "[INFO] OpenAI model id: ${SERVED_MODEL_NAME}"
echo "[INFO] local base url: http://${ACCESS_HOST}:${PORT}/v1"

if [[ -x "${ROOT_DIR}/.venv/bin/vllm" ]]; then
  exec "${ROOT_DIR}/.venv/bin/vllm" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    "${TRUST_REMOTE_CODE_FLAG[@]}" \
    "${EXTRA_ARGS[@]}"
fi

if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("vllm") else 1)
PY
then
  exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    "${TRUST_REMOTE_CODE_FLAG[@]}" \
    "${EXTRA_ARGS[@]}"
fi

echo "[ERROR] vLLM is not installed in the current environment: ${PYTHON_BIN}" >&2
echo "[ERROR] Install it first, for example: .venv/bin/pip install vllm" >&2
exit 127
