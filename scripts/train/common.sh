#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

resolve_python_bin() {
  if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    echo "${ROOT_DIR}/.venv/bin/python"
  else
    echo "python"
  fi
}

resolve_torchrun_bin() {
  if [ -x "${ROOT_DIR}/.venv/bin/torchrun" ]; then
    echo "${ROOT_DIR}/.venv/bin/torchrun"
  else
    echo "torchrun"
  fi
}

resolve_deepspeed_bin() {
  if [ -x "${ROOT_DIR}/.venv/bin/deepspeed" ]; then
    echo "${ROOT_DIR}/.venv/bin/deepspeed"
  else
    echo "deepspeed"
  fi
}
