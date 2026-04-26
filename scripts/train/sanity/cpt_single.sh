#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="$(resolve_python_bin)"
"${PYTHON_BIN}" src/train/train_cpt.py \
  --model_config configs/model/qwen3_1_7b_base.yaml \
  --train_config configs/train/sanity/cpt_sanity_single.yaml \
  --dataset_config configs/dataset_manifest.json \
  "$@"
