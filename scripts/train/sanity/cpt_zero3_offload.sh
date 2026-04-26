#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

DEEPSPEED_BIN="$(resolve_deepspeed_bin)"
"${DEEPSPEED_BIN}" src/train/train_cpt.py \
  --model_config configs/model/qwen3_1_7b_base.yaml \
  --train_config configs/train/sanity/cpt_sanity_zero3_offload.yaml \
  --deepspeed configs/train/ds/ds_zero3_offload_sanity.json \
  --dataset_config configs/dataset_manifest.json \
  "$@"
