#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../common.sh
source "${SCRIPT_DIR}/../../common.sh"

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

TORCHRUN_BIN="$(resolve_torchrun_bin)"
"${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE:-2}" src/train/train_cpt.py \
  --model_config configs/model/qwen3_1_7b_base.yaml \
  --train_config configs/train/optimize/native/cpt_opt_native_ddp_no_gc.yaml \
  --dataset_config configs/dataset_manifest.json \
  "$@"
