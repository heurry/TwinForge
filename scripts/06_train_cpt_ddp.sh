#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

if [ -x .venv/bin/torchrun ]; then
  TORCHRUN_BIN=.venv/bin/torchrun
else
  TORCHRUN_BIN=torchrun
fi

"${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE:-2}" src/train/train_cpt.py \
  --model_config configs/model/qwen3_1_7b_base.yaml \
  --train_config configs/train/cpt_ddp.yaml \
  --dataset_config configs/dataset_manifest.json \
  "$@"
