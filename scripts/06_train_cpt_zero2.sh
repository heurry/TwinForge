#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

if [ -x .venv/bin/deepspeed ]; then
  DEEPSPEED_BIN=.venv/bin/deepspeed
else
  DEEPSPEED_BIN=deepspeed
fi

"${DEEPSPEED_BIN}" --num_gpus="${NUM_GPUS:-2}" src/train/train_cpt.py \
  --model_config configs/model/qwen3_1_7b_base.yaml \
  --train_config configs/train/cpt_zero2.yaml \
  --deepspeed configs/train/deepspeed_zero2.json \
  --dataset_config configs/dataset_manifest.json \
  "$@"
