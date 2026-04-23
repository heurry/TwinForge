#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=2 src/train/train_cpt.py   --model_config configs/model/qwen25_3b_base.yaml   --train_config configs/train/cpt.yaml   --deepspeed configs/train/deepspeed_zero2.json   --dataset_config configs/dataset_manifest.json
