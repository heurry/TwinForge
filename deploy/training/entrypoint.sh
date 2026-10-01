#!/usr/bin/env bash
set -euo pipefail

nnodes="${NNODES:-${WORLD_SIZE:-1}}"
node_rank="${RANK:-${PET_NODE_RANK:-0}}"
master_addr="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
master_port="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
processes="${GPUS_PER_NODE:-1}"

echo "Starting Qwen3.5 LoRA training: nnodes=${nnodes} node_rank=${node_rank} processes=${processes} master=${master_addr}:${master_port}"

exec torchrun \
  --nnodes="${nnodes}" \
  --node_rank="${node_rank}" \
  --nproc_per_node="${processes}" \
  --master_addr="${master_addr}" \
  --master_port="${master_port}" \
  /workspace/train_sft.py
