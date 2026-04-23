#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

vllm serve runs/sft/qwen25_3b_miniv2_sft_merged   --served-model-name qwen25-3b-miniv2   --tensor-parallel-size 2   --dtype float16   --max-model-len 4096   --gpu-memory-utilization 0.90   --host 0.0.0.0   --port 8000
