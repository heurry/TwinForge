#!/usr/bin/env bash
set -e

python -m src.eval.eval_gsm8k --model_path runs/sft/qwen25_3b_miniv2_sft_merged
python -m src.eval.eval_mmlu --model_path runs/sft/qwen25_3b_miniv2_sft_merged
python -m src.eval.eval_humaneval --model_path runs/sft/qwen25_3b_miniv2_sft_merged
python -m src.eval.eval_mbpp --model_path runs/sft/qwen25_3b_miniv2_sft_merged

python -m src.eval.aggregate --input_dir runs/eval --output runs/eval/summary.json
