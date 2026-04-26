#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model.")
    parser.add_argument("--base_model", type=str, default="model/Qwen3-1.7B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=False,
    )
    adapter_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = adapter_model.merge_and_unload()
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[DONE] merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
