#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_texts(paths: List[str]) -> Dataset:
    import jsonlines

    records = []
    for p in paths:
        with jsonlines.open(p, "r") as reader:
            for obj in reader:
                text = obj.get("text", "")
                if isinstance(text, str) and text.strip():
                    records.append({"text": text.strip()})
    return Dataset.from_list(records)


def pack_examples(examples, tokenizer, max_length: int):
    tokenized = tokenizer(
        examples["text"],
        add_special_tokens=False,
        truncation=False,
    )
    input_ids = []
    for ids in tokenized["input_ids"]:
        input_ids.extend(ids + [tokenizer.eos_token_id])

    total_length = (len(input_ids) // max_length) * max_length
    input_ids = input_ids[:total_length]

    chunks = [input_ids[i:i + max_length] for i in range(0, total_length, max_length)]
    return {"input_ids": chunks, "labels": chunks.copy()}


def build_dataset(tokenizer, raw_text_paths: List[str], max_length: int, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    raw_ds = load_jsonl_texts(raw_text_paths)

    packed_ds = raw_ds.map(
        lambda batch: pack_examples(batch, tokenizer, max_length),
        batched=True,
        batch_size=1000,
        remove_columns=raw_ds.column_names,
        desc="Packing CPT dataset",
    )

    packed_ds.save_to_disk(cache_dir)
    return packed_ds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def resolve_cpt_raw_paths(dataset_manifest: Dict[str, Any], raw_root: str = "data/raw/cpt") -> List[str]:
    paths = []
    for entry in dataset_manifest["datasets"].get("cpt", []):
        name = entry["name"]
        p = Path(raw_root) / f"{name}.jsonl"
        if p.exists():
            paths.append(str(p))
        else:
            print(f"[WARN] missing CPT raw file: {p}")
    if not paths:
        raise FileNotFoundError("No CPT raw jsonl files found under data/raw/cpt/")
    return paths


def main():
    args = parse_args()

    model_cfg = read_yaml(args.model_config)
    train_cfg = read_yaml(args.train_config)
    dataset_cfg = read_json(args.dataset_config)

    set_seed(train_cfg.get("seed", 42))

    model_name = model_cfg["model_name_or_path"]
    tokenizer_name = model_cfg["tokenizer_name_or_path"]
    max_seq_length = train_cfg.get("max_seq_length", 2048)
    output_dir = train_cfg["output_dir"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(train_cfg["logging_dir"]).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_paths = resolve_cpt_raw_paths(dataset_cfg, raw_root="data/raw/cpt")
    tokenized_cache_dir = "data/tokenized/cpt/train"

    if Path(tokenized_cache_dir).exists():
        print(f"[INFO] loading packed dataset from {tokenized_cache_dir}")
        train_dataset = load_from_disk(tokenized_cache_dir)
    else:
        train_dataset = build_dataset(
            tokenizer=tokenizer,
            raw_text_paths=raw_paths,
            max_length=max_seq_length,
            cache_dir=tokenized_cache_dir,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if model_cfg.get("torch_dtype", "float16") == "float16" else "auto",
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    model.config.use_cache = False
    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_dir=train_cfg["logging_dir"],
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=3,
        bf16=train_cfg.get("bf16", False),
        fp16=train_cfg.get("fp16", True),
        tf32=train_cfg.get("tf32", True),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        remove_unused_columns=train_cfg.get("remove_unused_columns", False),
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.state.log_history
    with open(Path(output_dir) / "train_log_history.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[DONE] CPT training finished. model saved to {output_dir}")


if __name__ == "__main__":
    main()
