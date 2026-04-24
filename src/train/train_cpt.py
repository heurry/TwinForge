#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
from inspect import signature
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from src.train.callbacks import CudaMemoryLoggingCallback
except ModuleNotFoundError as exc:
    if exc.name != "src":
        raise
    from callbacks import CudaMemoryLoggingCallback


class MemoryMetricTrainer(Trainer):
    def log(self, logs: Dict[str, float], start_time: float | None = None) -> None:
        merged_logs = dict(logs)
        for callback in self.callback_handler.callbacks:
            consume_pending_logs = getattr(callback, "consume_pending_logs", None)
            if callable(consume_pending_logs):
                merged_logs.update(consume_pending_logs(self.state.global_step))
        super().log(merged_logs, start_time=start_time)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dtype(model_cfg: Dict[str, Any]) -> Any:
    dtype_name = model_cfg.get("torch_dtype", "auto")
    if dtype_name not in (None, "auto") and not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype in model config: {dtype_name}")
    return "auto"


def resolve_warmup_steps(train_cfg: Dict[str, Any]) -> int:
    if "warmup_steps" in train_cfg:
        return int(train_cfg["warmup_steps"])
    max_steps = int(train_cfg.get("max_steps", -1))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    if max_steps <= 0 or warmup_ratio <= 0:
        return 0
    return max(1, int(max_steps * warmup_ratio))


def resolve_precision_settings(train_cfg: Dict[str, Any]) -> tuple[bool, bool, bool]:
    bf16 = bool(train_cfg.get("bf16", False))
    fp16 = bool(train_cfg.get("fp16", True))
    tf32 = bool(train_cfg.get("tf32", True))

    if not torch.cuda.is_available():
        return False, False, False

    if bf16 and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        bf16 = False

    if tf32:
        try:
            major, _ = torch.cuda.get_device_capability(0)
            tf32 = major >= 8
        except Exception:
            tf32 = False

    return bf16, fp16, tf32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--local-rank", dest="local_rank", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--tokenized_dataset_path", type=str, default="data/tokenized/cpt/train")
    return parser.parse_args()


def main():
    args = parse_args()

    model_cfg = read_yaml(args.model_config)
    train_cfg = read_yaml(args.train_config)
    _ = read_json(args.dataset_config)

    set_seed(train_cfg.get("seed", 42))

    model_name = model_cfg["model_name_or_path"]
    tokenizer_name = model_cfg["tokenizer_name_or_path"]
    output_dir = train_cfg["output_dir"]
    output_path = Path(output_dir)
    logging_path = Path(train_cfg["logging_dir"])

    if train_cfg.get("overwrite_output_dir", False) and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logging_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(logging_path))

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset_path = Path(args.tokenized_dataset_path)
    if not tokenized_dataset_path.exists():
        raise FileNotFoundError(
            f"Tokenized CPT dataset not found at {tokenized_dataset_path}. "
            "Run `python scripts/02_build_cpt_corpus.py` and "
            "`python scripts/04_tokenize_cpt.py` first."
        )
    print(f"[INFO] loading tokenized CPT dataset from {tokenized_dataset_path}")
    train_dataset = load_from_disk(str(tokenized_dataset_path))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=resolve_dtype(model_cfg),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    enable_gradient_checkpointing = train_cfg.get(
        "gradient_checkpointing",
        model_cfg.get("gradient_checkpointing", True),
    )
    model.config.use_cache = False
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    use_bf16, use_fp16, use_tf32 = resolve_precision_settings(train_cfg)

    training_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        warmup_steps=resolve_warmup_steps(train_cfg),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_tf32,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        remove_unused_columns=train_cfg.get("remove_unused_columns", False),
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", False),
    )
    training_args_signature = signature(TrainingArguments.__init__).parameters
    if "overwrite_output_dir" in training_args_signature:
        training_kwargs["overwrite_output_dir"] = train_cfg.get("overwrite_output_dir", False)

    training_args = TrainingArguments(**training_kwargs)
    cuda_memory_logging_steps = int(
        train_cfg.get("cuda_memory_logging_steps", train_cfg.get("logging_steps", 10))
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    if cuda_memory_logging_steps > 0:
        trainer_kwargs["callbacks"] = [CudaMemoryLoggingCallback(cuda_memory_logging_steps)]
    trainer_signature = signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = MemoryMetricTrainer(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        metrics = trainer.state.log_history
        with open(Path(output_dir) / "train_log_history.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"[DONE] CPT training finished. model saved to {output_dir}")

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
