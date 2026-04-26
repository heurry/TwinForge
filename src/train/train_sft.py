#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from inspect import signature
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

try:
    from src.train.train_cpt import (
        MemoryMetricTrainer,
        build_train_summary,
        export_deepspeed_comms_summary,
        read_json,
        read_yaml,
        resolve_model_dtype,
        resolve_precision_settings,
        resolve_torch_profiler_config,
        resolve_warmup_steps,
    )
    from src.train.callbacks import CudaMemoryLoggingCallback, TorchProfilerCallback
except ModuleNotFoundError as exc:
    if exc.name != "src":
        raise
    from train_cpt import (
        MemoryMetricTrainer,
        build_train_summary,
        export_deepspeed_comms_summary,
        read_json,
        read_yaml,
        resolve_model_dtype,
        resolve_precision_settings,
        resolve_torch_profiler_config,
        resolve_warmup_steps,
    )
    from callbacks import CudaMemoryLoggingCallback, TorchProfilerCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--local-rank", dest="local_rank", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--tokenized_dataset_path", type=str, default="data/tokenized/sft/train")
    return parser.parse_args()


def build_lora_config(model_cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> LoraConfig:
    lora_cfg = dict(model_cfg.get("lora", {}) or {})
    lora_cfg.update(train_cfg.get("lora", {}) or {})
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("lora_alpha", lora_cfg.get("alpha", 128))),
        lora_dropout=float(lora_cfg.get("lora_dropout", lora_cfg.get("dropout", 0.05))),
        target_modules=list(lora_cfg.get("target_modules", [])),
        bias="none",
    )


def main() -> None:
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

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token

    tokenized_dataset_path = Path(train_cfg.get("tokenized_dataset_path", args.tokenized_dataset_path))
    if not tokenized_dataset_path.exists():
        raise FileNotFoundError(
            f"Tokenized SFT dataset not found at {tokenized_dataset_path}. "
            "Run `python scripts/03_build_sft_dataset.py` and `python scripts/05_tokenize_sft.py` first."
        )
    print(f"[INFO] loading tokenized SFT dataset from {tokenized_dataset_path}")
    train_dataset = load_from_disk(str(tokenized_dataset_path))

    eval_dataset = None
    eval_dataset_path = train_cfg.get("eval_tokenized_dataset_path")
    if eval_dataset_path:
        eval_path = Path(eval_dataset_path)
        if eval_path.exists():
            eval_dataset = load_from_disk(str(eval_path))

    use_bf16, use_fp16, use_tf32 = resolve_precision_settings(train_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=resolve_model_dtype(
            model_cfg,
            use_bf16=use_bf16,
            use_fp16=use_fp16,
            deepspeed_enabled=args.deepspeed is not None,
        ),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    if train_cfg.get("gradient_checkpointing", model_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model = get_peft_model(model, build_lora_config(model_cfg, train_cfg))
    model.print_trainable_parameters()

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
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_tf32,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        remove_unused_columns=train_cfg.get("remove_unused_columns", False),
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", False),
    )
    save_model_artifacts = bool(train_cfg.get("save_model_artifacts", True))
    training_args_signature = signature(TrainingArguments.__init__).parameters
    if "overwrite_output_dir" in training_args_signature:
        training_kwargs["overwrite_output_dir"] = train_cfg.get("overwrite_output_dir", False)
    if "include_num_input_tokens_seen" in training_args_signature:
        training_kwargs["include_num_input_tokens_seen"] = train_cfg.get("include_num_input_tokens_seen", "all")
    if "save_only_model" in training_args_signature:
        training_kwargs["save_only_model"] = train_cfg.get("save_only_model", False)
    if "save_strategy" in training_args_signature and not save_model_artifacts:
        training_kwargs["save_strategy"] = "no"
    if "run_name" in training_args_signature and "experiment_name" in train_cfg:
        training_kwargs["run_name"] = train_cfg["experiment_name"]

    training_args = TrainingArguments(**training_kwargs)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
    )

    callbacks = []
    cuda_memory_logging_steps = int(train_cfg.get("cuda_memory_logging_steps", train_cfg.get("logging_steps", 10)))
    if cuda_memory_logging_steps > 0:
        callbacks.append(CudaMemoryLoggingCallback(cuda_memory_logging_steps))
    torch_profiler_cfg = resolve_torch_profiler_config(train_cfg, output_dir)
    if torch_profiler_cfg.get("enabled", False):
        callbacks.append(TorchProfilerCallback(**torch_profiler_cfg))

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
        save_model_artifacts=save_model_artifacts,
    )
    trainer_signature = signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = MemoryMetricTrainer(**trainer_kwargs)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()

    deepspeed_comms_summary_path = export_deepspeed_comms_summary(
        trainer=trainer,
        output_dir=output_dir,
        train_cfg=train_cfg,
    )

    if trainer.is_world_process_zero():
        train_summary = build_train_summary(
            trainer=trainer,
            train_result_metrics=train_result.metrics,
            train_cfg=train_cfg,
            tokenized_dataset_path=tokenized_dataset_path,
            resume_from_checkpoint=args.resume_from_checkpoint,
            deepspeed_comms_summary_path=deepspeed_comms_summary_path,
            torch_profiler_dir=torch_profiler_cfg.get("output_dir") if torch_profiler_cfg.get("enabled", False) else None,
        )
        train_summary["lora"] = {
            "r": trainer.model.peft_config["default"].r,
            "lora_alpha": trainer.model.peft_config["default"].lora_alpha,
            "lora_dropout": trainer.model.peft_config["default"].lora_dropout,
            "target_modules": sorted(trainer.model.peft_config["default"].target_modules),
        }
        trainer.state.log_history.append(
            {
                "resume_requested": train_summary["resume_requested"],
                "resume_success": train_summary["resume_success"],
                "num_input_tokens_seen": train_summary["num_input_tokens_seen"],
                "train_tokens_per_second": train_summary["train_tokens_per_second"],
                "step": trainer.state.global_step,
            }
        )

    if trainer.is_deepspeed_enabled and trainer.save_model_artifacts:
        trainer.save_model(output_dir)

    if trainer.is_world_process_zero():
        if not trainer.is_deepspeed_enabled and trainer.save_model_artifacts:
            trainer.save_model(output_dir)
        if trainer.save_model_artifacts:
            tokenizer.save_pretrained(output_dir)

        with open(Path(output_dir) / "train_log_history.json", "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
        with open(Path(output_dir) / "train_summary.json", "w", encoding="utf-8") as f:
            json.dump(train_summary, f, ensure_ascii=False, indent=2)

        if trainer.save_model_artifacts:
            print(f"[DONE] SFT training finished. adapter saved to {output_dir}")
        else:
            print(f"[DONE] SFT training finished. model saving disabled; logs saved to {output_dir}")

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
