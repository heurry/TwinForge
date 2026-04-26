#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
from inspect import signature
from pathlib import Path
from typing import Any, Dict
from time import perf_counter

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
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

try:
    from src.train.callbacks import CudaMemoryLoggingCallback, TorchProfilerCallback
except ModuleNotFoundError as exc:
    if exc.name != "src":
        raise
    from callbacks import CudaMemoryLoggingCallback, TorchProfilerCallback


class MemoryMetricTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.save_model_artifacts = bool(kwargs.pop("save_model_artifacts", True))
        super().__init__(*args, **kwargs)
        self.checkpoint_history: list[Dict[str, Any]] = []

    def log(self, logs: Dict[str, float], start_time: float | None = None) -> None:
        merged_logs = dict(logs)
        for callback in self.callback_handler.callbacks:
            consume_pending_logs = getattr(callback, "consume_pending_logs", None)
            if callable(consume_pending_logs):
                merged_logs.update(consume_pending_logs(self.state.global_step))
        super().log(merged_logs, start_time=start_time)

    def _save_checkpoint(self, model, trial) -> None:
        if not self.save_model_artifacts:
            return

        checkpoint_dir = Path(self._get_output_dir(trial=trial)) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        save_started_at = perf_counter()
        super()._save_checkpoint(model, trial)
        save_seconds = round(perf_counter() - save_started_at, 4)

        if not self.is_world_process_zero():
            return

        checkpoint_record = {
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_save_seconds": save_seconds,
            "checkpoint_size_bytes": compute_directory_size_bytes(checkpoint_dir),
            "step": self.state.global_step,
        }
        self.checkpoint_history.append(checkpoint_record)
        self.state.log_history.append(dict(checkpoint_record))


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_dtype(
    model_cfg: Dict[str, Any],
    use_bf16: bool,
    use_fp16: bool,
    deepspeed_enabled: bool,
) -> Any:
    # Native Trainer mixed precision expects fp32 master weights and uses autocast/scaler at runtime.
    # DeepSpeed mixed precision manages low-precision parameter copies itself, so we align the load dtype there.
    if use_bf16 or use_fp16:
        if deepspeed_enabled:
            return torch.bfloat16 if use_bf16 else torch.float16
        return torch.float32

    dtype_name = model_cfg.get("torch_dtype", "auto")
    if dtype_name not in (None, "auto") and not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype in model config: {dtype_name}")
    if dtype_name in (None, "auto"):
        return "auto"
    return getattr(torch, dtype_name)


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


def compute_directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0

    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def resolve_torch_profiler_config(train_cfg: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    profiler_cfg = dict(train_cfg.get("torch_profiler", {}) or {})
    if not profiler_cfg.get("enabled", False):
        return {"enabled": False}

    output_subdir = profiler_cfg.get("output_subdir", "profiling/torch")
    return {
        "enabled": True,
        "output_dir": str(Path(output_dir) / output_subdir),
        "wait": int(profiler_cfg.get("wait", 1)),
        "warmup": int(profiler_cfg.get("warmup", 1)),
        "active": int(profiler_cfg.get("active", 3)),
        "repeat": int(profiler_cfg.get("repeat", 1)),
        "record_shapes": bool(profiler_cfg.get("record_shapes", True)),
        "profile_memory": bool(profiler_cfg.get("profile_memory", True)),
        "with_stack": bool(profiler_cfg.get("with_stack", False)),
        "with_flops": bool(profiler_cfg.get("with_flops", True)),
        "row_limit": int(profiler_cfg.get("row_limit", 50)),
        "export_chrome_trace": bool(profiler_cfg.get("export_chrome_trace", True)),
    }


def export_deepspeed_comms_summary(
    trainer: MemoryMetricTrainer,
    output_dir: str,
    train_cfg: Dict[str, Any],
) -> str | None:
    comms_cfg = dict(train_cfg.get("deepspeed_comms_logging", {}) or {})
    if not trainer.is_deepspeed_enabled or not comms_cfg.get("enabled", False):
        return None

    import deepspeed.comm as ds_comm

    if not ds_comm.has_comm_data():
        return None

    comms_summary = ds_comm.log_summary(
        show_straggler=bool(comms_cfg.get("show_straggler", False)),
        return_dict=True,
    )

    if not trainer.is_world_process_zero():
        return None

    output_subdir = comms_cfg.get("output_subdir", "profiling/deepspeed")
    comms_output_dir = Path(output_dir) / output_subdir
    comms_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = comms_output_dir / "deepspeed_comms_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(comms_summary, f, ensure_ascii=False, indent=2)
    return str(summary_path)


def build_train_summary(
    trainer: MemoryMetricTrainer,
    train_result_metrics: Dict[str, Any],
    train_cfg: Dict[str, Any],
    tokenized_dataset_path: Path,
    resume_from_checkpoint: str | None,
    deepspeed_comms_summary_path: str | None = None,
    torch_profiler_dir: str | None = None,
) -> Dict[str, Any]:
    metrics = dict(train_result_metrics)
    num_input_tokens_seen = int(getattr(trainer.state, "num_input_tokens_seen", 0))
    train_runtime = float(metrics.get("train_runtime", 0.0) or 0.0)
    train_tokens_per_second = metrics.get("train_tokens_per_second")
    if train_tokens_per_second is None and train_runtime > 0 and num_input_tokens_seen > 0:
        train_tokens_per_second = num_input_tokens_seen / train_runtime

    last_checkpoint = trainer.checkpoint_history[-1] if trainer.checkpoint_history else {}
    return {
        "experiment_tier": train_cfg.get("experiment_tier", "unspecified"),
        "experiment_name": train_cfg.get("experiment_name", Path(trainer.args.output_dir).name),
        "training_backend": train_cfg.get("training_backend", "trainer"),
        "benchmark_group": train_cfg.get("benchmark_group"),
        "output_dir": trainer.args.output_dir,
        "logging_dir": train_cfg.get("logging_dir"),
        "tokenized_dataset_path": str(tokenized_dataset_path),
        "world_size": int(trainer.args.world_size),
        "per_device_train_batch_size": int(trainer.args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(trainer.args.gradient_accumulation_steps),
        "effective_batch_size": int(
            trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps * trainer.args.world_size
        ),
        "dataloader_num_workers": int(trainer.args.dataloader_num_workers),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", False)),
        "max_seq_length": int(train_cfg.get("max_seq_length", 0) or 0),
        "precision": {
            "bf16": bool(trainer.args.bf16),
            "fp16": bool(trainer.args.fp16),
            "tf32": bool(getattr(trainer.args, "tf32", False)),
        },
        "max_steps": int(trainer.args.max_steps),
        "logging_steps": int(trainer.args.logging_steps),
        "save_steps": int(trainer.args.save_steps),
        "save_model_artifacts": bool(trainer.save_model_artifacts),
        "global_step": int(trainer.state.global_step),
        "resume_requested": resume_from_checkpoint is not None,
        "resume_checkpoint": resume_from_checkpoint,
        "resume_success": True if resume_from_checkpoint is not None else None,
        "oom_or_nan": False,
        "run_status": "completed",
        "num_input_tokens_seen": num_input_tokens_seen,
        "train_tokens_per_second": train_tokens_per_second,
        "torch_profiler_dir": torch_profiler_dir,
        "deepspeed_comms_summary_path": deepspeed_comms_summary_path,
        "last_checkpoint_save_seconds": last_checkpoint.get("checkpoint_save_seconds"),
        "last_checkpoint_size_bytes": last_checkpoint.get("checkpoint_size_bytes"),
        "checkpoint_history": trainer.checkpoint_history,
        "train_metrics": metrics,
    }


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

    tokenized_dataset_path = Path(train_cfg.get("tokenized_dataset_path", args.tokenized_dataset_path))
    if not tokenized_dataset_path.exists():
        raise FileNotFoundError(
            f"Tokenized CPT dataset not found at {tokenized_dataset_path}. "
            "Run `python scripts/02_build_cpt_corpus.py` and "
            "`python scripts/04_tokenize_cpt.py` first."
        )
    print(f"[INFO] loading tokenized CPT dataset from {tokenized_dataset_path}")
    train_dataset = load_from_disk(str(tokenized_dataset_path))

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
    cuda_memory_logging_steps = int(
        train_cfg.get("cuda_memory_logging_steps", train_cfg.get("logging_steps", 10))
    )
    torch_profiler_cfg = resolve_torch_profiler_config(train_cfg, output_dir)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        save_model_artifacts=save_model_artifacts,
    )
    callbacks = []
    if cuda_memory_logging_steps > 0:
        callbacks.append(CudaMemoryLoggingCallback(cuda_memory_logging_steps))
    if torch_profiler_cfg.get("enabled", False):
        callbacks.append(TorchProfilerCallback(**torch_profiler_cfg))
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks
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
        trainer.state.log_history.append(
            {
                "resume_requested": train_summary["resume_requested"],
                "resume_success": train_summary["resume_success"],
                "num_input_tokens_seen": train_summary["num_input_tokens_seen"],
                "train_tokens_per_second": train_summary["train_tokens_per_second"],
                "checkpoint_save_seconds": train_summary["last_checkpoint_save_seconds"],
                "checkpoint_size_bytes": train_summary["last_checkpoint_size_bytes"],
                "oom_or_nan": train_summary["oom_or_nan"],
                "step": trainer.state.global_step,
            }
        )

    # ZeRO-3 final model consolidation uses distributed collectives, so every rank
    # must enter save_model even though only the main process writes files.
    if trainer.is_deepspeed_enabled and trainer.save_model_artifacts:
        trainer.save_model(output_dir)

    if trainer.is_world_process_zero():
        if not trainer.is_deepspeed_enabled and trainer.save_model_artifacts:
            trainer.save_model(output_dir)
        if trainer.save_model_artifacts:
            tokenizer.save_pretrained(output_dir)

        metrics = trainer.state.log_history
        with open(Path(output_dir) / "train_log_history.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(Path(output_dir) / "train_summary.json", "w", encoding="utf-8") as f:
            json.dump(train_summary, f, ensure_ascii=False, indent=2)

        if trainer.save_model_artifacts:
            print(f"[DONE] CPT training finished. model saved to {output_dir}")
        else:
            print(f"[DONE] CPT training finished. model saving disabled; logs saved to {output_dir}")

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
