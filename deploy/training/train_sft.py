"""Text-only LoRA SFT entrypoint for Qwen3.5 customer-service conversations."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import boto3
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForImageTextToText, AutoTokenizer


SYSTEM_PROMPT = "你是一名专业、礼貌且合规的中文客服，请结合对话上下文准确解决客户问题。"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


class ConversationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str, tokenizer: Any, max_length: int, max_samples: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                if max_samples > 0 and len(rows) >= max_samples:
                    break
        if not rows:
            raise ValueError(f"no SFT samples found in {path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        history = row["messages"]
        if not any(message.get("role") == "user" for message in history):
            history = [{"role": "user", "content": "您好"}, *history]
        context = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        response = row["response"]
        prefix_ids = self.tokenizer.apply_chat_template(
            context,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        full_ids = self.tokenizer.apply_chat_template(
            [*context, {"role": "assistant", "content": response}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        labels = [-100] * min(len(prefix_ids), len(full_ids)) + full_ids[len(prefix_ids) :]
        if all(label == -100 for label in labels):
            raise ValueError(f"sample {index} has no assistant response tokens")
        if len(full_ids) > self.max_length:
            trim = len(full_ids) - self.max_length
            full_ids = full_ids[trim:]
            labels = labels[trim:]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(tokenizer: Any, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row, item in enumerate(batch):
        size = item["input_ids"].numel()
        input_ids[row, :size] = item["input_ids"]
        labels[row, :size] = item["labels"]
        attention_mask[row, :size] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def language_lora_targets(model: torch.nn.Module) -> list[str]:
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [
        name
        for name, _ in model.named_modules()
        if name.startswith("model.language_model.layers.") and name.rsplit(".", 1)[-1] in suffixes
    ]
    if not targets:
        raise RuntimeError("Qwen3.5 language LoRA target modules were not found")
    return targets


def upload_adapter(output_dir: Path, artifact_key: str) -> None:
    endpoint = os.environ["S3_ENDPOINT"]
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "http://" + endpoint
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="us-east-1",
    )
    bucket = os.getenv("S3_BUCKET", "infra-artifacts")
    for path in output_dir.rglob("*"):
        if path.is_file():
            key = f"{artifact_key.rstrip('/')}/{path.relative_to(output_dir).as_posix()}"
            client.upload_file(str(path), bucket, key)
            print(f"uploaded s3://{bucket}/{key}", flush=True)


def main() -> None:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    distributed = world_size > 1
    torch.cuda.set_device(local_rank)
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if distributed else 0

    seed = env_int("SEED", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    base_model = os.environ["BASE_MODEL"]
    dataset_uri = os.environ["DATASET_URI"]
    output_dir = Path(os.getenv("OUTPUT_DIR", "/output/adapter"))
    precision = os.getenv("PRECISION", "bf16").lower()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = ConversationDataset(
        dataset_uri,
        tokenizer,
        max_length=env_int("MAX_SEQ_LENGTH", 1024),
        max_samples=env_int("MAX_SAMPLES", 0),
    )
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1),
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=lambda rows: collate(tokenizer, rows),
        num_workers=0,
        pin_memory=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if env_bool("GRADIENT_CHECKPOINTING", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=env_int("LORA_RANK", 16),
            lora_alpha=env_int("LORA_ALPHA", 32),
            lora_dropout=env_float("LORA_DROPOUT", 0.05),
            bias="none",
            target_modules=language_lora_targets(model),
        ),
    )
    model.to(local_rank)
    if rank == 0:
        model.print_trainable_parameters()

    gradient_accumulation = env_int("GRADIENT_ACCUMULATION_STEPS", 8)
    deepspeed_mode = os.getenv("DEEPSPEED", "off").lower()
    engine = None
    optimizer = None
    if deepspeed_mode in {"zero2", "zero3"}:
        import deepspeed

        stage = 2 if deepspeed_mode == "zero2" else 3
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=[parameter for parameter in model.parameters() if parameter.requires_grad],
            config={
                "train_micro_batch_size_per_gpu": env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1),
                "gradient_accumulation_steps": gradient_accumulation,
                "gradient_clipping": 1.0,
                "bf16": {"enabled": precision == "bf16"},
                "fp16": {"enabled": precision == "fp16"},
                "zero_optimization": {"stage": stage},
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": env_float("LEARNING_RATE", 2e-4),
                        "weight_decay": 0.01,
                        "torch_adam": True,
                    },
                },
            },
        )
        train_model = engine
    else:
        train_model = DistributedDataParallel(model, device_ids=[local_rank]) if distributed else model
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=env_float("LEARNING_RATE", 2e-4),
            weight_decay=0.01,
        )

    epochs = env_int("EPOCHS", 1)
    total_steps = max(1, math.ceil(len(loader) * epochs / gradient_accumulation))
    global_step = 0
    train_model.train()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            batch = {key: value.to(local_rank, non_blocking=True) for key, value in batch.items()}
            loss = train_model(**batch).loss
            if engine is not None:
                engine.backward(loss)
                engine.step()
            else:
                (loss / gradient_accumulation).backward()
                if (step + 1) % gradient_accumulation == 0 or step + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            if (step + 1) % gradient_accumulation == 0 or step + 1 == len(loader):
                global_step += 1
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "step": global_step,
                                "total_steps": total_steps,
                                "loss": round(float(loss.detach().cpu()), 6),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    if distributed:
        dist.barrier()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = engine.module if engine is not None else (train_model.module if distributed else train_model)
        model_to_save.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        upload_adapter(output_dir, os.environ["OUTPUT_URI"])
        print(f"adapter saved to {output_dir}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
