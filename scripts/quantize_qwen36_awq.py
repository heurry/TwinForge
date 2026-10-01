#!/usr/bin/env python3
"""Quantize Qwen3.6-27B to activation-aware W4A16 with LLM Compressor."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration", type=Path)
    return parser.parse_args()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def write_status(path: Path, stage: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def validate_source(model_path: Path) -> dict[str, Any]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    missing = [name for name in shards if not (model_path / name).is_file()]
    if missing:
        raise RuntimeError(f"missing model shards: {missing}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(config, "model_type", None) != "qwen3_5":
        raise RuntimeError(f"unexpected model type: {config.model_type}")
    return {
        "architecture": config.architectures[0],
        "model_type": config.model_type,
        "shards": len(shards),
    }


def add_unquantized_visual_tower(model_path: Path, output_path: Path) -> None:
    """Restore Qwen3.6's BF16 visual tower around the quantized text model."""
    source_index_path = model_path / "model.safetensors.index.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    visual_weight_map = {
        name: shard
        for name, shard in source_index["weight_map"].items()
        if name.startswith("model.visual.")
    }
    if not visual_weight_map:
        raise RuntimeError("source checkpoint contains no model.visual weights")

    visual_tensors: dict[str, torch.Tensor] = {}
    shards = sorted(set(visual_weight_map.values()))
    for shard in shards:
        names = [
            name for name, filename in visual_weight_map.items() if filename == shard
        ]
        with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
            for name in names:
                visual_tensors[name] = handle.get_tensor(name)

    visual_filename = "model-vision-bf16.safetensors"
    save_file(visual_tensors, output_path / visual_filename)

    quantized_files = sorted(
        path
        for path in output_path.glob("*.safetensors")
        if path.name != visual_filename
    )
    if not quantized_files:
        raise RuntimeError("quantized checkpoint contains no safetensors weights")

    weight_map: dict[str, str] = {}
    for checkpoint in quantized_files:
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in weight_map:
                    raise RuntimeError(f"duplicate quantized tensor: {name}")
                weight_map[name] = checkpoint.name
    for name in visual_tensors:
        if name in weight_map:
            raise RuntimeError(f"duplicate visual tensor: {name}")
        weight_map[name] = visual_filename

    tensor_files = [*quantized_files, output_path / visual_filename]
    index = {
        "metadata": {"total_size": sum(path.stat().st_size for path in tensor_files)},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (output_path / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    model_path = resolve_path(root, config["model_path"])
    calibration_path = args.calibration or resolve_path(
        root, config["calibration_path"]
    )
    output_path = args.output or resolve_path(root, config["output_path"])
    offload_path = resolve_path(root, config["offload_path"])
    status_path = resolve_path(root, config["status_path"])
    samples = args.samples or int(config["num_calibration_samples"])
    max_seq_length = args.max_seq_length or int(config["max_seq_length"])
    partial_path = output_path.with_name(output_path.name + ".partial")

    source_info = validate_source(model_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    if partial_path.exists():
        shutil.rmtree(partial_path)
    partial_path.mkdir(parents=True)
    if offload_path.exists():
        shutil.rmtree(offload_path)
    offload_path.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    base_status = {
        "model_path": str(model_path),
        "output_path": str(output_path),
        "calibration_path": str(calibration_path),
        "samples": samples,
        "max_seq_length": max_seq_length,
        "scheme": config["scheme"],
        **source_info,
    }
    write_status(status_path, "loading_model", **base_status)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    calibration = load_dataset(
        "json", data_files=str(calibration_path), split=f"train[:{samples}]"
    )
    if len(calibration) != samples:
        raise RuntimeError(f"expected {samples} samples, loaded {len(calibration)}")

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    max_memory = {
        index: f"{int(config.get('max_gpu_memory_gib', 20))}GiB"
        for index in range(torch.cuda.device_count())
    }
    max_memory["cpu"] = f"{int(config.get('max_cpu_memory_gib', 32))}GiB"
    try:
        with load_context():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype="auto",
                device_map="auto",
                max_memory=max_memory,
                offload_folder=str(offload_path),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        write_status(
            status_path,
            "quantizing",
            elapsed_seconds=round(time.monotonic() - started, 1),
            device_map={
                str(k): str(v)
                for k, v in getattr(model, "hf_device_map", {}).items()
            },
            **base_status,
        )
        recipe = [
            AWQModifier(
                duo_scaling=config.get("duo_scaling", "both"),
                offload_device=torch.device("cpu"),
            ),
            QuantizationModifier(
                ignore=config.get("ignore", ["lm_head"]),
                scheme=config.get("scheme", "W4A16_ASYM"),
                targets=config.get("targets", ["Linear"]),
            ),
        ]
        oneshot(
            model=model,
            processor=tokenizer,
            dataset=calibration,
            recipe=recipe,
            output_dir=str(partial_path),
            num_calibration_samples=samples,
            max_seq_length=max_seq_length,
            batch_size=1,
            data_collator="truncation",
            pipeline=config.get("pipeline", "sequential"),
            sequential_offload_device=config.get(
                "sequential_offload_device", "cpu"
            ),
            shuffle_calibration_samples=False,
            save_compressed=True,
        )

        quant_config_path = partial_path / "config.json"
        saved_config = json.loads(quant_config_path.read_text(encoding="utf-8"))
        quant_config = saved_config.get("quantization_config")
        if not quant_config:
            raise RuntimeError("saved model is missing quantization_config")
        deployment_config = json.loads(
            (model_path / "config.json").read_text(encoding="utf-8")
        )
        ignored_modules = list(quant_config.get("ignore", []))
        for ignored_module in config.get(
            "deployment_ignore", [r"re:^visual\..*"]
        ):
            if ignored_module not in ignored_modules:
                ignored_modules.append(ignored_module)
        quant_config["ignore"] = ignored_modules
        deployment_config["quantization_config"] = quant_config
        quant_config_path.write_text(
            json.dumps(deployment_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        add_unquantized_visual_tower(model_path, partial_path)
        partial_path.replace(output_path)
        write_status(
            status_path,
            "completed",
            elapsed_seconds=round(time.monotonic() - started, 1),
            quantization_config=quant_config,
            **base_status,
        )
    except Exception as exc:
        write_status(
            status_path,
            "failed",
            elapsed_seconds=round(time.monotonic() - started, 1),
            error=f"{type(exc).__name__}: {exc}",
            **base_status,
        )
        raise


if __name__ == "__main__":
    main()
