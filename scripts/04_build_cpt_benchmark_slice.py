#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from datasets import load_from_disk


TOKENIZER_FINGERPRINT_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "chat_template.jinja",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a deterministic benchmark slice from tokenized CPT data.")
    parser.add_argument(
        "--slice_config",
        type=str,
        default="configs/data/cpt_benchmark_slice.yaml",
        help="Path to the benchmark slice YAML spec.",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default=None,
        help="Optional override for the full tokenized CPT dataset path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional override for the benchmark slice dataset path.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Optional override for the benchmark slice summary path.",
    )
    parser.add_argument(
        "--indices_path",
        type=str,
        default=None,
        help="Optional override for the benchmark slice indices path.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Optional override for the model config used to resolve tokenizer metadata.",
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default=None,
        choices=["head"],
        help="Selection strategy for benchmark slice generation.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Optional starting index for `head` selection.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Optional override for the number of packed samples to keep.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing benchmark slice outputs.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0

    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def compute_tokenizer_fingerprint(tokenizer_path: Path) -> tuple[str | None, list[str]]:
    if not tokenizer_path.exists() or not tokenizer_path.is_dir():
        return None, []

    selected_files = [tokenizer_path / filename for filename in TOKENIZER_FINGERPRINT_FILES if (tokenizer_path / filename).exists()]
    if not selected_files:
        return None, []

    hasher = hashlib.sha256()
    relative_paths: list[str] = []
    for file_path in selected_files:
        relative_path = str(file_path.relative_to(tokenizer_path))
        relative_paths.append(relative_path)
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(file_path.read_bytes())
    return hasher.hexdigest(), relative_paths


def resolve_slice_spec(args: argparse.Namespace) -> dict[str, Any]:
    slice_config_path = Path(args.slice_config)
    spec = read_yaml(slice_config_path) if slice_config_path.exists() else {}

    source_path = args.source_path or spec.get("source_path", "data/tokenized/cpt/train")
    output_path = args.output_path or spec.get("output_path", "data/tokenized/cpt/bench")
    summary_path = args.summary_path or spec.get("summary_path", "data/tokenized/cpt/bench_summary.json")
    indices_path = args.indices_path or spec.get("indices_path", "data/tokenized/cpt/bench_indices.json")
    model_config = args.model_config or spec.get("model_config", "configs/model/qwen3_1_7b_base.yaml")
    selection_strategy = args.selection_strategy or spec.get("selection_strategy", "head")
    start_index = args.start_index if args.start_index is not None else int(spec.get("start_index", 0))
    num_samples = args.num_samples if args.num_samples is not None else int(spec.get("num_samples", 1024))

    if selection_strategy != "head":
        raise ValueError(f"Unsupported selection strategy: {selection_strategy}")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    return {
        "slice_config_path": str(slice_config_path),
        "source_path": str(source_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "indices_path": str(indices_path),
        "model_config": str(model_config),
        "selection_strategy": selection_strategy,
        "start_index": start_index,
        "num_samples": num_samples,
        "overwrite": bool(args.overwrite),
    }


def load_tokenizer_metadata(model_config_path: Path) -> dict[str, Any]:
    if not model_config_path.exists():
        return {
            "model_config_path": str(model_config_path),
            "tokenizer_name_or_path": None,
            "tokenizer_fingerprint": None,
            "tokenizer_fingerprint_files": [],
        }

    model_cfg = read_yaml(model_config_path)
    tokenizer_name_or_path = model_cfg.get("tokenizer_name_or_path")
    tokenizer_path = Path(tokenizer_name_or_path) if tokenizer_name_or_path else None
    tokenizer_fingerprint = None
    tokenizer_files: list[str] = []
    if tokenizer_path is not None:
        tokenizer_fingerprint, tokenizer_files = compute_tokenizer_fingerprint(tokenizer_path)

    return {
        "model_config_path": str(model_config_path),
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "tokenizer_fingerprint_files": tokenizer_files,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    spec = resolve_slice_spec(args)

    source_path = Path(spec["source_path"])
    output_path = Path(spec["output_path"])
    summary_path = Path(spec["summary_path"])
    indices_path = Path(spec["indices_path"])
    model_config_path = Path(spec["model_config"])

    if not source_path.exists():
        raise FileNotFoundError(f"Tokenized CPT dataset not found at {source_path}")

    if output_path.exists():
        if not spec["overwrite"]:
            raise FileExistsError(f"Benchmark slice already exists at {output_path}. Use --overwrite to rebuild it.")
        shutil.rmtree(output_path)
    if spec["overwrite"]:
        for path in (summary_path, indices_path):
            if path.exists():
                path.unlink()

    dataset = load_from_disk(str(source_path))
    available_samples = len(dataset)
    start_index = min(spec["start_index"], available_samples)
    end_index = min(start_index + spec["num_samples"], available_samples)
    indices = list(range(start_index, end_index))
    if not indices:
        raise ValueError(
            f"Benchmark slice is empty. available_samples={available_samples}, "
            f"start_index={spec['start_index']}, num_samples={spec['num_samples']}"
        )
    benchmark_dataset = dataset.select(indices)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_dataset.save_to_disk(str(output_path))

    tokenizer_metadata = load_tokenizer_metadata(model_config_path)
    created_at = datetime.now(timezone.utc).isoformat()
    source_fingerprint = getattr(dataset, "_fingerprint", None)
    benchmark_fingerprint = getattr(benchmark_dataset, "_fingerprint", None)

    write_json(
        indices_path,
        {
            "source_path": str(source_path),
            "selection_strategy": spec["selection_strategy"],
            "start_index": start_index,
            "num_indices": len(indices),
            "indices": indices,
        },
    )
    write_json(
        summary_path,
        {
            "created_at_utc": created_at,
            "slice_config_path": spec["slice_config_path"],
            "source_path": str(source_path),
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "indices_path": str(indices_path),
            "selection_strategy": spec["selection_strategy"],
            "start_index": start_index,
            "requested_num_samples": int(spec["num_samples"]),
            "selected_num_samples": int(len(indices)),
            "available_num_samples": int(available_samples),
            "source_dataset_fingerprint": source_fingerprint,
            "benchmark_dataset_fingerprint": benchmark_fingerprint,
            "source_dataset_size_bytes": compute_directory_size_bytes(source_path),
            "benchmark_dataset_size_bytes": compute_directory_size_bytes(output_path),
            **tokenizer_metadata,
        },
    )

    print(f"[DONE] wrote CPT benchmark slice to {output_path} with {len(indices)} samples")
    print(f"[DONE] wrote benchmark indices to {indices_path}")
    print(f"[DONE] wrote benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
