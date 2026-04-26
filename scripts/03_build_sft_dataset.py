#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import jsonlines

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sft import assign_split, normalize_messages


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned SFT dataset from raw message data.")
    parser.add_argument("--manifest", type=str, default="configs/dataset_manifest.json")
    parser.add_argument("--input_root", type=str, default="data/raw/sft")
    parser.add_argument("--output_root", type=str, default="data/cleaned/sft")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict_missing", action="store_true")
    return parser.parse_args()


def resolve_input_paths(manifest: Dict[str, Any], input_root: Path, strict_missing: bool) -> tuple[List[Path], List[Path]]:
    input_paths: List[Path] = []
    missing_paths: List[Path] = []
    for entry in manifest["datasets"].get("sft", []):
        path = input_root / f"{entry['name']}.jsonl"
        if path.exists():
            input_paths.append(path)
        else:
            missing_paths.append(path)

    if strict_missing and missing_paths:
        raise FileNotFoundError(f"Missing SFT raw files: {', '.join(str(path) for path in missing_paths)}")
    if not input_paths:
        raise FileNotFoundError(f"No SFT raw files found under {input_root}")
    return input_paths, missing_paths


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    train_path = output_root / "train.jsonl"
    val_path = output_root / "val.jsonl"
    summary_path = output_root / "summary.json"

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_paths, missing_paths = resolve_input_paths(
        manifest=manifest,
        input_root=input_root,
        strict_missing=args.strict_missing,
    )

    summary: Dict[str, Any] = {
        "input_root": str(input_root),
        "input_paths": [str(path) for path in input_paths],
        "missing_inputs": [str(path) for path in missing_paths],
        "output_root": str(output_root),
        "val_ratio": args.val_ratio,
        "totals": {
            "raw_records": 0,
            "cleaned_records": 0,
            "train_records": 0,
            "val_records": 0,
            "dropped_records": 0,
        },
        "source_stats": defaultdict(
            lambda: {
                "raw_records": 0,
                "cleaned_records": 0,
                "train_records": 0,
                "val_records": 0,
                "dropped_records": 0,
            }
        ),
    }

    with jsonlines.open(train_path, mode="w") as train_writer, jsonlines.open(val_path, mode="w") as val_writer:
        for input_path in input_paths:
            source_name = input_path.stem
            with jsonlines.open(input_path, mode="r") as reader:
                for row in reader:
                    summary["totals"]["raw_records"] += 1
                    summary["source_stats"][source_name]["raw_records"] += 1

                    normalized_messages = normalize_messages(row.get("messages"))
                    if normalized_messages is None:
                        summary["totals"]["dropped_records"] += 1
                        summary["source_stats"][source_name]["dropped_records"] += 1
                        continue

                    cleaned_record = {
                        "messages": normalized_messages,
                        "source": source_name,
                    }
                    split = assign_split(normalized_messages, args.val_ratio)

                    summary["totals"]["cleaned_records"] += 1
                    summary["source_stats"][source_name]["cleaned_records"] += 1

                    if split == "val":
                        val_writer.write(cleaned_record)
                        summary["totals"]["val_records"] += 1
                        summary["source_stats"][source_name]["val_records"] += 1
                    else:
                        train_writer.write(cleaned_record)
                        summary["totals"]["train_records"] += 1
                        summary["source_stats"][source_name]["train_records"] += 1

    summary["source_stats"] = dict(summary["source_stats"])
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] cleaned SFT train dataset saved to {train_path}")
    print(f"[DONE] cleaned SFT val dataset saved to {val_path}")
    print(f"[DONE] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
