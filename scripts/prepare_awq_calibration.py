#!/usr/bin/env python3
"""Build a deterministic, held-out-safe AWQ calibration set from DianJin SFT data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in record.get("messages", [])
        if item.get("role") in {"system", "user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]
    response = record.get("response")
    if isinstance(response, str) and response.strip():
        messages.append({"role": "assistant", "content": response.strip()})
    return messages


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rng = random.Random(args.seed)
    reservoir: list[dict[str, Any]] = []
    valid_count = 0

    with args.source.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            messages = build_messages(record)
            if not messages or not any(item["role"] == "user" for item in messages):
                continue
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            sample = {
                "text": text,
                "source": record.get("source", "unknown"),
                "source_line": line_number,
            }
            valid_count += 1
            if len(reservoir) < args.samples:
                reservoir.append(sample)
                continue
            replacement = rng.randrange(valid_count)
            if replacement < args.samples:
                reservoir[replacement] = sample

    if len(reservoir) < args.samples:
        raise RuntimeError(
            f"requested {args.samples} samples, found only {len(reservoir)} valid rows"
        )

    rng.shuffle(reservoir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for sample in reservoir:
            output.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")

    sources = Counter(sample["source"] for sample in reservoir)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(reservoir),
                "eligible_rows": valid_count,
                "seed": args.seed,
                "source_categories": dict(sorted(sources.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
