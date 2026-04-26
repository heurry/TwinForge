#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import jsonlines
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sft import SFTPackingStats, TokenizedSFTDocument, iter_packed_sft_examples, tokenize_sft_messages


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize and pack cleaned SFT dataset.")
    parser.add_argument("--manifest", type=str, default="configs/dataset_manifest.json")
    parser.add_argument("--model_config", type=str, default="configs/model/qwen3_1_7b_base.yaml")
    parser.add_argument("--input_root", type=str, default="data/cleaned/sft")
    parser.add_argument("--output_root", type=str, default="data/tokenized/sft")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--seq_length", type=int, default=None)
    parser.add_argument("--train_on_prompt", action="store_true")
    parser.add_argument("--drop_remainder", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_tokenizer_name_or_path(args: argparse.Namespace, manifest: Dict[str, Any]) -> str:
    if args.tokenizer_name_or_path:
        return args.tokenizer_name_or_path
    if args.model_config:
        model_cfg = read_yaml(args.model_config)
        return model_cfg["tokenizer_name_or_path"]
    return manifest["tokenizer"]


def iter_cleaned_messages(path: Path) -> Iterable[Dict[str, Any]]:
    with jsonlines.open(path, mode="r") as reader:
        for row in reader:
            messages = row.get("messages")
            if isinstance(messages, list) and messages:
                yield {
                    "messages": messages,
                    "source": row.get("source", path.stem),
                }


class PackedSFTDatasetBuilder:
    def __init__(
        self,
        input_path: Path,
        tokenizer_name_or_path: str,
        seq_length: int,
        packing: bool,
        train_on_prompt: bool,
        drop_remainder: bool,
    ) -> None:
        self.input_path = input_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.seq_length = seq_length
        self.packing = packing
        self.train_on_prompt = train_on_prompt
        self.drop_remainder = drop_remainder
        self.summary: Dict[str, Any] = {}

    def _reset_summary(self) -> None:
        self.summary = {
            "input_path": str(self.input_path),
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "seq_length": self.seq_length,
            "packing": self.packing,
            "train_on_prompt": self.train_on_prompt,
            "drop_remainder": self.drop_remainder,
        }

    def _iter_tokenized_documents(self, tokenizer) -> Iterable[TokenizedSFTDocument]:
        for row in iter_cleaned_messages(self.input_path):
            tokenized = tokenize_sft_messages(
                tokenizer=tokenizer,
                messages=row["messages"],
                train_on_prompt=self.train_on_prompt,
            )
            yield TokenizedSFTDocument(
                input_ids=tokenized.input_ids,
                labels=tokenized.labels,
                source=row.get("source"),
            )

    def generate(self) -> Iterable[Dict[str, List[int]]]:
        self._reset_summary()
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True)
        stats = SFTPackingStats()

        if self.packing:
            for example in iter_packed_sft_examples(
                documents=self._iter_tokenized_documents(tokenizer),
                seq_length=self.seq_length,
                drop_remainder=self.drop_remainder,
                stats=stats,
            ):
                yield example
        else:
            for document in self._iter_tokenized_documents(tokenizer):
                stats.conversations += 1
                stats.tokens_before_packing += len(document.input_ids)
                stats.tokens_after_packing += len(document.input_ids)
                stats.packed_sequences += 1
                yield {
                    "input_ids": document.input_ids,
                    "attention_mask": [1] * len(document.input_ids),
                    "labels": document.labels,
                }

        self.summary.update(stats.to_dict())


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    model_cfg = read_yaml(args.model_config)
    tokenization_cfg = manifest.get("tokenization", {})
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_root / ".hf_cache"
    summary_path = output_root / "summary.json"

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_name_or_path = resolve_tokenizer_name_or_path(args, manifest)
    seq_length = args.seq_length or int(tokenization_cfg.get("sft_seq_len", 2048))
    packing = bool(tokenization_cfg.get("packing", True))
    drop_remainder = args.drop_remainder or packing
    train_on_prompt = args.train_on_prompt or bool(model_cfg.get("train_on_prompt", False))

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "labels": Sequence(Value("int32")),
        }
    )

    split_summaries: Dict[str, Any] = {}
    for split in ("train", "val"):
        input_path = input_root / f"{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing cleaned SFT split at {input_path}")
        output_path = output_root / split
        builder = PackedSFTDatasetBuilder(
            input_path=input_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            seq_length=seq_length,
            packing=packing,
            train_on_prompt=train_on_prompt,
            drop_remainder=drop_remainder,
        )
        dataset = Dataset.from_generator(
            builder.generate,
            features=features,
            cache_dir=str(cache_dir),
        )
        dataset.save_to_disk(str(output_path))
        split_summaries[split] = {
            **builder.summary,
            "num_rows": dataset.num_rows,
            "output_path": str(output_path),
        }

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "cache_dir": str(cache_dir),
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "splits": split_summaries,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] tokenized SFT dataset saved to {output_root}")
    print(f"[DONE] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
