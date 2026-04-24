#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import jsonlines
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.packers import PackConfig, PackingStats, TokenizedDocument, iter_packed_examples


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize and pack cleaned CPT corpus.")
    parser.add_argument("--manifest", type=str, default="configs/dataset_manifest.json")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--input_root", type=str, default="data/cleaned/cpt")
    parser.add_argument("--output_path", type=str, default="data/tokenized/cpt/train")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--seq_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_documents", type=int, default=None)
    parser.add_argument("--drop_remainder", action="store_true")
    parser.add_argument("--strict_missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_cleaned_paths(
    manifest: Dict[str, Any],
    input_root: Path,
    strict_missing: bool,
) -> tuple[List[str], List[str]]:
    input_paths: List[str] = []
    missing: List[str] = []

    for entry in manifest["datasets"].get("cpt", []):
        path = input_root / f"{entry['name']}.jsonl"
        if path.exists():
            input_paths.append(str(path))
        else:
            missing.append(str(path))

    if strict_missing and missing:
        raise FileNotFoundError(f"Missing cleaned CPT files: {', '.join(missing)}")
    if not input_paths:
        raise FileNotFoundError(f"No cleaned CPT files were found under {input_root}")
    return input_paths, missing


def resolve_tokenizer_name_or_path(args: argparse.Namespace, manifest: Dict[str, Any]) -> str:
    if args.tokenizer_name_or_path:
        return args.tokenizer_name_or_path
    if args.model_config:
        model_cfg = read_yaml(args.model_config)
        return model_cfg["tokenizer_name_or_path"]
    if manifest.get("tokenizer"):
        return manifest["tokenizer"]
    raise ValueError("Tokenizer path is required via --tokenizer_name_or_path or --model_config.")


class PackedCPTDatasetBuilder:
    def __init__(
        self,
        input_paths: List[str],
        tokenizer_name_or_path: str,
        seq_length: int,
        batch_size: int,
        add_eos: bool,
        drop_remainder: bool,
        max_documents: Optional[int] = None,
    ) -> None:
        self.input_paths = input_paths
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.add_eos = add_eos
        self.drop_remainder = drop_remainder
        self.max_documents = max_documents
        self.summary: Dict[str, Any] = {}

    def _reset_summary(self) -> None:
        self.summary = {
            "documents": 0,
            "source_stats": defaultdict(lambda: {"documents": 0, "tokens_before_packing": 0}),
            "language_stats": defaultdict(lambda: {"documents": 0, "tokens_before_packing": 0}),
        }

    def _iter_cleaned_batches(self) -> Iterable[List[Dict[str, str]]]:
        batch: List[Dict[str, str]] = []
        seen_documents = 0
        for path in self.input_paths:
            default_source = Path(path).stem
            with jsonlines.open(path, "r") as reader:
                for row in reader:
                    text = row.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    batch.append(
                        {
                            "text": text,
                            "source": str(row.get("source", default_source)),
                            "language": str(row.get("language", "unknown")),
                        }
                    )
                    seen_documents += 1
                    if self.max_documents is not None and seen_documents >= self.max_documents:
                        if batch:
                            yield batch
                        return
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []

        if batch:
            yield batch

    def _iter_tokenized_documents(self, tokenizer: AutoTokenizer) -> Iterable[TokenizedDocument]:
        for batch in self._iter_cleaned_batches():
            encoded = tokenizer(
                [item["text"] for item in batch],
                add_special_tokens=False,
                truncation=False,
            )
            for item, token_ids in zip(batch, encoded["input_ids"]):
                source = item["source"]
                language = item["language"]
                token_count = len(token_ids) + int(self.add_eos and tokenizer.eos_token_id is not None)
                self.summary["documents"] += 1
                self.summary["source_stats"][source]["documents"] += 1
                self.summary["source_stats"][source]["tokens_before_packing"] += token_count
                self.summary["language_stats"][language]["documents"] += 1
                self.summary["language_stats"][language]["tokens_before_packing"] += token_count
                yield TokenizedDocument(
                    input_ids=list(token_ids),
                    source=source,
                    language=language,
                )

    def generate(self) -> Iterable[Dict[str, List[int]]]:
        self._reset_summary()
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            use_fast=True,
        )
        pack_config = PackConfig(
            seq_length=self.seq_length,
            eos_token_id=tokenizer.eos_token_id,
            add_eos=self.add_eos,
            drop_remainder=self.drop_remainder,
        )
        packing_stats = PackingStats()
        for example in iter_packed_examples(
            documents=self._iter_tokenized_documents(tokenizer),
            config=pack_config,
            stats=packing_stats,
        ):
            yield example

        self.summary.update(packing_stats.to_dict())
        self.summary["source_stats"] = dict(self.summary["source_stats"])
        self.summary["language_stats"] = dict(self.summary["language_stats"])
        self.summary["tokenizer_name_or_path"] = self.tokenizer_name_or_path
        self.summary["seq_length"] = self.seq_length
        self.summary["add_eos"] = self.add_eos
        self.summary["drop_remainder"] = self.drop_remainder


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    tokenization_cfg = manifest.get("tokenization", {})
    input_root = Path(args.input_root)
    output_path = Path(args.output_path)
    summary_path = output_path.parent / "summary.json"
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_path.parent / ".hf_cache"

    if args.overwrite and output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    input_paths, missing = resolve_cleaned_paths(
        manifest=manifest,
        input_root=input_root,
        strict_missing=args.strict_missing,
    )
    tokenizer_name_or_path = resolve_tokenizer_name_or_path(args, manifest)
    seq_length = args.seq_length or int(tokenization_cfg.get("cpt_seq_len", 2048))
    add_eos = bool(tokenization_cfg.get("add_eos", True))
    drop_remainder = args.drop_remainder or bool(tokenization_cfg.get("packing", True))

    builder = PackedCPTDatasetBuilder(
        input_paths=input_paths,
        tokenizer_name_or_path=tokenizer_name_or_path,
        seq_length=seq_length,
        batch_size=args.batch_size,
        add_eos=add_eos,
        drop_remainder=drop_remainder,
        max_documents=args.max_documents,
    )
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
        }
    )
    dataset = Dataset.from_generator(
        builder.generate,
        features=features,
        cache_dir=str(cache_dir),
    )
    dataset.save_to_disk(str(output_path))

    summary = {
        "input_root": str(input_root),
        "input_paths": input_paths,
        "missing_inputs": missing,
        "output_path": str(output_path),
        "cache_dir": str(cache_dir),
        "num_rows": dataset.num_rows,
        **builder.summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] tokenized CPT dataset saved to {output_path}")
    print(f"[DONE] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
