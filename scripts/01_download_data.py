#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import jsonlines
from datasets import load_dataset
from tqdm import tqdm


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_jsonl(records: Iterable[Dict[str, Any]], out_path: str, max_count: Optional[int] = None) -> int:
    ensure_dir(str(Path(out_path).parent))
    count = 0
    with jsonlines.open(out_path, mode="w") as writer:
        for item in records:
            writer.write(item)
            count += 1
            if max_count is not None and count >= max_count:
                break
    return count


def normalize_text_record(example: Dict[str, Any], text_field: str) -> Optional[Dict[str, Any]]:
    text = example.get(text_field, None)
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return None
    return {"text": text}


def normalize_ultrachat_record(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    msgs = example.get("messages", None)
    if not msgs or not isinstance(msgs, list):
        return None
    return {"messages": msgs}


def normalize_wildchat_record(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "conversation" in example and isinstance(example["conversation"], list):
        return {"messages": example["conversation"]}
    if "messages" in example and isinstance(example["messages"], list):
        return {"messages": example["messages"]}
    return None


def normalize_gsm8k_record(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = example.get("question")
    a = example.get("answer")
    if not q or not a:
        return None
    return {"question": q, "answer": a}


def normalize_generic_record(example: Dict[str, Any]) -> Dict[str, Any]:
    return dict(example)


def iter_streaming_dataset(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    text_field: Optional[str] = None,
    target_samples: Optional[int] = None,
    seed: int = 42,
):
    if subset and subset != "default":
        ds = load_dataset(dataset_name, subset, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=True)

    seen = 0
    for ex in ds:
        if text_field:
            item = normalize_text_record(ex, text_field)
        else:
            item = normalize_generic_record(ex)
        if item is None:
            continue

        yield item
        seen += 1
        if target_samples is not None and seen >= target_samples:
            break


def load_nonstreaming_dataset(
    dataset_name: str,
    subset: Optional[str],
    split: str,
):
    if subset and subset != "default":
        return load_dataset(dataset_name, subset, split=split)
    return load_dataset(dataset_name, split=split)


def sample_indices(n: int, k: int, seed: int) -> List[int]:
    k = min(n, k)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return sorted(idx[:k])


def process_cpt_entry(entry: Dict[str, Any], raw_dir: Path, seed: int) -> None:
    name = entry["name"]
    source = entry["source"]
    subset = entry.get("subset", None)
    split = entry.get("split", "train")
    streaming = entry.get("streaming", False)
    text_field = entry.get("text_field", "text")
    max_samples = entry.get("max_samples", None)

    out_path = raw_dir / "cpt" / f"{name}.jsonl"
    ensure_dir(str(out_path.parent))

    print(f"[CPT] downloading {name} -> {out_path}")

    if streaming:
        iterator = iter_streaming_dataset(
            dataset_name=source,
            subset=subset,
            split=split,
            text_field=text_field,
            target_samples=max_samples,
            seed=seed,
        )
        count = save_jsonl(iterator, str(out_path), max_count=max_samples)
    else:
        ds = load_nonstreaming_dataset(source, subset, split)
        indices = sample_indices(len(ds), max_samples or len(ds), seed)
        with jsonlines.open(out_path, mode="w") as writer:
            count = 0
            for i in tqdm(indices, desc=f"[CPT:{name}]"):
                item = normalize_text_record(ds[i], text_field)
                if item is None:
                    continue
                writer.write(item)
                count += 1

    print(f"[CPT] saved {count} rows for {name}")


def process_sft_entry(entry: Dict[str, Any], raw_dir: Path, seed: int) -> None:
    name = entry["name"]
    source = entry["source"]
    subset = entry.get("subset", None)
    split = entry.get("split", "train")
    target_samples = entry.get("target_samples", None)

    out_path = raw_dir / "sft" / f"{name}.jsonl"
    ensure_dir(str(out_path.parent))

    print(f"[SFT] downloading {name} -> {out_path}")

    ds = load_nonstreaming_dataset(source, subset, split)
    indices = sample_indices(len(ds), target_samples or len(ds), seed)

    with jsonlines.open(out_path, mode="w") as writer:
        count = 0
        for i in tqdm(indices, desc=f"[SFT:{name}]"):
            ex = ds[i]
            if "ultrachat" in name.lower():
                item = normalize_ultrachat_record(ex)
            elif "wildchat" in name.lower():
                item = normalize_wildchat_record(ex)
            else:
                item = normalize_generic_record(ex)

            if item is None:
                continue
            writer.write(item)
            count += 1

    print(f"[SFT] saved {count} rows for {name}")


def process_eval_entry(entry: Dict[str, Any], raw_dir: Path) -> None:
    name = entry["name"]
    source = entry["source"]
    subset = entry.get("subset", None)
    split = entry.get("split", "test")

    out_path = raw_dir / "eval" / f"{name}.jsonl"
    ensure_dir(str(out_path.parent))

    print(f"[EVAL] downloading {name} -> {out_path}")

    if name == "mmlu_mini":
        subjects = entry["subjects"]
        total = 0
        with jsonlines.open(out_path, mode="w") as writer:
            for subject in subjects:
                ds = load_dataset(source, subject, split=split)
                for ex in ds:
                    ex = dict(ex)
                    ex["_subject"] = subject
                    writer.write(ex)
                    total += 1
        print(f"[EVAL] saved {total} rows for {name}")
        return

    ds = load_nonstreaming_dataset(source, subset, split)
    with jsonlines.open(out_path, mode="w") as writer:
        total = 0
        for ex in ds:
            if name == "gsm8k":
                item = normalize_gsm8k_record(ex)
            else:
                item = normalize_generic_record(ex)
            if item is None:
                continue
            writer.write(item)
            total += 1

    print(f"[EVAL] saved {total} rows for {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="configs/dataset_manifest.json")
    parser.add_argument("--output_root", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    raw_dir = Path(args.output_root)
    ensure_dir(str(raw_dir))

    os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

    for entry in manifest["datasets"].get("cpt", []):
        try:
            process_cpt_entry(entry, raw_dir, args.seed)
        except Exception as e:
            print(f"[WARN] failed on CPT dataset {entry.get('name')}: {e}")

    for entry in manifest["datasets"].get("sft", []):
        try:
            process_sft_entry(entry, raw_dir, args.seed)
        except Exception as e:
            print(f"[WARN] failed on SFT dataset {entry.get('name')}: {e}")

    for entry in manifest["datasets"].get("eval", []):
        try:
            process_eval_entry(entry, raw_dir)
        except Exception as e:
            print(f"[WARN] failed on EVAL dataset {entry.get('name')}: {e}")

    print("all requested datasets processed.")


if __name__ == "__main__":
    main()
