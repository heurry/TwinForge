#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import jsonlines
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
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


def normalize_sft_record(dataset_name: str, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lowered = dataset_name.lower()
    if "ultrachat" in lowered:
        return normalize_ultrachat_record(example)
    if "wildchat" in lowered:
        return normalize_wildchat_record(example)
    return normalize_generic_record(example)


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


def load_streaming_dataset(
    dataset_name: str,
    subset: Optional[str],
    split: str,
):
    if subset and subset != "default":
        return load_dataset(dataset_name, subset, split=split, streaming=True)
    return load_dataset(dataset_name, split=split, streaming=True)


def sample_indices(n: int, k: int, seed: int) -> List[int]:
    k = min(n, k)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return sorted(idx[:k])


def iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_hf_repo_text_files(
    repo_id: str,
    paths: List[str],
    text_field: str,
    revision: str = "main",
    seed: int = 42,
) -> Iterable[Dict[str, Any]]:
    ordered_paths = list(paths)
    random.Random(seed).shuffle(ordered_paths)

    for rel_path in ordered_paths:
        print(f"[CPT] downloading file {repo_id}/{rel_path}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=rel_path,
            repo_type="dataset",
            revision=revision,
        )
        for ex in iter_jsonl_records(Path(local_path)):
            item = normalize_text_record(ex, text_field)
            if item is not None:
                yield item


def iter_hf_repo_records(
    repo_id: str,
    paths: List[str],
    revision: str = "main",
    seed: int = 42,
) -> Iterable[Dict[str, Any]]:
    ordered_paths = list(paths)
    random.Random(seed).shuffle(ordered_paths)

    for rel_path in ordered_paths:
        print(f"[DATA] downloading file {repo_id}/{rel_path}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=rel_path,
            repo_type="dataset",
            revision=revision,
        )
        yield from iter_jsonl_records(Path(local_path))


def resolve_hf_repo_paths(entry: Dict[str, Any]) -> List[str]:
    if entry.get("paths"):
        return list(entry["paths"])

    prefix = entry.get("path_prefix", "")
    suffix = entry.get("path_suffix", "")
    revision = entry.get("revision", "main")

    api = HfApi()
    repo_files = api.list_repo_files(entry["source"], repo_type="dataset", revision=revision)
    matched = [p for p in repo_files if p.startswith(prefix) and p.endswith(suffix)]

    if not matched:
        raise FileNotFoundError(
            f"No dataset files matched prefix={prefix!r}, suffix={suffix!r} in {entry['source']}"
        )

    return matched


def process_cpt_entry(entry: Dict[str, Any], raw_dir: Path, seed: int, skip_existing: bool) -> None:
    name = entry["name"]
    source = entry["source"]
    source_type = entry.get("source_type", "hf_dataset")
    subset = entry.get("subset", None)
    split = entry.get("split", "train")
    streaming = entry.get("streaming", False)
    text_field = entry.get("text_field", "text")
    max_samples = entry.get("max_samples", None)

    out_path = raw_dir / "cpt" / f"{name}.jsonl"
    ensure_dir(str(out_path.parent))

    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[CPT] skip existing {name} -> {out_path}")
        return

    print(f"[CPT] downloading {name} -> {out_path}")

    if source_type == "hf_repo_files":
        iterator = iter_hf_repo_text_files(
            repo_id=source,
            paths=resolve_hf_repo_paths(entry),
            text_field=text_field,
            revision=entry.get("revision", "main"),
            seed=seed,
        )
        count = save_jsonl(iterator, str(out_path), max_count=max_samples)
        print(f"[CPT] saved {count} rows for {name}")
        return

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


def process_sft_entry(entry: Dict[str, Any], raw_dir: Path, seed: int, skip_existing: bool) -> None:
    name = entry["name"]
    source = entry["source"]
    subset = entry.get("subset", None)
    split = entry.get("split", "train")
    target_samples = entry.get("target_samples", None)
    streaming = entry.get("streaming", False)
    shuffle_buffer_size = int(entry.get("shuffle_buffer_size", min(max(target_samples or 10000, 1000), 50000)))

    out_path = raw_dir / "sft" / f"{name}.jsonl"
    ensure_dir(str(out_path.parent))

    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SFT] skip existing {name} -> {out_path}")
        return

    print(f"[SFT] downloading {name} -> {out_path}")

    if streaming:
        ds = load_streaming_dataset(source, subset, split)
        if target_samples is not None:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

        with jsonlines.open(out_path, mode="w") as writer:
            count = 0
            for ex in ds:
                item = normalize_sft_record(name, ex)
                if item is None:
                    continue
                writer.write(item)
                count += 1
                if target_samples is not None and count >= target_samples:
                    break

        print(f"[SFT] saved {count} rows for {name}")
        return

    ds = load_nonstreaming_dataset(source, subset, split)
    indices = sample_indices(len(ds), target_samples or len(ds), seed)

    with jsonlines.open(out_path, mode="w") as writer:
        count = 0
        for i in tqdm(indices, desc=f"[SFT:{name}]"):
            ex = ds[i]
            item = normalize_sft_record(name, ex)

            if item is None:
                continue
            writer.write(item)
            count += 1

    print(f"[SFT] saved {count} rows for {name}")


def process_eval_entry(entry: Dict[str, Any], raw_dir: Path, skip_existing: bool) -> None:
    name = entry["name"]
    source = entry["source"]
    source_type = entry.get("source_type", "hf_dataset")
    subset = entry.get("subset", None)
    split = entry.get("split", "test")

    out_path = raw_dir / "eval" / f"{name}.jsonl"
    ensure_dir(str(out_path.parent))

    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[EVAL] skip existing {name} -> {out_path}")
        return

    print(f"[EVAL] downloading {name} -> {out_path}")

    if source_type == "hf_repo_files":
        iterator = iter_hf_repo_records(
            repo_id=source,
            paths=resolve_hf_repo_paths(entry),
            revision=entry.get("revision", "main"),
        )
        count = save_jsonl(iterator, str(out_path))
        print(f"[EVAL] saved {count} rows for {name}")
        return

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
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    raw_dir = Path(args.output_root)
    ensure_dir(str(raw_dir))

    os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

    for entry in manifest["datasets"].get("cpt", []):
        try:
            process_cpt_entry(entry, raw_dir, args.seed, args.skip_existing)
        except Exception as e:
            print(f"[WARN] failed on CPT dataset {entry.get('name')}: {e}")

    for entry in manifest["datasets"].get("sft", []):
        try:
            process_sft_entry(entry, raw_dir, args.seed, args.skip_existing)
        except Exception as e:
            print(f"[WARN] failed on SFT dataset {entry.get('name')}: {e}")

    for entry in manifest["datasets"].get("eval", []):
        try:
            process_eval_entry(entry, raw_dir, args.skip_existing)
        except Exception as e:
            print(f"[WARN] failed on EVAL dataset {entry.get('name')}: {e}")

    print("all requested datasets processed.")


if __name__ == "__main__":
    main()
