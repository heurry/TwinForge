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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaners import CleanTextConfig, clean_text, stable_text_hash
from src.data.samplers import SourceSpec, build_sampling_plan, parse_mapping_arg


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned CPT corpus from raw jsonl files.")
    parser.add_argument("--manifest", type=str, default="configs/dataset_manifest.json")
    parser.add_argument("--input_root", type=str, default="data/raw/cpt")
    parser.add_argument("--output_root", type=str, default="data/cleaned/cpt")
    parser.add_argument("--min_chars", type=int, default=None)
    parser.add_argument("--max_chars", type=int, default=None)
    parser.add_argument("--max_total_samples", type=int, default=None)
    parser.add_argument("--language_ratios", type=str, default=None)
    parser.add_argument("--language_max_samples", type=str, default=None)
    parser.add_argument(
        "--dedup_scope",
        type=str,
        choices=["off", "source", "global"],
        default="global",
    )
    parser.add_argument("--strict_missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_clean_config(manifest: Dict[str, Any], args: argparse.Namespace) -> CleanTextConfig:
    preprocess_cfg = manifest.get("preprocess", {})
    return CleanTextConfig(
        min_chars=args.min_chars if args.min_chars is not None else preprocess_cfg.get("min_text_chars", 50),
        max_chars=args.max_chars if args.max_chars is not None else preprocess_cfg.get("max_text_chars", 12000),
        drop_empty=preprocess_cfg.get("drop_empty", True),
        normalize_whitespace=preprocess_cfg.get("normalize_whitespace", True),
    )


def resolve_available_sources(
    manifest: Dict[str, Any],
    input_root: Path,
    strict_missing: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    available_entries: List[Dict[str, Any]] = []
    missing_sources: List[Dict[str, str]] = []

    for entry in manifest["datasets"].get("cpt", []):
        input_path = input_root / f"{entry['name']}.jsonl"
        if input_path.exists():
            enriched = dict(entry)
            enriched["input_path"] = str(input_path)
            available_entries.append(enriched)
            continue

        missing_sources.append(
            {
                "name": entry["name"],
                "language": entry.get("language", "unknown"),
                "input_path": str(input_path),
            }
        )

    if strict_missing and missing_sources:
        missing_names = ", ".join(item["name"] for item in missing_sources)
        raise FileNotFoundError(f"Missing required CPT raw files: {missing_names}")
    if not available_entries:
        raise FileNotFoundError(f"No CPT raw files were found under {input_root}")

    return available_entries, missing_sources


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with jsonlines.open(path, "r") as reader:
        for row in reader:
            yield row


def initialize_source_stats(entry: Dict[str, Any], quota: int, output_path: Path) -> Dict[str, Any]:
    return {
        "source": entry["name"],
        "language": entry.get("language", "unknown"),
        "input_path": entry["input_path"],
        "output_path": str(output_path),
        "target_samples": quota,
        "raw_records_seen": 0,
        "written_records": 0,
        "input_chars": 0,
        "output_chars": 0,
        "dropped_missing_text": 0,
        "dropped_empty": 0,
        "dropped_too_short": 0,
        "dropped_too_long": 0,
        "dropped_duplicate": 0,
        "quota_reached": False,
    }


def apply_drop_reason(stats: Dict[str, Any], reason: Optional[str]) -> None:
    if reason == "missing_text":
        stats["dropped_missing_text"] += 1
    elif reason == "empty":
        stats["dropped_empty"] += 1
    elif reason == "too_short":
        stats["dropped_too_short"] += 1
    elif reason == "too_long":
        stats["dropped_too_long"] += 1


def process_source(
    entry: Dict[str, Any],
    quota: int,
    clean_config: CleanTextConfig,
    output_root: Path,
    dedup_scope: str,
    global_hashes: set[str],
) -> Dict[str, Any]:
    input_path = Path(entry["input_path"])
    output_path = output_root / f"{entry['name']}.jsonl"
    stats = initialize_source_stats(entry, quota, output_path)
    local_hashes: set[str] = set()

    ensure_dir(output_path.parent)
    with jsonlines.open(output_path, "w") as writer:
        for row in iter_jsonl(input_path):
            if stats["written_records"] >= quota:
                stats["quota_reached"] = True
                break

            stats["raw_records_seen"] += 1
            raw_text = row.get("text")
            raw_text_str = "" if raw_text is None else str(raw_text)
            stats["input_chars"] += len(raw_text_str)

            cleaned_text, reason = clean_text(
                raw_text,
                config=clean_config,
                language=entry.get("language"),
            )
            if cleaned_text is None:
                apply_drop_reason(stats, reason)
                continue

            text_hash = stable_text_hash(cleaned_text)
            if dedup_scope == "global":
                if text_hash in global_hashes:
                    stats["dropped_duplicate"] += 1
                    continue
                global_hashes.add(text_hash)
            elif dedup_scope == "source":
                if text_hash in local_hashes:
                    stats["dropped_duplicate"] += 1
                    continue
                local_hashes.add(text_hash)

            writer.write(
                {
                    "text": cleaned_text,
                    "source": entry["name"],
                    "language": entry.get("language", "unknown"),
                    "num_chars": len(cleaned_text),
                    "text_sha1": text_hash,
                }
            )
            stats["written_records"] += 1
            stats["output_chars"] += len(cleaned_text)

    return stats


def build_summary(
    manifest_path: str,
    input_root: Path,
    output_root: Path,
    clean_config: CleanTextConfig,
    sampling_plan: Dict[str, Any],
    source_stats: List[Dict[str, Any]],
    missing_sources: List[Dict[str, str]],
    dedup_scope: str,
) -> Dict[str, Any]:
    totals = defaultdict(int)
    language_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"written_records": 0, "output_chars": 0})
    for stats in source_stats:
        for key in [
            "raw_records_seen",
            "written_records",
            "input_chars",
            "output_chars",
            "dropped_missing_text",
            "dropped_empty",
            "dropped_too_short",
            "dropped_too_long",
            "dropped_duplicate",
        ]:
            totals[key] += int(stats[key])

        language = stats["language"]
        language_totals[language]["written_records"] += int(stats["written_records"])
        language_totals[language]["output_chars"] += int(stats["output_chars"])

    return {
        "manifest_path": manifest_path,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "clean_config": {
            "min_chars": clean_config.min_chars,
            "max_chars": clean_config.max_chars,
            "drop_empty": clean_config.drop_empty,
            "normalize_whitespace": clean_config.normalize_whitespace,
            "dedup_scope": dedup_scope,
        },
        "sampling_plan": sampling_plan,
        "totals": dict(totals),
        "language_totals": dict(language_totals),
        "sources": source_stats,
        "missing_sources": missing_sources,
    }


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    summary_path = output_root / "summary.json"

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root)

    available_entries, missing_sources = resolve_available_sources(
        manifest=manifest,
        input_root=input_root,
        strict_missing=args.strict_missing,
    )
    clean_config = build_clean_config(manifest, args)
    source_specs = [
        SourceSpec(
            name=entry["name"],
            language=entry.get("language", "unknown"),
            requested_max_samples=entry.get("max_samples"),
        )
        for entry in available_entries
    ]
    sampling_plan = build_sampling_plan(
        specs=source_specs,
        max_total_samples=args.max_total_samples,
        language_ratios=parse_mapping_arg(args.language_ratios, float),
        language_max_samples=parse_mapping_arg(args.language_max_samples, int),
    )

    print("[INFO] CPT cleaning plan:")
    print(json.dumps(sampling_plan.to_dict(), ensure_ascii=False, indent=2))

    global_hashes: set[str] = set()
    source_stats: List[Dict[str, Any]] = []
    for entry in available_entries:
        quota = sampling_plan.source_quotas.get(entry["name"], 0)
        if quota <= 0:
            print(f"[WARN] skip {entry['name']} because assigned quota is 0")
            continue

        stats = process_source(
            entry=entry,
            quota=quota,
            clean_config=clean_config,
            output_root=output_root,
            dedup_scope=args.dedup_scope,
            global_hashes=global_hashes,
        )
        source_stats.append(stats)
        print(
            f"[DONE] {entry['name']}: wrote {stats['written_records']} cleaned rows "
            f"(raw={stats['raw_records_seen']}, dup={stats['dropped_duplicate']}, "
            f"short={stats['dropped_too_short']}, long={stats['dropped_too_long']})"
        )

    summary = build_summary(
        manifest_path=args.manifest,
        input_root=input_root,
        output_root=output_root,
        clean_config=clean_config,
        sampling_plan=sampling_plan.to_dict(),
        source_stats=source_stats,
        missing_sources=missing_sources,
        dedup_scope=args.dedup_scope,
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] cleaned CPT corpus saved to {output_root}")
    print(f"[DONE] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
