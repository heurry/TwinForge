#!/usr/bin/env python3
"""Build deterministic 1K/2K customer-support benchmark prompts from DianJin CSC."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as parquet
from transformers import AutoTokenizer


DATASET_ID = "DianJin/DianJin-CSC-Data"
DEFAULT_MODEL = "/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-FP8"
SYSTEM_PROMPT = (
    "你是中文客服助手。请依据给定的历史服务案例回答最后一位客户，先确认诉求，"
    "再给出可执行步骤、预计时效和必要的风险提示；回复控制在120个汉字以内，"
    "不得索取密码或验证码，无法确认时建议转人工客服。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--raw-path", default="data/raw/dianjin_csc_train.parquet")
    parser.add_argument("--output", default="data/cleaned/dianjin_csc_benchmark.jsonl")
    parser.add_argument("--manifest", default="data/manifests/dianjin_csc_benchmark.json")
    parser.add_argument("--context-lengths", default="1024,2048")
    parser.add_argument("--samples-per-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def fetch_parquet_url(dataset_id: str) -> str:
    api = f"https://huggingface.co/api/datasets/{dataset_id}/parquet/default/train"
    with urllib.request.urlopen(api, timeout=30) as response:
        urls = json.load(response)
    if not urls:
        raise RuntimeError(f"dataset API returned no parquet files: {api}")
    return str(urls[0])


def download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


def render_dialogue(row: dict[str, Any]) -> str:
    source = clean_text(row.get("source"))
    lines = [f"业务分类：{source}" if source else "业务分类：中文客服"]
    for turn in row.get("dialogue") or []:
        speaker = clean_text(turn.get("speaker")) or "未知角色"
        strategy = clean_text(turn.get("strategy"))
        text = clean_text(turn.get("text"))
        if not text:
            continue
        strategy_text = f"[{strategy}]" if strategy else ""
        lines.append(f"{speaker}{strategy_text}：{text}")
    return "\n".join(lines)


def customer_exchange(row: dict[str, Any]) -> tuple[str, str]:
    turns = row.get("dialogue") or []
    for index, turn in enumerate(turns):
        question = clean_text(turn.get("text"))
        if clean_text(turn.get("speaker")) != "客户" or not question:
            continue
        for answer_turn in turns[index + 1 :]:
            answer = clean_text(answer_turn.get("text"))
            if clean_text(answer_turn.get("speaker")) == "客服" and answer:
                return question, answer
    return (
        "请说明这项业务的办理步骤、预计时效，以及遇到异常时应如何处理。",
        "我会先核实您的具体诉求，再说明办理步骤和预计时效；如系统状态异常，将为您转接人工客服并创建工单。",
    )


def build_shared_prefix(
    rendered: list[str], tokenizer: Any, target_tokens: int, start: int
) -> tuple[str, int]:
    header = SYSTEM_PROMPT + "\n\n以下是脱敏后的历史服务案例：\n"
    chunks = [header]
    index = start
    desired = max(target_tokens - 96, 128)
    while True:
        chunks.append(f"\n--- 案例 {index + 1} ---\n{rendered[index % len(rendered)]}\n")
        token_ids = tokenizer.encode("".join(chunks), add_special_tokens=False)
        if len(token_ids) >= desired:
            clipped = token_ids[:desired]
            return tokenizer.decode(clipped, skip_special_tokens=True), len(clipped)
        index += 1


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    context_lengths = [int(item) for item in args.context_lengths.split(",") if int(item) > 0]
    if not context_lengths or args.samples_per_length <= 0:
        raise SystemExit("context lengths and samples per length must be positive")

    raw_path = Path(args.raw_path)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    parquet_url = fetch_parquet_url(args.dataset_id)
    download(parquet_url, raw_path, args.force_download)

    table = parquet.read_table(raw_path, columns=["dialogue", "source"])
    source_rows = table.to_pylist()
    usable = [row for row in source_rows if row.get("dialogue")]
    if not usable:
        raise RuntimeError("DianJin CSC parquet contains no usable dialogues")

    rng = random.Random(args.seed)
    rng.shuffle(usable)
    rendered = [render_dialogue(row) for row in usable]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    benchmark_rows: list[dict[str, Any]] = []
    for context_length in context_lengths:
        # Keep shared-prefix source rows disjoint from held-out customer questions.
        shared_prefix, prefix_tokens = build_shared_prefix(
            rendered, tokenizer, context_length, len(usable) // 2
        )
        prefix_id = f"dianjin-csc-shared-{context_length}-v1"
        for index in range(args.samples_per_length):
            source_row = usable[index % len(usable)]
            question, reference_answer = customer_exchange(source_row)
            prompt = (
                shared_prefix
                + "\n\n当前客户问题："
                + question
                + "\n请直接给出客服回复。"
            )
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            benchmark_rows.append(
                {
                    "id": f"dianjin-csc-{context_length}-{index + 1:04d}",
                    "dataset": args.dataset_id,
                    "scenario": "customer_support_shared_prefix",
                    "context_length": context_length,
                    "prompt_tokens": prompt_tokens,
                    "prompt": prompt,
                    "reference_answer": reference_answer,
                    "expected_keywords": [],
                    "shared_prefix_id": prefix_id,
                    "shared_prefix_tokens": prefix_tokens,
                    "source": clean_text(source_row.get("source")),
                }
            )

    written = write_jsonl(output_path, benchmark_rows)
    manifest = {
        "dataset": args.dataset_id,
        "license": "MIT",
        "source_url": f"https://huggingface.co/datasets/{args.dataset_id}",
        "parquet_url": parquet_url,
        "raw_path": str(raw_path),
        "raw_sha256": sha256(raw_path),
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
        "source_rows": len(source_rows),
        "usable_rows": len(usable),
        "benchmark_rows": written,
        "context_lengths": context_lengths,
        "samples_per_length": args.samples_per_length,
        "response_character_limit": 120,
        "seed": args.seed,
        "tokenizer_path": args.model_path,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
