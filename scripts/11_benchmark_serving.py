#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Dict, List

import requests
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serve.openai_client import list_models, normalize_base_url, stream_chat_completion


DEFAULT_PROMPTS = [
    "用一句话介绍 continued pretraining 和 SFT 的区别。",
    "请解释为什么 ZeRO-2 能降低优化器状态显存占用。",
    "给出一个最小 LoRA 训练闭环需要经过的步骤。",
]


def extract_stream_content(event: Dict[str, Any]) -> str:
    choices = event.get("choices")
    if not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    delta = first_choice.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal serving benchmark against a vLLM endpoint.")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-1.7b-miniv2",
        help="OpenAI model id exposed by the server, not the local filesystem path.",
    )
    parser.add_argument("--tokenizer_path", type=str, default="model/Qwen3-1.7B")
    parser.add_argument("--output_json", type=str, default="runs/serve/vllm_minimal_benchmark.json")
    parser.add_argument("--output_report", type=str, default="reports/serving_benchmark.md")
    parser.add_argument("--max_tokens", type=int, default=256)
    return parser.parse_args()


def run_single_request(base_url: str, model: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    started_at = perf_counter()
    first_token_latency = None
    chunks: List[str] = []

    for event in stream_chat_completion(
        base_url=base_url,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    ):
        delta = extract_stream_content(event)
        if delta:
            chunks.append(delta)
            if first_token_latency is None:
                first_token_latency = perf_counter() - started_at

    end_to_end_latency = perf_counter() - started_at
    return {
        "prompt": prompt,
        "response_text": "".join(chunks).strip(),
        "ttft_seconds": first_token_latency,
        "end_to_end_latency_seconds": end_to_end_latency,
    }


def render_markdown(results: List[Dict[str, Any]], model: str) -> str:
    valid_decode_tps = [item["decode_tokens_per_second"] for item in results if item["decode_tokens_per_second"] is not None]
    valid_ttft = [item["ttft_seconds"] for item in results if item["ttft_seconds"] is not None]
    valid_latency = [item["end_to_end_latency_seconds"] for item in results if item["end_to_end_latency_seconds"] is not None]
    lines = [
        "# Serving Benchmark",
        "",
        f"最小可运行 serving benchmark，当前后端：`{model}`。",
        "",
        "| Request | TTFT (s) | End-to-End (s) | Output Tokens | Decode Tokens/s |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for idx, item in enumerate(results, start=1):
        lines.append(
            "| {idx} | {ttft:.3f} | {latency:.3f} | {tokens} | {decode_tps:.3f} |".format(
                idx=idx,
                ttft=item["ttft_seconds"] or 0.0,
                latency=item["end_to_end_latency_seconds"],
                tokens=item["output_tokens"],
                decode_tps=item["decode_tokens_per_second"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- mean TTFT: `{mean(valid_ttft):.3f}s`" if valid_ttft else "- mean TTFT: `-`",
            f"- mean end-to-end latency: `{mean(valid_latency):.3f}s`" if valid_latency else "- mean end-to-end latency: `-`",
            f"- mean decode tokens/s: `{mean(valid_decode_tps):.3f}`" if valid_decode_tps else "- mean decode tokens/s: `-`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def format_request_error(exc: requests.RequestException, base_url: str, model: str) -> str:
    endpoint = normalize_base_url(base_url) + "/chat/completions"
    lines = [f"[ERROR] request to {endpoint} failed: {exc}"]

    if isinstance(exc, requests.ConnectionError):
        lines.append("[ERROR] Serving endpoint is unreachable. Confirm `scripts/10_serve_vllm.sh` is still running on this port.")
        return "\n".join(lines)

    response = getattr(exc, "response", None)
    if response is None:
        return "\n".join(lines)

    if response.status_code == 404:
        try:
            available_models = list_models(base_url=base_url)
        except requests.RequestException as list_exc:
            lines.append(f"[ERROR] Failed to query {normalize_base_url(base_url)}/models: {list_exc}")
            return "\n".join(lines)

        if available_models:
            rendered_models = ", ".join(f"`{item}`" for item in available_models)
            lines.append(f"[ERROR] Available model ids: {rendered_models}")
            if model not in available_models:
                lines.append(
                    f"[ERROR] Requested `--model {model}` is not exposed by the server. "
                    f"Use one of the ids above, or restart vLLM with `--served-model-name {model}`."
                )
        else:
            lines.append(f"[ERROR] {normalize_base_url(base_url)}/models returned no model ids.")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    results = []
    try:
        for prompt in DEFAULT_PROMPTS:
            result = run_single_request(
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )
            output_tokens = len(tokenizer.encode(result["response_text"], add_special_tokens=False))
            decode_seconds = None
            if result["ttft_seconds"] is not None:
                decode_seconds = max(result["end_to_end_latency_seconds"] - result["ttft_seconds"], 1e-6)
            result["output_tokens"] = output_tokens
            result["decode_tokens_per_second"] = output_tokens / decode_seconds if decode_seconds is not None else None
            results.append(result)
    except requests.RequestException as exc:
        raise SystemExit(format_request_error(exc, args.base_url, args.model)) from exc

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "backend": "vllm",
                "base_url": args.base_url,
                "model": args.model,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(results, args.model), encoding="utf-8")
    print(f"[DONE] wrote serving benchmark json to {output_json}")
    print(f"[DONE] wrote serving benchmark report to {report_path}")


if __name__ == "__main__":
    main()
