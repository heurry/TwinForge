#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.prof.torch_profile import get_profile_events, is_communication_event, load_torch_profile_summary, summarize_event, top_events, total_primary_time_us


DEFAULT_RUN_DIRS = [
    "runs/cpt/profile/native/qwen3_1_7b/single",
    "runs/cpt/profile/native/qwen3_1_7b/ddp",
    "runs/cpt/profile/deepspeed/qwen3_1_7b/zero2",
    "runs/cpt/profile/deepspeed/qwen3_1_7b/zero3_offload",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Render a markdown summary from CPT torch/deepspeed profiling artifacts.")
    parser.add_argument("--run_dirs", nargs="*", default=DEFAULT_RUN_DIRS)
    parser.add_argument("--output", type=str, default="reports/cpt_profile.md")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value: Any, precision: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{precision}f}"


def format_ratio(numerator: Any, denominator: Any, precision: int = 2) -> str:
    if numerator in (None, 0) or denominator in (None, 0):
        return "-"
    return f"{float(numerator) / float(denominator):.{precision}f}x"


def format_percent_delta(new_value: Any, base_value: Any, precision: int = 1) -> str:
    if new_value is None or base_value in (None, 0):
        return "-"
    delta = (float(new_value) - float(base_value)) / float(base_value) * 100.0
    return f"{delta:+.{precision}f}%"


def guess_profile_dir(summary: dict[str, Any] | None, run_dir: Path) -> Path:
    if summary and summary.get("torch_profiler_dir"):
        return Path(summary["torch_profiler_dir"])
    return run_dir / "profiling" / "torch"


def build_row(run_dir: str) -> dict[str, Any]:
    run_path = Path(run_dir)
    summary = load_json(run_path / "train_summary.json")
    train_metrics = summary.get("train_metrics", {}) if summary else {}
    profile_dir = guess_profile_dir(summary, run_path)
    profile_summary = load_torch_profile_summary(profile_dir)
    events = get_profile_events(profile_summary)
    hotspots = top_events(events, limit=5, predicate=lambda event: not is_communication_event(event))
    total_time = total_primary_time_us(events)
    top_hotspot = summarize_event(hotspots[0], total_time_us=total_time) if hotspots else None
    flops_profile_path = run_path / "profiling" / "deepspeed" / "deepspeed_flops_profile.txt"

    return {
        "experiment": summary.get("experiment_name") if summary else run_path.name,
        "backend": summary.get("training_backend") if summary else run_path.name,
        "benchmark_group": summary.get("benchmark_group") if summary else None,
        "effective_batch_size": summary.get("effective_batch_size") if summary else None,
        "runtime": train_metrics.get("train_runtime"),
        "tokens_per_second": summary.get("train_tokens_per_second") if summary else None,
        "max_allocated_mb": extract_peak_memory(summary),
        "top_hotspot": top_hotspot,
        "torch_profile_dir": str(profile_dir),
        "torch_profile_available": bool(profile_summary),
        "deepspeed_flops_profile": str(flops_profile_path) if flops_profile_path.exists() else None,
        "hotspots": [summarize_event(event, total_time_us=total_time) for event in hotspots],
    }


def extract_peak_memory(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    output_dir = summary.get("output_dir")
    if not output_dir:
        return None
    history = load_json(Path(output_dir) / "train_log_history.json")
    if not history:
        return None
    values = [
        float(item["cuda_memory_max_allocated_mb"])
        for item in history
        if isinstance(item, dict) and "cuda_memory_max_allocated_mb" in item
    ]
    return max(values) if values else None


def find_row(rows: list[dict[str, Any]], backend: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("backend") == backend:
            return row
    return None


def build_analysis(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Analysis",
        "",
        "- 这些 profile 结果带 instrumentation 开销，绝对吞吐只用于解释瓶颈，不替代正式 benchmark 结论。",
    ]

    native_single = find_row(rows, "single_gpu")
    native_ddp = find_row(rows, "ddp")
    if native_single and native_ddp:
        single_hotspot = (native_single.get("top_hotspot") or {}).get("key", "-")
        ddp_hotspot = (native_ddp.get("top_hotspot") or {}).get("key", "-")
        lines.extend(
            [
                "- Native family：`ddp` 相比 `single` 的 profile 吞吐从 `{single_tokens}` 提升到 `{ddp_tokens}`，约 `{ratio}`；运行时从 `{single_runtime}s` 降到 `{ddp_runtime}s`，变化 `{runtime_delta}`；峰值显存从 `{single_mem} MB` 增到 `{ddp_mem} MB`，变化 `{mem_delta}`。".format(
                    single_tokens=format_float(native_single.get("tokens_per_second")),
                    ddp_tokens=format_float(native_ddp.get("tokens_per_second")),
                    ratio=format_ratio(native_ddp.get("tokens_per_second"), native_single.get("tokens_per_second")),
                    single_runtime=format_float(native_single.get("runtime"), 1),
                    ddp_runtime=format_float(native_ddp.get("runtime"), 1),
                    runtime_delta=format_percent_delta(native_ddp.get("runtime"), native_single.get("runtime")),
                    single_mem=format_float(native_single.get("max_allocated_mb"), 1),
                    ddp_mem=format_float(native_ddp.get("max_allocated_mb"), 1),
                    mem_delta=format_percent_delta(native_ddp.get("max_allocated_mb"), native_single.get("max_allocated_mb")),
                ),
                "- Native family：两条路的 top hotspot 都集中在 `{single_hotspot}` / `{ddp_hotspot}` 一类 launch 或同步事件，说明当前 native 路线更像 launch-bound / sync-bound，而不是显式通信主导。".format(
                    single_hotspot=single_hotspot,
                    ddp_hotspot=ddp_hotspot,
                ),
            ]
        )

    zero2 = find_row(rows, "deepspeed_zero2")
    zero3 = find_row(rows, "deepspeed_zero3_offload")
    if zero2 and zero3:
        zero2_hotspot = (zero2.get("top_hotspot") or {}).get("key", "-")
        lines.extend(
            [
                "- DeepSpeed family：`zero2` 的 profile 吞吐为 `{zero2_tokens}`，`zero3_offload` 为 `{zero3_tokens}`，前者约是后者的 `{ratio}`；峰值显存分别为 `{zero2_mem} MB` 和 `{zero3_mem} MB`，`zero3_offload` 约只用到 `zero2` 的 `{mem_ratio}`。".format(
                    zero2_tokens=format_float(zero2.get("tokens_per_second")),
                    zero3_tokens=format_float(zero3.get("tokens_per_second")),
                    ratio=format_ratio(zero2.get("tokens_per_second"), zero3.get("tokens_per_second")),
                    zero2_mem=format_float(zero2.get("max_allocated_mb"), 1),
                    zero3_mem=format_float(zero3.get("max_allocated_mb"), 1),
                    mem_ratio=(
                        f"{(float(zero3.get('max_allocated_mb')) / float(zero2.get('max_allocated_mb')) * 100.0):.1f}%"
                        if zero2.get("max_allocated_mb") not in (None, 0) and zero3.get("max_allocated_mb") is not None
                        else "-"
                    ),
                ),
                "- DeepSpeed family：`zero2` 的 top hotspot 是 `{zero2_hotspot}`，且保留了 DeepSpeed flops profile；`zero3_offload` 当前未启用 torch profiler，只保留 runtime、显存和 comms summary，因为该路线在本机上叠加 torch profiler 不稳定。".format(
                    zero2_hotspot=zero2_hotspot,
                ),
            ]
        )

    lines.extend(
        [
            "- Interpretation：如果目标是正式训练吞吐，当前仍应优先参考 benchmark 里的 `native ddp` 和 `zero2`；如果目标是解释 profile 现象，这份报告更支持“native 侧主要受 kernel launch / sync 约束，DeepSpeed 侧主要受 optimizer / partition orchestration 约束”的判断。",
            "",
        ]
    )
    return lines


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# CPT Profiling Report",
        "",
        "本报告汇总 `torch profiler` 与 DeepSpeed Flops Profiler 产物，用于解释 benchmark 中的主要 hotspot。",
        "",
        "| Experiment | Backend | Runtime (s) | Tokens/s | Max Alloc (MB) | Top Hotspot | Hotspot Share | Flops Profile |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        top_hotspot = row["top_hotspot"] or {}
        lines.append(
            "| {experiment} | {backend} | {runtime} | {tokens} | {max_alloc} | {hotspot} | {share} | {flops} |".format(
                experiment=row["experiment"],
                backend=row["backend"],
                runtime=format_float(row["runtime"], 1),
                tokens=format_float(row["tokens_per_second"]),
                max_alloc=format_float(row["max_allocated_mb"]),
                hotspot=top_hotspot.get("key", "-"),
                share=format_float((top_hotspot.get("time_share") or 0.0) * 100, 1) if top_hotspot else "-",
                flops="yes" if row["deepspeed_flops_profile"] else "-",
            )
        )
    lines.append("")
    lines.extend(build_analysis(rows))

    for row in rows:
        lines.extend(
            [
                f"## {row['experiment']}",
                "",
                f"- Torch profile dir: `{row['torch_profile_dir']}`" if row["torch_profile_available"] else "- Torch profile dir: `-`",
                f"- DeepSpeed flops profile: `{row['deepspeed_flops_profile']}`" if row["deepspeed_flops_profile"] else "- DeepSpeed flops profile: `-`",
                "- Top 5 non-communication hotspots:",
            ]
        )
        if not row["hotspots"]:
            lines.append("- `-`")
        else:
            for hotspot in row["hotspots"]:
                lines.append(
                    "- `{key}` count=`{count}` time=`{time_us}`us share=`{share}`".format(
                        key=hotspot["key"],
                        count=hotspot["count"],
                        time_us=format_float(hotspot["primary_time_us"], 1),
                        share=format_float((hotspot.get("time_share") or 0.0) * 100, 1),
                    )
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    rows = [build_row(run_dir) for run_dir in args.run_dirs]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(rows), encoding="utf-8")
    print(f"[DONE] wrote profiling report to {output_path}")


if __name__ == "__main__":
    main()
