#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_NATIVE_RUN_DIRS = [
    "runs/cpt/bench/native/qwen3_1_7b/single",
    "runs/cpt/bench/native/qwen3_1_7b/ddp",
]
DEFAULT_DEEPSPEED_RUN_DIRS = [
    "runs/cpt/bench/deepspeed/qwen3_1_7b/zero2",
    "runs/cpt/bench/deepspeed/qwen3_1_7b/zero3_offload",
]
DEFAULT_BRIDGE_RUN_DIR = "runs/cpt/bench/native/qwen3_1_7b/ddp"


def parse_args():
    parser = argparse.ArgumentParser(description="Render a family-split markdown comparison table from CPT benchmark summaries.")
    parser.add_argument(
        "--native_run_dirs",
        nargs="*",
        default=DEFAULT_NATIVE_RUN_DIRS,
        help="Native benchmark run directories that contain train_summary.json.",
    )
    parser.add_argument(
        "--deepspeed_run_dirs",
        nargs="*",
        default=DEFAULT_DEEPSPEED_RUN_DIRS,
        help="DeepSpeed benchmark run directories that contain train_summary.json.",
    )
    parser.add_argument(
        "--bridge_run_dir",
        type=str,
        default=DEFAULT_BRIDGE_RUN_DIR,
        help="Native DDP run directory reused as a cross-family bridge reference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional markdown output file.",
    )
    return parser.parse_args()


def load_summary(run_dir: str) -> dict[str, Any] | None:
    summary_path = Path(run_dir) / "train_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value: Any, precision: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{precision}f}"


def format_bool(value: Any) -> str:
    if value is None:
        return "-"
    return str(bool(value))


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def guess_backend_from_dir(run_dir: str) -> str:
    name = Path(run_dir).name
    if name == "single":
        return "single_gpu"
    if name == "ddp":
        return "ddp"
    if name == "zero2":
        return "deepspeed_zero2"
    if name == "zero3_offload":
        return "deepspeed_zero3_offload"
    return name


def guess_experiment_from_dir(run_dir: str) -> str:
    name = Path(run_dir).name
    family = Path(run_dir).parent.parent.name if len(Path(run_dir).parts) >= 3 else "bench"
    if family == "native" and name == "single":
        return "cpt_bench_native_single"
    if family == "native" and name == "ddp":
        return "cpt_bench_native_ddp"
    if family == "deepspeed" and name == "zero2":
        return "cpt_bench_zero2"
    if family == "deepspeed" and name == "zero3_offload":
        return "cpt_bench_zero3_offload"
    return f"{family}_{name}"


def extract_peak_memory(summary: dict[str, Any]) -> float | None:
    output_dir = summary.get("output_dir")
    if not output_dir:
        return None

    train_log_path = Path(output_dir) / "train_log_history.json"
    if not train_log_path.exists():
        return None

    with open(train_log_path, "r", encoding="utf-8") as f:
        log_history = json.load(f)
    values = [
        float(item["cuda_memory_max_allocated_mb"])
        for item in log_history
        if isinstance(item, dict) and "cuda_memory_max_allocated_mb" in item
    ]
    return max(values) if values else None


def build_row(
    run_dir: str,
    *,
    fallback_experiment: str | None = None,
    fallback_backend: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    summary = load_summary(run_dir)
    train_metrics = summary.get("train_metrics", {}) if summary else {}
    experiment_name = fallback_experiment or (summary.get("experiment_name") if summary else None) or guess_experiment_from_dir(run_dir)
    if note:
        experiment_name = f"{experiment_name} ({note})"

    return {
        "experiment_name": experiment_name,
        "training_backend": fallback_backend or (summary.get("training_backend") if summary else None) or guess_backend_from_dir(run_dir),
        "benchmark_group": summary.get("benchmark_group") if summary else None,
        "train_runtime": train_metrics.get("train_runtime"),
        "train_samples_per_second": train_metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_metrics.get("train_steps_per_second"),
        "train_tokens_per_second": summary.get("train_tokens_per_second") if summary else None,
        "train_loss": train_metrics.get("train_loss"),
        "max_allocated_mb": extract_peak_memory(summary) if summary else None,
        "last_checkpoint_save_seconds": summary.get("last_checkpoint_save_seconds") if summary else None,
        "resume_success": summary.get("resume_success") if summary else None,
    }


def build_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Experiment | Backend | Benchmark Group | Runtime (s) | Samples/s | Steps/s | Tokens/s | Train Loss | Max Alloc (MB) | Last Save (s) | Resume |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {backend} | {group} | {runtime} | {samples} | {steps} | {tokens} | {loss} | {max_alloc} | {save_s} | {resume} |".format(
                experiment=row["experiment_name"],
                backend=row["training_backend"],
                group=row["benchmark_group"] or "-",
                runtime=format_float(row["train_runtime"], 1),
                samples=format_float(row["train_samples_per_second"]),
                steps=format_float(row["train_steps_per_second"]),
                tokens=format_float(row["train_tokens_per_second"]),
                loss=format_float(row["train_loss"]),
                max_alloc=format_float(row["max_allocated_mb"]),
                save_s=format_float(row["last_checkpoint_save_seconds"]),
                resume=format_bool(row["resume_success"]),
            )
        )
    lines.append("")
    return lines


def find_row(rows: list[dict[str, Any]], backend: str) -> dict[str, Any] | None:
    for row in rows:
        if row["training_backend"] == backend:
            return row
    return None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def relative_change(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return (new - old) / old


def format_percent(value: float | None, precision: int = 1, signed: bool = False) -> str:
    if value is None:
        return "-"
    if signed:
        return f"{value * 100:+.{precision}f}%"
    return f"{value * 100:.{precision}f}%"


def build_analysis_section(
    native_rows: list[dict[str, Any]],
    bridge_row: dict[str, Any],
    deepspeed_rows: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "## Analysis",
        "",
    ]

    native_single = find_row(native_rows, "single_gpu")
    native_ddp = find_row(native_rows, "ddp")
    if native_single and native_ddp:
        single_tokens = to_float(native_single["train_tokens_per_second"])
        ddp_tokens = to_float(native_ddp["train_tokens_per_second"])
        single_runtime = to_float(native_single["train_runtime"])
        ddp_runtime = to_float(native_ddp["train_runtime"])
        single_loss = to_float(native_single["train_loss"])
        ddp_loss = to_float(native_ddp["train_loss"])
        single_mem = to_float(native_single["max_allocated_mb"])
        ddp_mem = to_float(native_ddp["max_allocated_mb"])

        lines.extend(
            [
                "### Native Family",
                "",
                (
                    f"- 在相同 `effective_batch_size=8` 条件下，`ddp` 的 `tokens/s` 从 "
                    f"`{format_float(single_tokens)}` 提升到 `{format_float(ddp_tokens)}`，约 "
                    f"`{format_float(ratio(ddp_tokens, single_tokens), 2)}x`；`runtime` 从 "
                    f"`{format_float(single_runtime, 1)}s` 降到 `{format_float(ddp_runtime, 1)}s`，约 "
                    f"`{format_percent(-relative_change(ddp_runtime, single_runtime))}` 更快。"
                ),
                (
                    f"- 两者 `train_loss` 基本一致：`{format_float(single_loss)}` vs "
                    f"`{format_float(ddp_loss)}`，差值约 "
                    f"`{format_float(abs(ddp_loss - single_loss) if ddp_loss is not None and single_loss is not None else None, 6)}`。"
                ),
                (
                    f"- 代价是峰值显存从 `{format_float(single_mem)}` MB 增到 "
                    f"`{format_float(ddp_mem)}` MB，约 `{format_percent(relative_change(ddp_mem, single_mem))}`。"
                ),
                "- 结论：在当前双卡 `3090` 的 native 轨道里，`ddp` 是默认首选训练后端。",
                "",
            ]
        )

    zero2 = find_row(deepspeed_rows, "deepspeed_zero2")
    zero3 = find_row(deepspeed_rows, "deepspeed_zero3_offload")
    if zero2 and zero3:
        zero2_tokens = to_float(zero2["train_tokens_per_second"])
        zero3_tokens = to_float(zero3["train_tokens_per_second"])
        zero2_runtime = to_float(zero2["train_runtime"])
        zero3_runtime = to_float(zero3["train_runtime"])
        zero2_loss = to_float(zero2["train_loss"])
        zero3_loss = to_float(zero3["train_loss"])
        zero2_mem = to_float(zero2["max_allocated_mb"])
        zero3_mem = to_float(zero3["max_allocated_mb"])
        zero2_save = to_float(zero2["last_checkpoint_save_seconds"])
        zero3_save = to_float(zero3["last_checkpoint_save_seconds"])

        lines.extend(
            [
                "### DeepSpeed Family",
                "",
                (
                    f"- 在相同 `effective_batch_size=16` 的 `fp16` 协议下，`zero2` 的 `tokens/s` 为 "
                    f"`{format_float(zero2_tokens)}`，`zero3_offload` 为 `{format_float(zero3_tokens)}`；"
                    f"`zero2` 约快 `{format_float(ratio(zero2_tokens, zero3_tokens), 2)}x`。"
                ),
                (
                    f"- `zero3_offload` 的峰值显存从 `{format_float(zero2_mem)}` MB 降到 "
                    f"`{format_float(zero3_mem)}` MB，约减少 "
                    f"`{format_percent(-relative_change(zero3_mem, zero2_mem))}`。"
                ),
                (
                    f"- 两者 `train_loss` 仍基本一致：`{format_float(zero2_loss)}` vs "
                    f"`{format_float(zero3_loss)}`，差值约 "
                    f"`{format_float(abs(zero3_loss - zero2_loss) if zero3_loss is not None and zero2_loss is not None else None, 6)}`；"
                    f"`last save` 为 `{format_float(zero2_save)}`s vs `{format_float(zero3_save)}`s。"
                ),
                "- 结论：显存扛得住时优先 `zero2`；需要为更大模型、更长上下文或更小显存预算让路时再选 `zero3_offload`。",
                "",
            ]
        )

    if bridge_row:
        lines.extend(
            [
                "### Cross-Family Note",
                "",
                (
                    "- `native_ddp_bridge_ref` 仍然只作参考，因为它和 DeepSpeed family 的 "
                    "`benchmark_group`、precision、effective batch 都不同，不能直接拿来下严格性能结论。"
                ),
                "",
            ]
        )

    return lines


def build_markdown(native_rows: list[dict[str, Any]], bridge_row: dict[str, Any], deepspeed_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# CPT Benchmark Report",
        "",
        "本报告按 family 拆分展示：`Native Family` 内部可比，`DeepSpeed Family` 内部可比。",
        "DeepSpeed 章节中的 `native_ddp_bridge_ref` 仅作跨 family 参考，不参与严格同协议结论。",
        "",
    ]
    lines.extend(build_table("Native Family", native_rows))
    lines.append("## DeepSpeed Family")
    lines.append("")
    lines.append("注：`native_ddp_bridge_ref` 为跨 family 参考行，不能直接与 ZeRO family 得出严格同协议结论。")
    lines.append("")
    lines.extend(build_table_rows([bridge_row] + deepspeed_rows))
    lines.extend(build_analysis_section(native_rows, bridge_row, deepspeed_rows))
    return "\n".join(lines) + "\n"


def build_table_rows(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Experiment | Backend | Benchmark Group | Runtime (s) | Samples/s | Steps/s | Tokens/s | Train Loss | Max Alloc (MB) | Last Save (s) | Resume |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {backend} | {group} | {runtime} | {samples} | {steps} | {tokens} | {loss} | {max_alloc} | {save_s} | {resume} |".format(
                experiment=row["experiment_name"],
                backend=row["training_backend"],
                group=row["benchmark_group"] or "-",
                runtime=format_float(row["train_runtime"], 1),
                samples=format_float(row["train_samples_per_second"]),
                steps=format_float(row["train_steps_per_second"]),
                tokens=format_float(row["train_tokens_per_second"]),
                loss=format_float(row["train_loss"]),
                max_alloc=format_float(row["max_allocated_mb"]),
                save_s=format_float(row["last_checkpoint_save_seconds"]),
                resume=format_bool(row["resume_success"]),
            )
        )
    lines.append("")
    return lines


def main():
    args = parse_args()
    native_rows = [build_row(run_dir) for run_dir in args.native_run_dirs]
    bridge_row = build_row(
        args.bridge_run_dir,
        fallback_experiment="native_ddp_bridge_ref",
        fallback_backend="ddp_bridge_ref",
        note="cross-family reference only",
    )
    deepspeed_rows = [build_row(run_dir) for run_dir in args.deepspeed_run_dirs]

    markdown = build_markdown(native_rows, bridge_row, deepspeed_rows)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"[DONE] wrote benchmark report to {output_path}")
        return

    print(markdown)


if __name__ == "__main__":
    main()
