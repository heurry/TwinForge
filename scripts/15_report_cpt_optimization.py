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

import yaml


DEFAULT_RUN_DIRS = [
    "runs/cpt/optimize/native/qwen3_1_7b/ddp_base",
    "runs/cpt/optimize/native/qwen3_1_7b/ddp_workers4",
    "runs/cpt/optimize/native/qwen3_1_7b/ddp_no_gc",
    "runs/cpt/optimize/deepspeed/qwen3_1_7b/zero2_base",
    "runs/cpt/optimize/deepspeed/qwen3_1_7b/zero2_workers4",
    "runs/cpt/optimize/deepspeed/qwen3_1_7b/zero2_no_gc",
]
DEFAULT_STATUS_FILE = "runs/cpt/optimize/run_status.tsv"


def parse_args():
    parser = argparse.ArgumentParser(description="Render a markdown summary from CPT optimization sweep artifacts.")
    parser.add_argument("--run_dirs", nargs="*", default=DEFAULT_RUN_DIRS)
    parser.add_argument("--status_file", type=str, default=DEFAULT_STATUS_FILE)
    parser.add_argument("--output", type=str, default="reports/cpt_optimization.md")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_status_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    lines = path.read_text(encoding="utf-8").splitlines()
    entries: dict[str, dict[str, Any]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        step, status, exit_code = parts
        entries[step] = {
            "status": status,
            "exit_code": int(exit_code),
        }
    return entries


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


def infer_family(run_path: Path) -> str:
    if "native" in run_path.parts:
        return "native"
    if "deepspeed" in run_path.parts:
        return "deepspeed"
    return "other"


def guess_config_path(family: str, variant: str) -> Path | None:
    if family == "native":
        return ROOT_DIR / "configs" / "train" / "optimize" / "native" / f"cpt_opt_native_{variant}.yaml"
    if family == "deepspeed":
        return ROOT_DIR / "configs" / "train" / "optimize" / "deepspeed" / f"cpt_opt_{variant}.yaml"
    return None


def status_entry_for_variant(status_map: dict[str, dict[str, Any]], family: str, variant: str) -> dict[str, Any] | None:
    candidates = [variant, f"{family}_{variant}"]
    if family == "native":
        candidates.insert(0, f"native_{variant}")
    if family == "deepspeed":
        candidates.insert(0, variant)

    for key in candidates:
        if key in status_map:
            return status_map[key]
    return None


def build_row(run_dir: str, status_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_path = Path(run_dir)
    family = infer_family(run_path)
    variant = run_path.name
    summary = load_json(run_path / "train_summary.json")
    config = load_yaml(guess_config_path(family, variant) or Path(""))
    status_entry = status_entry_for_variant(status_map, family, variant)
    train_metrics = summary.get("train_metrics", {}) if summary else {}
    per_device_train_batch_size = (
        summary.get("per_device_train_batch_size")
        if summary
        else (config.get("per_device_train_batch_size") if config else None)
    )
    gradient_accumulation_steps = (
        summary.get("gradient_accumulation_steps")
        if summary
        else (config.get("gradient_accumulation_steps") if config else None)
    )
    world_size = summary.get("world_size") if summary else (2 if family in {"native", "deepspeed"} else None)
    effective_batch_size = summary.get("effective_batch_size") if summary else (
        per_device_train_batch_size * gradient_accumulation_steps * world_size
        if per_device_train_batch_size is not None and gradient_accumulation_steps is not None and world_size is not None
        else None
    )
    if summary:
        status = summary.get("run_status", "completed")
    elif status_entry:
        status = status_entry["status"]
    else:
        status = "missing"
    return {
        "family": family,
        "variant": variant,
        "experiment": summary.get("experiment_name") if summary else run_path.name,
        "backend": summary.get("training_backend") if summary else (config.get("training_backend") if config else run_path.name),
        "benchmark_group": summary.get("benchmark_group") if summary else None,
        "effective_batch_size": effective_batch_size,
        "dataloader_num_workers": summary.get("dataloader_num_workers") if summary else (config.get("dataloader_num_workers") if config else None),
        "gradient_checkpointing": summary.get("gradient_checkpointing") if summary else (config.get("gradient_checkpointing") if config else None),
        "runtime": train_metrics.get("train_runtime"),
        "tokens_per_second": summary.get("train_tokens_per_second") if summary else None,
        "max_allocated_mb": extract_peak_memory(summary),
        "status": status,
        "exit_code": status_entry["exit_code"] if status_entry else None,
        "path": str(run_path),
    }


def find_variant(rows: list[dict[str, Any]], family: str, variant: str) -> dict[str, Any] | None:
    for row in rows:
        if row["family"] == family and row["variant"] == variant:
            return row
    return None


def completed_rows(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["family"] == family and row["status"] == "completed"]


def build_family_analysis(rows: list[dict[str, Any]], family: str, baseline_variant: str, title: str) -> list[str]:
    baseline = find_variant(rows, family, baseline_variant)
    completed = completed_rows(rows, family)
    if not baseline or baseline["status"] != "completed":
        return [f"- {title}：baseline `{baseline_variant}` 结果缺失，当前还不能形成稳定推荐。"]

    analysis = []
    best = max(completed, key=lambda item: float(item["tokens_per_second"] or 0.0)) if completed else baseline
    analysis.append(
        "- {title}：baseline `{baseline}` 的吞吐为 `{baseline_tokens}`，峰值显存 `{baseline_mem} MB`。".format(
            title=title,
            baseline=baseline["variant"],
            baseline_tokens=format_float(baseline["tokens_per_second"]),
            baseline_mem=format_float(baseline["max_allocated_mb"], 1),
        )
    )
    if best and best["variant"] != baseline["variant"]:
        analysis.append(
            "- {title}：当前已完成变体中吞吐最高的是 `{best}`，达到 `{best_tokens}`，相对 baseline 变化 `{delta}`。".format(
                title=title,
                best=best["variant"],
                best_tokens=format_float(best["tokens_per_second"]),
                delta=format_percent_delta(best["tokens_per_second"], baseline["tokens_per_second"]),
            )
        )
    else:
        analysis.append(f"- {title}：当前已完成变体里，baseline 仍是吞吐最高或唯一可比样本。")

    failed = [row["variant"] for row in rows if row["family"] == family and row["status"] == "failed"]
    missing = [row["variant"] for row in rows if row["family"] == family and row["status"] == "missing"]
    if failed:
        analysis.append(
            "- {title}：以下变体在本轮 sweep 中已确认失败：`{variants}`。".format(
                title=title,
                variants="`, `".join(failed),
            )
        )
    if missing:
        analysis.append(
            "- {title}：以下变体当前仍缺少结果文件：`{variants}`。".format(
                title=title,
                variants="`, `".join(missing),
            )
        )
    return analysis


def build_analysis(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Analysis",
        "",
        "- 当前 Optimization Track 的第一版 sweep 只覆盖当前 `train_cpt.py` 已稳定接入且可直接控制的旋钮：`dataloader_num_workers` 和 `gradient_checkpointing`。",
        "- `max_seq_length` 与更激进的 batch-density sweep 仍需要新的 tokenized slice 或额外显存验证，因此暂不纳入这一轮自动 sweep。",
    ]
    lines.extend(build_family_analysis(rows, "native", "ddp_base", "Native DDP"))
    lines.extend(build_family_analysis(rows, "deepspeed", "zero2_base", "DeepSpeed ZeRO-2"))
    lines.extend(
        [
            "- Recommendation：先用这轮结果筛出 `native ddp` 和 `zero2` 各自的推荐配置，再进入更长 step 的正式长训和中断恢复验证。",
            "",
        ]
    )
    return lines


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# CPT Optimization Report",
        "",
        "本报告汇总 `Optimization Track` 的第一版 sweep，当前聚焦已验证主线 `native ddp` 与 `zero2`，用于筛选进入正式长训的推荐配置。",
        "",
        "| Family | Variant | Backend | Workers | GC | Eff Batch | Runtime (s) | Tokens/s | Max Alloc (MB) | Status | Exit Code |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {family} | {variant} | {backend} | {workers} | {gc} | {eff_batch} | {runtime} | {tokens} | {max_alloc} | {status} | {exit_code} |".format(
                family=row["family"],
                variant=row["variant"],
                backend=row["backend"],
                workers=row["dataloader_num_workers"] if row["dataloader_num_workers"] is not None else "-",
                gc=row["gradient_checkpointing"] if row["gradient_checkpointing"] is not None else "-",
                eff_batch=row["effective_batch_size"] if row["effective_batch_size"] is not None else "-",
                runtime=format_float(row["runtime"], 1),
                tokens=format_float(row["tokens_per_second"]),
                max_alloc=format_float(row["max_allocated_mb"], 1),
                status=row["status"],
                exit_code=row["exit_code"] if row["exit_code"] is not None else "-",
            )
        )
    lines.append("")
    lines.extend(build_analysis(rows))

    for family in ("native", "deepspeed"):
        lines.extend([f"## {family.title()} Variants", ""])
        family_rows = [row for row in rows if row["family"] == family]
        for row in family_rows:
            lines.append(
                "- `{variant}` backend=`{backend}` workers=`{workers}` gc=`{gc}` tokens/s=`{tokens}` max_alloc=`{mem}` status=`{status}` path=`{path}`".format(
                    variant=row["variant"],
                    backend=row["backend"],
                    workers=row["dataloader_num_workers"] if row["dataloader_num_workers"] is not None else "-",
                    gc=row["gradient_checkpointing"] if row["gradient_checkpointing"] is not None else "-",
                    tokens=format_float(row["tokens_per_second"]),
                    mem=format_float(row["max_allocated_mb"], 1),
                    status=row["status"],
                    path=row["path"],
                )
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    status_map = load_status_map(Path(args.status_file))
    rows = [build_row(run_dir, status_map) for run_dir in args.run_dirs]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(rows), encoding="utf-8")
    print(f"[DONE] wrote optimization report to {output_path}")


if __name__ == "__main__":
    main()
