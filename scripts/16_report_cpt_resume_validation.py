#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RUNS = [
    (
        "native",
        "ddp_recommended",
        "runs/cpt/resume_validation/native/qwen3_1_7b/ddp_recommended/resume_validation",
    ),
    (
        "deepspeed",
        "zero2_recommended",
        "runs/cpt/resume_validation/deepspeed/qwen3_1_7b/zero2_recommended/resume_validation",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Render CPT resume validation report.")
    parser.add_argument("--output", type=str, default="reports/cpt_resume_validation.md")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value: Any, precision: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{precision}f}"


def build_row(family: str, variant: str, base_dir: Path) -> dict[str, Any]:
    stage1 = load_json(base_dir / "stage1_train_summary.json")
    stage2 = load_json(base_dir / "stage2_train_summary.json")
    checkpoint_10_exists = (base_dir.parent / "checkpoint-10").exists()
    checkpoint_12_exists = (base_dir.parent / "checkpoint-12").exists()

    stage1_step = stage1.get("global_step") if stage1 else None
    stage2_step = stage2.get("global_step") if stage2 else None
    resume_requested = stage2.get("resume_requested") if stage2 else None
    resume_success = stage2.get("resume_success") if stage2 else None
    stage2_checkpoint = stage2.get("resume_checkpoint") if stage2 else None

    validation_passed = bool(
        stage1
        and stage2
        and checkpoint_10_exists
        and checkpoint_12_exists
        and stage1_step == 10
        and stage2_step == 12
        and resume_requested is True
        and resume_success is True
    )

    return {
        "family": family,
        "variant": variant,
        "status": "passed" if validation_passed else ("incomplete" if stage1 or stage2 else "missing"),
        "stage1_step": stage1_step,
        "stage2_step": stage2_step,
        "resume_requested": resume_requested,
        "resume_success": resume_success,
        "resume_checkpoint": stage2_checkpoint,
        "checkpoint_10_exists": checkpoint_10_exists,
        "checkpoint_12_exists": checkpoint_12_exists,
        "stage1_tokens_per_second": stage1.get("train_tokens_per_second") if stage1 else None,
        "stage2_tokens_per_second": stage2.get("train_tokens_per_second") if stage2 else None,
        "stage1_max_allocated_mb": _extract_peak_memory(base_dir / "stage1_train_log_history.json"),
        "stage2_max_allocated_mb": _extract_peak_memory(base_dir / "stage2_train_log_history.json"),
        "base_dir": str(base_dir),
    }


def _extract_peak_memory(path: Path) -> float | None:
    history = load_json(path)
    if not history:
        return None
    values = [
        float(item["cuda_memory_max_allocated_mb"])
        for item in history
        if isinstance(item, dict) and "cuda_memory_max_allocated_mb" in item
    ]
    return max(values) if values else None


def build_analysis(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Analysis",
        "",
        "- 本报告用于验证推荐训练配置能否稳定完成 `checkpoint -> resume -> continue training`，而不是比较最终收敛。",
    ]
    for row in rows:
        if row["status"] == "passed":
            lines.append(
                "- `{variant}`：resume 验证通过，已确认 `step 10 -> checkpoint-10 -> resume -> step 12 -> checkpoint-12` 的完整链路成立。".format(
                    variant=row["variant"]
                )
            )
        elif row["status"] == "incomplete":
            lines.append(
                "- `{variant}`：已产生部分 resume 验证产物，但链路尚未完整闭环，需要检查 checkpoint 或 stage2 结果。".format(
                    variant=row["variant"]
                )
            )
        else:
            lines.append(
                "- `{variant}`：当前尚未产生 resume 验证产物。".format(
                    variant=row["variant"]
                )
            )
    lines.extend(
        [
            "- 只有当推荐配置的 resume 验证通过后，再用它去承接后续 SFT / eval / serving，才能避免在下游阶段暴露 checkpoint 不可恢复的问题。",
            "",
        ]
    )
    return lines


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# CPT Resume Validation Report",
        "",
        "本报告验证推荐训练配置的短程 `resume` 能力，确保后续 SFT / eval / serving 可基于稳定 checkpoint 继续推进。",
        "",
        "| Family | Variant | Status | Stage1 Step | Stage2 Step | Resume Requested | Resume Success | Ckpt-10 | Ckpt-12 | Stage1 Tokens/s | Stage2 Tokens/s |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {family} | {variant} | {status} | {stage1_step} | {stage2_step} | {resume_requested} | {resume_success} | {checkpoint_10_exists} | {checkpoint_12_exists} | {stage1_tokens_per_second} | {stage2_tokens_per_second} |".format(
                family=row["family"],
                variant=row["variant"],
                status=row["status"],
                stage1_step=row["stage1_step"] if row["stage1_step"] is not None else "-",
                stage2_step=row["stage2_step"] if row["stage2_step"] is not None else "-",
                resume_requested=row["resume_requested"] if row["resume_requested"] is not None else "-",
                resume_success=row["resume_success"] if row["resume_success"] is not None else "-",
                checkpoint_10_exists=row["checkpoint_10_exists"],
                checkpoint_12_exists=row["checkpoint_12_exists"],
                stage1_tokens_per_second=format_float(row["stage1_tokens_per_second"]),
                stage2_tokens_per_second=format_float(row["stage2_tokens_per_second"]),
            )
        )
    lines.append("")
    lines.extend(build_analysis(rows))
    for row in rows:
        lines.extend(
            [
                f"## {row['variant']}",
                "",
                f"- stage1 max alloc: `{format_float(row['stage1_max_allocated_mb'], 1)} MB`",
                f"- stage2 max alloc: `{format_float(row['stage2_max_allocated_mb'], 1)} MB`",
                f"- resume checkpoint: `{row['resume_checkpoint'] or '-'}`",
                f"- artifacts dir: `{row['base_dir']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    rows = [build_row(family, variant, Path(base_dir)) for family, variant, base_dir in DEFAULT_RUNS]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(rows), encoding="utf-8")
    print(f"[DONE] wrote resume validation report to {output_path}")


if __name__ == "__main__":
    main()
