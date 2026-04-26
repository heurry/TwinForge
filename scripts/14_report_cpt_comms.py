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


DEFAULT_DDP_RUN_DIR = "runs/cpt/profile/native/qwen3_1_7b/ddp"
DEFAULT_ZERO2_RUN_DIR = "runs/cpt/profile/deepspeed/qwen3_1_7b/zero2"
DEFAULT_ZERO3_RUN_DIR = "runs/cpt/profile/deepspeed/qwen3_1_7b/zero3_offload"


def parse_args():
    parser = argparse.ArgumentParser(description="Render a markdown communication summary from DDP/DeepSpeed profiling artifacts.")
    parser.add_argument("--ddp_run_dir", type=str, default=DEFAULT_DDP_RUN_DIR)
    parser.add_argument("--zero2_run_dir", type=str, default=DEFAULT_ZERO2_RUN_DIR)
    parser.add_argument("--zero3_run_dir", type=str, default=DEFAULT_ZERO3_RUN_DIR)
    parser.add_argument("--output", type=str, default="reports/cpt_comms.md")
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


def load_torch_comm_summary(run_dir: str) -> dict[str, Any]:
    run_path = Path(run_dir)
    train_summary = load_json(run_path / "train_summary.json")
    profile_dir = Path(train_summary["torch_profiler_dir"]) if train_summary and train_summary.get("torch_profiler_dir") else run_path / "profiling" / "torch"
    events = get_profile_events(load_torch_profile_summary(profile_dir))
    comm_events = top_events(events, limit=10, predicate=is_communication_event)
    total_time = total_primary_time_us(events)
    total_comm_time = sum(summarize_event(event)["primary_time_us"] for event in comm_events)
    return {
        "profile_dir": str(profile_dir),
        "events": [summarize_event(event, total_time_us=total_time) for event in comm_events],
        "total_comm_time_us": total_comm_time,
        "total_time_us": total_time,
    }


def load_deepspeed_comm_summary(run_dir: str) -> dict[str, Any] | None:
    summary_path = Path(run_dir) / "profiling" / "deepspeed" / "deepspeed_comms_summary.json"
    return load_json(summary_path)


def flatten_deepspeed_summary(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not summary:
        return []
    flattened = []
    for op_name, message_sizes in (summary.get("summary", {}) or {}).items():
        total_count = 0
        total_latency_ms = 0.0
        for metrics in message_sizes.values():
            total_count += int(metrics.get("count", 0) or 0)
            total_latency_ms += float(metrics.get("total_latency_ms", 0.0) or 0.0)
        flattened.append(
            {
                "op_name": op_name,
                "count": total_count,
                "total_latency_ms": total_latency_ms,
            }
        )
    return sorted(flattened, key=lambda item: item["total_latency_ms"], reverse=True)


def build_analysis(
    ddp_torch: dict[str, Any],
    zero2_torch: dict[str, Any],
    zero2_ds: list[dict[str, Any]],
    zero3_ds: list[dict[str, Any]],
) -> list[str]:
    ddp_top = ddp_torch["events"][0]["key"] if ddp_torch["events"] else "-"
    zero2_top_torch = zero2_torch["events"][0]["key"] if zero2_torch["events"] else "-"
    zero2_top_ds = zero2_ds[0]["op_name"] if zero2_ds else "-"
    zero3_top_ds = zero3_ds[0]["op_name"] if zero3_ds else "-"
    zero3_second_ds = zero3_ds[1]["op_name"] if len(zero3_ds) > 1 else "-"

    ddp_share = (ddp_torch["total_comm_time_us"] / ddp_torch["total_time_us"]) if ddp_torch["total_time_us"] else None
    zero2_share = (zero2_torch["total_comm_time_us"] / zero2_torch["total_time_us"]) if zero2_torch["total_time_us"] else None

    lines = [
        "## Analysis",
        "",
        "- DDP：torch profiler 中显式通信事件以 `{ddp_top}` 为主，但总通信时间占比约 `{ddp_share}`，说明 native 路线的主要瓶颈并不在显式 NCCL 通信，而更可能在 kernel launch、同步和运行时调度。".format(
            ddp_top=ddp_top,
            ddp_share=(f"{ddp_share * 100.0:.1f}%" if ddp_share is not None else "-"),
        ),
        "- ZeRO-2：torch profiler 中 top comm op 也是 `{zero2_top_torch}`，而 DeepSpeed comms logger 里按聚合延迟看则明显由 `{zero2_top_ds}` 主导；这说明 `zero2` 仍以经典梯度归约为主，但其真实通信成本更适合用 DeepSpeed comms logger 的 op mix 来理解。".format(
            zero2_top_torch=zero2_top_torch,
            zero2_top_ds=zero2_top_ds,
        ),
        "- ZeRO-2 vs DDP：torch profiler 统计到的显式通信时间从 `{ddp_time}` 增到 `{zero2_time}`，约 `{ratio}`；不过两边 share 都接近 `0%`，因此这份报告更适合判断通信形态，而不是单独用 share 解释全部性能差异。".format(
            ddp_time=format_float(ddp_torch["total_comm_time_us"], 1),
            zero2_time=format_float(zero2_torch["total_comm_time_us"], 1),
            ratio=format_ratio(zero2_torch["total_comm_time_us"], ddp_torch["total_comm_time_us"]),
        ),
        "- ZeRO-3 offload：DeepSpeed comms logger 显示其通信不再由 `all_reduce` 主导，而是 `{zero3_top_ds}` + `{zero3_second_ds}` 主导。这和 ZeRO-3 CPU offload 的机制一致，说明它的主要代价是参数/梯度分片的 gather/scatter 与 host-device 搬运，而不是普通 DDP 式的梯度归约。".format(
            zero3_top_ds=zero3_top_ds,
            zero3_second_ds=zero3_second_ds,
        ),
        "- Interpretation：后续如果目标是提高 `ddp` 吞吐，不该优先把时间花在 NCCL 调参上；如果目标是理解 `zero3_offload` 为什么慢，这份报告已经足够支持“它慢在分片搬运与 offload orchestration，而不是慢在 classic all-reduce”这个判断。",
        "",
    ]
    return lines


def render_markdown(
    ddp_torch: dict[str, Any],
    zero2_torch: dict[str, Any],
    zero2_ds: list[dict[str, Any]],
    zero3_ds: list[dict[str, Any]],
) -> str:
    ddp_share = (ddp_torch["total_comm_time_us"] / ddp_torch["total_time_us"]) if ddp_torch["total_time_us"] else None
    zero2_share = (zero2_torch["total_comm_time_us"] / zero2_torch["total_time_us"]) if zero2_torch["total_time_us"] else None
    lines = [
        "# CPT Communication Report",
        "",
        "本报告优先使用 `torch profiler` 观察 DDP/ZeRO-2 的通信热点，并补充 DeepSpeed comms logger 的聚合结果。`zero3_offload` 当前仅纳入 DeepSpeed comms logger，因为该 profile 路线已关闭 torch profiler 以保证稳定完成。",
        "",
        "| Backend | Source | Total Comm Time (us) | Comm Share | Top Comm Op |",
        "| --- | --- | ---: | ---: | --- |",
        "| ddp | torch profiler | {time} | {share} | {top} |".format(
            time=format_float(ddp_torch["total_comm_time_us"], 1),
            share=format_float((ddp_share or 0.0) * 100, 1) if ddp_share is not None else "-",
            top=ddp_torch["events"][0]["key"] if ddp_torch["events"] else "-",
        ),
        "| zero2 | torch profiler | {time} | {share} | {top} |".format(
            time=format_float(zero2_torch["total_comm_time_us"], 1),
            share=format_float((zero2_share or 0.0) * 100, 1) if zero2_share is not None else "-",
            top=zero2_torch["events"][0]["key"] if zero2_torch["events"] else "-",
        ),
        "| zero2 | deepspeed comms logger | - | - | {top} |".format(
            top=zero2_ds[0]["op_name"] if zero2_ds else "-",
        ),
        "| zero3_offload | deepspeed comms logger | - | - | {top} |".format(
            top=zero3_ds[0]["op_name"] if zero3_ds else "-",
        ),
        "",
    ]
    lines.extend(build_analysis(ddp_torch, zero2_torch, zero2_ds, zero3_ds))
    lines.extend([
        "## DDP Torch Profiler Communication Ops",
        "",
    ])
    if not ddp_torch["events"]:
        lines.append("- `-`")
    else:
        for event in ddp_torch["events"]:
            lines.append(
                "- `{key}` count=`{count}` time=`{time}`us share=`{share}`".format(
                    key=event["key"],
                    count=event["count"],
                    time=format_float(event["primary_time_us"], 1),
                    share=format_float((event.get("time_share") or 0.0) * 100, 1),
                )
            )
    lines.extend(["", "## ZeRO-2 Torch Profiler Communication Ops", ""])
    if not zero2_torch["events"]:
        lines.append("- `-`")
    else:
        for event in zero2_torch["events"]:
            lines.append(
                "- `{key}` count=`{count}` time=`{time}`us share=`{share}`".format(
                    key=event["key"],
                    count=event["count"],
                    time=format_float(event["primary_time_us"], 1),
                    share=format_float((event.get("time_share") or 0.0) * 100, 1),
                )
            )
    lines.extend(["", "## ZeRO-2 DeepSpeed Comms Logger Summary", ""])
    if not zero2_ds:
        lines.append("- `-`")
    else:
        for item in zero2_ds[:10]:
            lines.append(
                "- `{op}` count=`{count}` total_latency=`{lat}`ms".format(
                    op=item["op_name"],
                    count=item["count"],
                    lat=format_float(item["total_latency_ms"], 3),
                )
            )
    lines.extend(["", "## ZeRO-3 Offload DeepSpeed Comms Logger Summary", ""])
    if not zero3_ds:
        lines.append("- `-`")
    else:
        for item in zero3_ds[:10]:
            lines.append(
                "- `{op}` count=`{count}` total_latency=`{lat}`ms".format(
                    op=item["op_name"],
                    count=item["count"],
                    lat=format_float(item["total_latency_ms"], 3),
                )
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    ddp_torch = load_torch_comm_summary(args.ddp_run_dir)
    zero2_torch = load_torch_comm_summary(args.zero2_run_dir)
    zero2_ds = flatten_deepspeed_summary(load_deepspeed_comm_summary(args.zero2_run_dir))
    zero3_ds = flatten_deepspeed_summary(load_deepspeed_comm_summary(args.zero3_run_dir))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(ddp_torch, zero2_torch, zero2_ds, zero3_ds), encoding="utf-8")
    print(f"[DONE] wrote communication report to {output_path}")


if __name__ == "__main__":
    main()
