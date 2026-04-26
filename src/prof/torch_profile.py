#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any


COMMUNICATION_PATTERNS = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "barrier",
    "nccl",
    "c10d",
)


def load_torch_profile_summary(profile_dir: str | Path, rank: int = 0) -> dict[str, Any] | None:
    summary_path = Path(profile_dir) / f"torch_profile_summary_rank{rank}.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_profile_events(profile_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not profile_summary:
        return []
    events = profile_summary.get("events", [])
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def get_primary_time_key(events: list[dict[str, Any]]) -> str:
    cuda_total = sum(float(event.get("self_cuda_time_total_us", 0.0) or 0.0) for event in events)
    if cuda_total > 0:
        return "self_cuda_time_total_us"
    return "self_cpu_time_total_us"


def top_events(
    events: list[dict[str, Any]],
    *,
    limit: int = 10,
    predicate: Any = None,
) -> list[dict[str, Any]]:
    filtered = events if predicate is None else [event for event in events if predicate(event)]
    sort_key = get_primary_time_key(filtered or events)
    return sorted(filtered, key=lambda event: float(event.get(sort_key, 0.0) or 0.0), reverse=True)[:limit]


def is_communication_event(event: dict[str, Any]) -> bool:
    key = str(event.get("key", "")).lower()
    return any(pattern in key for pattern in COMMUNICATION_PATTERNS)


def summarize_event(event: dict[str, Any], *, total_time_us: float | None = None) -> dict[str, Any]:
    self_cuda = float(event.get("self_cuda_time_total_us", 0.0) or 0.0)
    self_cpu = float(event.get("self_cpu_time_total_us", 0.0) or 0.0)
    primary_time_us = self_cuda if self_cuda > 0 else self_cpu
    time_share = None
    if total_time_us and total_time_us > 0:
        time_share = primary_time_us / total_time_us
    return {
        "key": event.get("key", "-"),
        "count": int(event.get("count", 0) or 0),
        "primary_time_us": primary_time_us,
        "self_cuda_time_total_us": self_cuda,
        "self_cpu_time_total_us": self_cpu,
        "time_share": time_share,
        "flops": float(event.get("flops", 0.0) or 0.0),
    }


def total_primary_time_us(events: list[dict[str, Any]]) -> float:
    time_key = get_primary_time_key(events)
    return sum(float(event.get(time_key, 0.0) or 0.0) for event in events)
