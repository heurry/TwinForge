from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import floor
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class SourceSpec:
    name: str
    language: str
    requested_max_samples: Optional[int] = None


@dataclass
class SamplingPlan:
    total_target_samples: int
    language_quotas: Dict[str, int]
    source_quotas: Dict[str, int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_target_samples": self.total_target_samples,
            "language_quotas": dict(self.language_quotas),
            "source_quotas": dict(self.source_quotas),
        }


@dataclass
class QuotaTracker:
    source_quotas: Dict[str, int]
    counts: Dict[str, int] = field(default_factory=dict)

    def can_accept(self, source_name: str) -> bool:
        return self.counts.get(source_name, 0) < self.source_quotas.get(source_name, 0)

    def accept(self, source_name: str) -> None:
        self.counts[source_name] = self.counts.get(source_name, 0) + 1


def parse_mapping_arg(raw_value: Optional[str], value_type=float) -> Dict[str, float]:
    if not raw_value:
        return {}

    mapping = {}
    for chunk in raw_value.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(f"Invalid mapping item: {item!r}")
        mapping[key.strip()] = value_type(value.strip())
    return mapping


def _allocate_integer_quotas(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    positive_weights = {key: value for key, value in weights.items() if value > 0}
    quotas = {key: 0 for key in weights}
    if total <= 0 or not positive_weights:
        return quotas

    weight_sum = sum(positive_weights.values())
    raw_allocations = {
        key: total * (weight / weight_sum)
        for key, weight in positive_weights.items()
    }
    floor_allocations = {
        key: floor(value)
        for key, value in raw_allocations.items()
    }
    remaining = total - sum(floor_allocations.values())

    for key, value in floor_allocations.items():
        quotas[key] = value

    if remaining > 0:
        ranked = sorted(
            positive_weights,
            key=lambda key: (raw_allocations[key] - floor_allocations[key], key),
            reverse=True,
        )
        for key in ranked[:remaining]:
            quotas[key] += 1

    return quotas


def _allocate_capped_quotas(total: int, weights: Dict[str, float], caps: Dict[str, int]) -> Dict[str, int]:
    quotas = {key: 0 for key in caps}
    capped_total = min(total, sum(max(cap, 0) for cap in caps.values()))
    available = [key for key, cap in caps.items() if cap > 0]

    while capped_total > sum(quotas.values()) and available:
        remaining = capped_total - sum(quotas.values())
        active_weights = {key: weights.get(key, 0.0) for key in available}
        if sum(active_weights.values()) <= 0:
            active_weights = {key: 1.0 for key in available}

        proposed = _allocate_integer_quotas(remaining, active_weights)
        progress = False
        for key in list(available):
            room = caps[key] - quotas[key]
            add = min(room, proposed.get(key, 0))
            if add > 0:
                quotas[key] += add
                progress = True

        available = [key for key in available if quotas[key] < caps[key]]
        if progress:
            continue

        for key in available:
            if sum(quotas.values()) >= capped_total:
                break
            quotas[key] += 1
        available = [key for key in available if quotas[key] < caps[key]]

    return quotas


def _default_language_weights(specs: Iterable[SourceSpec]) -> Dict[str, float]:
    weights: Dict[str, float] = defaultdict(float)
    for spec in specs:
        weights[spec.language] += float(spec.requested_max_samples or 1)
    return dict(weights)


def build_sampling_plan(
    specs: Iterable[SourceSpec],
    max_total_samples: Optional[int] = None,
    language_ratios: Optional[Dict[str, float]] = None,
    language_max_samples: Optional[Dict[str, int]] = None,
) -> SamplingPlan:
    specs = list(specs)
    if not specs:
        raise ValueError("No source specs were provided for sampling.")

    default_weights = _default_language_weights(specs)
    available_languages = set(default_weights)
    if language_ratios:
        active_ratios = {
            language: float(ratio)
            for language, ratio in language_ratios.items()
            if language in available_languages and float(ratio) > 0
        }
        if active_ratios:
            language_weights = active_ratios
        else:
            raise ValueError("None of the provided language ratios match available CPT sources.")
    else:
        language_weights = default_weights

    default_total = sum(spec.requested_max_samples or 0 for spec in specs)
    if default_total <= 0:
        default_total = len(specs)
    total_target_samples = max_total_samples or default_total

    quotas = {language: 0 for language in available_languages}
    fixed_quotas = {
        language: int(value)
        for language, value in (language_max_samples or {}).items()
        if language in available_languages and int(value) > 0
    }

    fixed_total = sum(fixed_quotas.values())
    if fixed_total > total_target_samples:
        raise ValueError(
            f"language_max_samples total {fixed_total} exceeds total target {total_target_samples}."
        )

    quotas.update(fixed_quotas)
    remaining_languages = [language for language in available_languages if language not in fixed_quotas]
    remaining_total = total_target_samples - fixed_total
    if remaining_total > 0 and remaining_languages:
        remainder_weights = {language: language_weights.get(language, 0.0) for language in remaining_languages}
        allocated = _allocate_integer_quotas(remaining_total, remainder_weights)
        for language, value in allocated.items():
            quotas[language] += value

    source_quotas: Dict[str, int] = {}
    specs_by_language: Dict[str, list[SourceSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_language[spec.language].append(spec)

    for language, language_specs in specs_by_language.items():
        language_total = quotas.get(language, 0)
        caps = {
            spec.name: int(spec.requested_max_samples or language_total)
            for spec in language_specs
        }
        weights = {
            spec.name: float(spec.requested_max_samples or 1)
            for spec in language_specs
        }
        source_quotas.update(_allocate_capped_quotas(language_total, weights, caps))

    realized_total = sum(source_quotas.values())
    language_quotas = {
        language: sum(source_quotas[spec.name] for spec in language_specs)
        for language, language_specs in specs_by_language.items()
    }
    return SamplingPlan(
        total_target_samples=realized_total,
        language_quotas=language_quotas,
        source_quotas=source_quotas,
    )
