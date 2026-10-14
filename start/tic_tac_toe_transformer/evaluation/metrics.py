"""Small reusable metric helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_latencies(milliseconds: Sequence[float]) -> dict[str, float]:
    if not milliseconds:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": sum(milliseconds) / len(milliseconds),
        "p50_ms": percentile(milliseconds, 0.50),
        "p95_ms": percentile(milliseconds, 0.95),
        "max_ms": max(milliseconds),
    }
