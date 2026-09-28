"""Small measurement helpers for full paper benchmark evidence."""

from __future__ import annotations

import platform
import resource
from collections.abc import Sequence


def peak_rss_bytes() -> int:
    """Return this process's high-water resident memory in bytes."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def summarize_query_latency(
    samples_ms: Sequence[float], *, warmup_count: int
) -> dict[str, int | float | str | None]:
    """Summarize warm in-process scoring; the initial queries still enter quality metrics."""

    if warmup_count < 0:
        raise ValueError("warmup count must be non-negative")
    retained = sorted(samples_ms[warmup_count:])

    def percentile(fraction: float) -> float | None:
        if not retained:
            return None
        position = fraction * (len(retained) - 1)
        low = int(position)
        high = min(low + 1, len(retained) - 1)
        return retained[low] + (retained[high] - retained[low]) * (position - low)

    return {
        "scope": (
            "warm in-process per-Query pool generation and LHF ranking; "
            "no source loading or training"
        ),
        "concurrency": 1,
        "scored_query_count": len(samples_ms),
        "warmup_count": min(warmup_count, len(samples_ms)),
        "timed_query_count": len(retained),
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }
