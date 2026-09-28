"""Small measurement helpers for full paper benchmark evidence."""

from __future__ import annotations

import platform
import resource
import subprocess
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def run_identity(source: Any) -> dict[str, object]:
    """Record code and prepared-source identity without consuming source rows."""

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    result: dict[str, object] = {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    dependencies: dict[str, str | None] = {}
    for name in ("duckdb", "lightgbm", "numpy", "pyarrow", "scipy", "torch"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    result["dependency_versions"] = dependencies
    path = getattr(source, "path", None)
    if isinstance(path, Path):
        manifest = path.with_name("source.json")
        if manifest.exists():
            import json

            value = json.loads(manifest.read_text(encoding="utf-8"))
            if value.get("archive_sha256") == getattr(source, "dataset_checksum", None):
                result["prepared_source_sha256"] = value.get("ordered_sha256")
                result["archive_md5"] = value.get("archive_md5")
                result["source_rating_count"] = value.get("rating_count")
    return result


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
