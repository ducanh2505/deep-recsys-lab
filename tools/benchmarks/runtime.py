"""Measure scoring latency for an immutable artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from recsys.artifacts import load_artifact
from recsys.artifacts.runtimes import RuntimeQuery, runtime_from_artifact
from recsys.core.types import QueryMode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    artifact = load_artifact(args.artifact)
    runtime = runtime_from_artifact(artifact)
    mode = artifact.manifest.capabilities[0]
    vector = np.zeros(len(artifact.item_ids), dtype=np.float32)
    user_index = 0 if mode is QueryMode.KNOWN_USER else None
    if mode is QueryMode.HISTORY:
        vector[0] = 1.0
    query = RuntimeQuery(mode, vector, (0,), user_index)

    for _ in range(3):
        runtime.score(query)
    samples: list[float] = []
    for _ in range(args.iterations):
        started = perf_counter()
        runtime.score(query)
        samples.append((perf_counter() - started) * 1000)

    print(
        json.dumps(
            {
                "artifact_id": artifact.manifest.artifact_id,
                "iterations": args.iterations,
                "latency_ms": {
                    "mean": float(np.mean(samples)),
                    "p50": float(np.percentile(samples, 50)),
                    "p95": float(np.percentile(samples, 95)),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
