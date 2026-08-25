#!/usr/bin/env python3
"""Compare PyTorch and ONNX Runtime on the same model and CPU inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
import torch

from deep_recsys_lab.cli import _checkpoint_model
from deep_recsys_lab.data.dataset import load_prepared_data
from deep_recsys_lab.serving.model_store import load_model_artifact


def _measure(call: Callable[[], Any], iterations: int) -> dict[str, float]:
    latencies: list[float] = []
    started = time.perf_counter()
    for _ in range(iterations):
        call_started = time.perf_counter_ns()
        call()
        latencies.append((time.perf_counter_ns() - call_started) / 1_000_000)
    elapsed = time.perf_counter() - started
    return {
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_inferences_per_second": iterations / elapsed,
    }


def _summarize(rounds: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: round(statistics.median(result[name] for result in rounds), 6) for name in rounds[0]
    }


def _benchmark_inputs(data_dir: Path, batch_size: int) -> np.ndarray:
    data = load_prepared_data(data_dir)
    matrix = data.validation_fold_in if data.validation_fold_in.shape[0] else data.train
    if matrix.shape[0] == 0:
        raise ValueError("prepared data does not contain benchmark rows")
    row_indices = np.arange(batch_size) % matrix.shape[0]
    return np.ascontiguousarray(matrix[row_indices].toarray(), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Record a regression without returning a failing exit status.",
    )
    args = parser.parse_args()
    if min(args.batch_size, args.warmup, args.iterations, args.rounds) < 1:
        parser.error("batch-size, warmup, iterations, and rounds must be positive")

    prepared = load_prepared_data(args.data_dir)
    interactions = _benchmark_inputs(args.data_dir, args.batch_size)
    pytorch_model = _checkpoint_model(args.checkpoint, prepared.n_items, torch.device("cpu")).eval()
    registered = bentoml.models.get(args.model_tag)
    loaded = load_model_artifact(registered.path, model_tag=str(registered.tag))
    tensor = torch.from_numpy(interactions)

    def pytorch_call() -> torch.Tensor:
        with torch.inference_mode():
            return pytorch_model(tensor, sample=False).logits

    def onnx_call() -> np.ndarray:
        return loaded.scorer.score(interactions)

    expected = pytorch_call().numpy()
    actual = onnx_call()
    maximum_absolute_error = float(np.max(np.abs(expected - actual)))
    for _ in range(args.warmup):
        pytorch_call()
        onnx_call()

    measurements: dict[str, list[dict[str, float]]] = {
        "pytorch": [],
        "onnxruntime": [],
    }
    calls: dict[str, Callable[[], Any]] = {
        "pytorch": pytorch_call,
        "onnxruntime": onnx_call,
    }
    for round_index in range(args.rounds):
        order = ("pytorch", "onnxruntime")
        if round_index % 2:
            order = tuple(reversed(order))
        for backend in order:
            measurements[backend].append(_measure(calls[backend], args.iterations))

    pytorch_result = _summarize(measurements["pytorch"])
    onnx_result = _summarize(measurements["onnxruntime"])
    speedup = pytorch_result["p50_latency_ms"] / onnx_result["p50_latency_ms"]
    no_regression = onnx_result["p50_latency_ms"] <= pytorch_result["p50_latency_ms"]
    result = {
        "model_tag": str(registered.tag),
        "n_items": prepared.n_items,
        "batch_size": args.batch_size,
        "warmup_iterations": args.warmup,
        "measured_iterations_per_round": args.iterations,
        "rounds": args.rounds,
        "maximum_absolute_error": round(maximum_absolute_error, 8),
        "pytorch": pytorch_result,
        "onnxruntime": onnx_result,
        "p50_speedup": round(speedup, 6),
        "no_regression": no_regression,
        "gating": not args.report_only,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    if not no_regression and not args.report_only:
        raise SystemExit("ONNX Runtime median p50 regressed against PyTorch")


if __name__ == "__main__":
    main()
