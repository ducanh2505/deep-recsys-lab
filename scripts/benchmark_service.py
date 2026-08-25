#!/usr/bin/env python3
"""Measure Bento HTTP serving and prove that adaptive batching is active."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from threading import Barrier
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

INFERENCE_SERVICE_NAME = "deep_recsys_inference"
INFERENCE_METHOD_NAME = "recommend_batch"
ADAPTIVE_BATCH_METRIC = "bentoml_service_adaptive_batch_size"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _open_connections(url: str, count: int) -> tuple[list[HTTPConnection], str]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("--base-url must be an absolute HTTP(S) URL")
    connection_type = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    connections = [connection_type(parsed.hostname, parsed.port, timeout=60) for _ in range(count)]
    for connection in connections:
        connection.connect()
    target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    return connections, target


def _request(
    connection: HTTPConnection,
    target: str,
    payload: bytes,
    api_key: str | None,
    barrier: Barrier,
) -> tuple[float, int]:
    barrier.wait(timeout=60)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    started = time.perf_counter()
    connection.request("POST", target, body=payload, headers=headers)
    response = connection.getresponse()
    response.read()
    return (time.perf_counter() - started) * 1000, response.status


def _metric_total(metrics: str, suffix: str) -> float:
    metric_name = f"{ADAPTIVE_BATCH_METRIC}_{suffix}"
    total = 0.0
    for line in metrics.splitlines():
        if not line.startswith(f"{metric_name}{{"):
            continue
        labels, separator, sample = line.partition("}")
        if not separator:
            continue
        if f'runner_name="{INFERENCE_SERVICE_NAME}"' not in labels:
            continue
        if f'method_name="{INFERENCE_METHOD_NAME}"' not in labels:
            continue
        total += float(sample.split()[0])
    return total


def _adaptive_batch_metrics(base_url: str) -> tuple[float, float]:
    endpoint = f"{base_url.rstrip('/')}/metrics"
    request = Request(endpoint, method="GET")
    with urlopen(request, timeout=60) as response:  # noqa: S310 - local service URL
        metrics = response.read().decode("utf-8")
        if response.status != 200:
            raise RuntimeError(f"unexpected metrics HTTP status: {response.status}")
    return _metric_total(metrics, "count"), _metric_total(metrics, "sum")


def _wave(
    executor: ThreadPoolExecutor,
    connections: list[HTTPConnection],
    target: str,
    payload: bytes,
    api_key: str | None,
) -> list[tuple[float, int]]:
    barrier = Barrier(len(connections))
    futures = [
        executor.submit(
            _request,
            connection,
            target,
            payload,
            api_key,
            barrier,
        )
        for connection in connections
    ]
    return [future.result() for future in futures]


def _warm_dispatcher(
    executor: ThreadPoolExecutor,
    count: int,
    connections: list[HTTPConnection],
    target: str,
    payload: bytes,
    api_key: str | None,
) -> None:
    """Feed BentoML 1.4's nine singles, batch-two, and batch-three calibration."""

    singles = min(count, 9)
    for _ in range(singles):
        _wave(executor, connections[:1], target, payload, api_key)
    remaining = count - singles
    for training_batch_size in (2, 3):
        if remaining < training_batch_size:
            break
        _wave(
            executor,
            connections[:training_batch_size],
            target,
            payload,
            api_key,
        )
        remaining -= training_batch_size
    while remaining:
        wave_size = min(len(connections), remaining)
        _wave(executor, connections[:wave_size], target, payload, api_key)
        remaining -= wave_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--movie-ids", default="1,2,3,4,5")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=14)
    parser.add_argument("--api-key")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-adaptive-batching",
        action="store_true",
        help="fail unless BentoML reports at least one multi-request inference batch",
    )
    args = parser.parse_args()

    if args.requests < 1:
        parser.error("--requests must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.require_adaptive_batching and args.concurrency < 2:
        parser.error("--require-adaptive-batching needs --concurrency of at least 2")

    movie_ids = [int(value) for value in args.movie_ids.split(",")]
    payload = json.dumps({"movie_ids": movie_ids, "top_k": args.top_k}).encode("utf-8")
    endpoint = f"{args.base_url.rstrip('/')}/recommend"
    connections, target = _open_connections(endpoint, args.concurrency)

    successful_latencies: list[float] = []
    status_counts: dict[int, int] = {}
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            _warm_dispatcher(
                executor,
                args.warmup,
                connections,
                target,
                payload,
                args.api_key,
            )
            started = time.perf_counter()
            for wave_start in range(0, args.requests, args.concurrency):
                wave_size = min(args.concurrency, args.requests - wave_start)
                wave_results = _wave(
                    executor,
                    connections[:wave_size],
                    target,
                    payload,
                    args.api_key,
                )
                for latency, status in wave_results:
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if status == 200:
                        successful_latencies.append(latency)
    finally:
        for connection in connections:
            connection.close()
    elapsed = time.perf_counter() - started
    adaptive_count, adaptive_sum = _adaptive_batch_metrics(args.base_url)
    adaptive_observed = adaptive_count > 0 and adaptive_sum > adaptive_count
    result = {
        "requests": args.requests,
        "successful_requests": len(successful_latencies),
        "failed_requests": args.requests - len(successful_latencies),
        "http_status_counts": {
            str(status): count for status, count in sorted(status_counts.items())
        },
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup,
        "p50_latency_ms": round(_percentile(successful_latencies, 0.50), 3)
        if successful_latencies
        else None,
        "p95_latency_ms": round(_percentile(successful_latencies, 0.95), 3)
        if successful_latencies
        else None,
        "throughput_requests_per_second": round(args.requests / elapsed, 3),
        "adaptive_batch_size_count": adaptive_count,
        "adaptive_batch_size_sum": adaptive_sum,
        "mean_adaptive_batch_size": round(adaptive_sum / adaptive_count, 3)
        if adaptive_count
        else 0.0,
        "adaptive_batching_observed": adaptive_observed,
        "adaptive_batching_gate_required": args.require_adaptive_batching,
        "http_performance_gating": False,
        "http_status_gating": False,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    if args.require_adaptive_batching and not adaptive_observed:
        print(
            "adaptive batching gate failed: expected batch-size sum to exceed count",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
