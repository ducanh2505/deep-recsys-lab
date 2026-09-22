from __future__ import annotations

import asyncio
import json
import math
import os
import platform
import statistics
import subprocess
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter_ns
from typing import Any, Literal

from .api import create_app
from .artifact import ServingArtifact

LatencyMode = Literal["known_user", "history_only", "empty_history"]
LATENCY_MODES: tuple[LatencyMode, ...] = ("known_user", "history_only", "empty_history")
SLO_P95_MS: dict[LatencyMode, float] = {
    "known_user": 200.0,
    "history_only": 200.0,
    "empty_history": 20.0,
}
DEFAULT_WARMUP_COUNT = 5
DEFAULT_SAMPLE_COUNT = 30
PERCENTILE_METHOD = "statistics.quantiles(method='inclusive')"


def summarize_latency_samples(
    durations_seconds: Sequence[float], *, slo_target_p95_ms: float
) -> dict[str, object]:
    """Summarize successful request durations without inventing missing measurements.

    Percentiles use linear interpolation through ``statistics.quantiles`` with the inclusive
    method. Throughput uses only the sum of successful request durations, as required by the
    portfolio benchmark contract; failed requests are never treated as zero-duration samples.
    """

    if not math.isfinite(slo_target_p95_ms) or slo_target_p95_ms <= 0:
        raise ValueError("slo_target_p95_ms must be a positive finite number")
    samples = tuple(float(value) for value in durations_seconds)
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("latency samples must be finite and non-negative")

    if not samples:
        return {
            "sample_count": 0,
            "successful_sample_count": 0,
            "duration_total_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "throughput_rps": None,
            "slo_target_p95_ms": float(slo_target_p95_ms),
            "slo_status": "not_measured",
            "percentile_method": PERCENTILE_METHOD,
        }

    ordered = tuple(sorted(samples))
    if len(ordered) == 1:
        p50 = p95 = p99 = ordered[0]
    else:
        percentiles = statistics.quantiles(ordered, n=100, method="inclusive")
        p50 = percentiles[49]
        p95 = percentiles[94]
        p99 = percentiles[98]
    total_seconds = sum(ordered)
    throughput = len(ordered) / total_seconds if total_seconds > 0 else None
    return {
        "sample_count": len(ordered),
        "successful_sample_count": len(ordered),
        "duration_total_ms": _round_ms(total_seconds * 1000.0),
        "p50_ms": _round_ms(p50 * 1000.0),
        "p95_ms": _round_ms(p95 * 1000.0),
        "p99_ms": _round_ms(p99 * 1000.0),
        "throughput_rps": _round_number(throughput),
        "slo_target_p95_ms": float(slo_target_p95_ms),
        "slo_status": "pass" if p95 * 1000.0 <= slo_target_p95_ms else "fail",
        "percentile_method": PERCENTILE_METHOD,
    }


def detect_runtime_environment() -> dict[str, object]:
    """Return runtime-confirmed platform evidence for the report provenance."""

    uname = platform.uname()
    return {
        "os": platform.platform(),
        "os_name": uname.system,
        "os_release": uname.release,
        "architecture": platform.machine(),
        "cpu": _cpu_model(uname.system) or platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def run_latency_benchmark(
    artifact: ServingArtifact,
    *,
    warmup_count: int = DEFAULT_WARMUP_COUNT,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    clock_ns: Callable[[], int] = perf_counter_ns,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Measure the loaded FastAPI request path for each query mode at concurrency one.

    The active artifact is passed in already loaded. The app is invoked through its ASGI
    interface in the current process; no TCP socket is opened, so these numbers are explicitly
    in-process serving latency rather than network latency. App construction, artifact loading,
    and event-loop creation happen before warm-up and sample timing.
    """

    if isinstance(warmup_count, bool) or not isinstance(warmup_count, int) or warmup_count < 0:
        raise ValueError("warmup_count must be a non-negative integer")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")

    record: dict[str, object] = {
        "schema_version": 1,
        "status": "not_measured",
        "artifact_id": artifact.artifact_id,
        "artifact_cutoff_percentage": artifact.manifest.get("data_cutoff_percentage"),
        "artifact_loaded_before_measurement": True,
        "serving_device": artifact.manifest.get("serving_device", "cpu"),
        "top_n": 10,
        "concurrency": 1,
        "warmup_count": warmup_count,
        "requested_sample_count": sample_count,
        "percentile_method": PERCENTILE_METHOD,
        "method": (
            "in-process ASGI invocation through FastAPI; artifact loaded before timing; "
            "no TCP socket, so this is not network latency"
        ),
        "environment": dict(environment)
        if environment is not None
        else detect_runtime_environment(),
        "modes": {},
    }

    modes = record["modes"]
    if not isinstance(modes, dict):  # pragma: no cover - record is constructed above
        raise RuntimeError("benchmark record modes must be mutable")

    try:
        app = create_app(artifact)
        loop = asyncio.new_event_loop()
    except Exception as error:
        reason = f"benchmark setup failed: {type(error).__name__}: {error}"
        for mode in LATENCY_MODES:
            modes[mode] = _unmeasured_mode(
                mode,
                warmup_count=warmup_count,
                requested_sample_count=sample_count,
                reason=reason,
            )
        record["status"] = "failed"
        return record

    try:
        requests = _benchmark_requests(artifact)
        for mode in LATENCY_MODES:
            modes[mode] = _measure_mode(
                app=app,
                loop=loop,
                mode=mode,
                request_body=requests[mode],
                warmup_count=warmup_count,
                sample_count=sample_count,
                clock_ns=clock_ns,
            )
    except Exception as error:
        reason = f"benchmark setup failed: {type(error).__name__}: {error}"
        for mode in LATENCY_MODES:
            if mode not in modes:
                modes[mode] = _unmeasured_mode(
                    mode,
                    warmup_count=warmup_count,
                    requested_sample_count=sample_count,
                    reason=reason,
                )
    finally:
        loop.close()

    mode_values = [value for value in modes.values() if isinstance(value, Mapping)]
    if any(value.get("measurement_status") == "failed" for value in mode_values):
        record["status"] = "completed_with_errors"
    elif any(value.get("failed_sample_count", 0) for value in mode_values):
        record["status"] = "completed_with_errors"
    elif any(value.get("slo_status") == "fail" for value in mode_values):
        record["status"] = "completed_with_slo_failures"
    elif all(value.get("slo_status") == "pass" for value in mode_values):
        record["status"] = "completed"
    else:
        record["status"] = "completed_with_missing_metrics"
    return record


def _measure_mode(
    *,
    app: Any,
    loop: asyncio.AbstractEventLoop,
    mode: LatencyMode,
    request_body: Mapping[str, object],
    warmup_count: int,
    sample_count: int,
    clock_ns: Callable[[], int],
) -> dict[str, object]:
    warmup_success_count = 0
    warmup_error: str | None = None
    for _ in range(warmup_count):
        ok, error = _invoke_request(loop, app, mode, request_body)
        if not ok:
            warmup_error = error
            break
        warmup_success_count += 1

    if warmup_error is not None:
        result = _unmeasured_mode(
            mode,
            warmup_count=warmup_count,
            requested_sample_count=sample_count,
            reason=f"warm-up failed: {warmup_error}",
        )
        result["warmup_success_count"] = warmup_success_count
        return result

    durations: list[float] = []
    errors: list[str] = []
    for _ in range(sample_count):
        started = clock_ns()
        ok, error = _invoke_request(loop, app, mode, request_body)
        elapsed = (clock_ns() - started) / 1_000_000_000.0
        if ok:
            durations.append(elapsed)
        else:
            errors.append(error)

    result = summarize_latency_samples(durations, slo_target_p95_ms=SLO_P95_MS[mode])
    result.update(
        {
            "measurement_status": "measured" if durations else "failed",
            "warmup_count": warmup_count,
            "warmup_success_count": warmup_success_count,
            "requested_sample_count": sample_count,
            "failed_sample_count": len(errors),
            "reason": _sample_failure_reason(errors),
        }
    )
    return result


def _unmeasured_mode(
    mode: LatencyMode,
    *,
    warmup_count: int,
    requested_sample_count: int,
    reason: str,
) -> dict[str, object]:
    result = summarize_latency_samples((), slo_target_p95_ms=SLO_P95_MS[mode])
    result.update(
        {
            "measurement_status": "failed" if reason else "not_measured",
            "warmup_count": warmup_count,
            "warmup_success_count": 0,
            "requested_sample_count": requested_sample_count,
            "failed_sample_count": 0,
            "reason": reason or "no successful samples",
        }
    )
    return result


def _benchmark_requests(artifact: ServingArtifact) -> dict[LatencyMode, dict[str, object]]:
    subject_ids = sorted(artifact.model.subject_histories)
    if not subject_ids:
        raise ValueError("active artifact has no Known-User Subject history for benchmarking")
    known_subject = subject_ids[0]
    histories = tuple(
        history for _, history in sorted(artifact.model.subject_histories.items()) if history
    )
    history = histories[0] if histories else artifact.model.catalog[:1]
    if not history:
        raise ValueError("active artifact has no Movie history for History-Only benchmarking")
    return {
        "known_user": {"subject_id": known_subject, "top_n": 10},
        "history_only": {"history": list(history), "top_n": 10},
        "empty_history": {"top_n": 10},
    }


def _invoke_request(
    loop: asyncio.AbstractEventLoop,
    app: Any,
    mode: LatencyMode,
    request_body: Mapping[str, object],
) -> tuple[bool, str]:
    try:
        status, body = loop.run_until_complete(_asgi_post(app, request_body))
    except Exception as error:
        return False, f"exception {type(error).__name__}"
    if status != 200:
        return False, f"HTTP {status} response"
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, "response was not JSON"
    if not isinstance(payload, Mapping) or payload.get("query_mode") != mode:
        return False, "response query mode did not match the benchmark mode"
    return True, ""


async def _asgi_post(app: Any, request_body: Mapping[str, object]) -> tuple[int, bytes]:
    body = json.dumps(request_body, separators=(",", ":")).encode("utf-8")
    received = False
    status = 500
    response_body: list[bytes] = []

    async def receive() -> dict[str, object]:
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: Mapping[str, object]) -> None:
        nonlocal status
        message_type = message.get("type")
        if message_type == "http.response.start":
            raw_status = message.get("status")
            if isinstance(raw_status, int):
                status = raw_status
        elif message_type == "http.response.body":
            raw_body = message.get("body", b"")
            if isinstance(raw_body, bytes):
                response_body.append(raw_body)

    scope: dict[str, object] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/recommendations",
        "raw_path": b"/recommendations",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"benchmark"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 80),
    }
    await app(scope, receive, send)
    return status, b"".join(response_body)


def _sample_failure_reason(errors: Sequence[str]) -> str | None:
    if not errors:
        return None
    counts: dict[str, int] = {}
    for error in errors:
        counts[error] = counts.get(error, 0) + 1
    return "; ".join(f"{count}× {error}" for error, count in sorted(counts.items()))


def _cpu_model(system: str) -> str | None:
    if system == "Darwin":
        return _run_system_command(("sysctl", "-n", "machdep.cpu.brand_string"))
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if ":" in line and line.split(":", 1)[0].strip().lower() in {
                        "model name",
                        "hardware",
                    }:
                        return line.split(":", 1)[1].strip() or None
        except OSError:
            return None
    return None


def _run_system_command(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=0.5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _round_ms(value: float) -> float:
    return round(float(value), 6)


def _round_number(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None
