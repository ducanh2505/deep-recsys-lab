from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer

from .history_only_benchmark import (
    HistoryOnlyBenchmarkConfig,
    run_history_only_benchmark,
    write_history_only_benchmark_report,
)
from .kafka import ConfluentKafkaBoundary
from .lifecycle import run_fast_lifecycle
from .lightgcn import LightGCNConfig
from .movielens import prepare_movielens_20m
from .multivae import MultVAEConfig
from .paper_known_user_fusion import (
    KnownUserHybridConfig,
    run_known_user_hybrid_benchmark,
    write_known_user_hybrid_report,
)
from .rolling import run_rolling_lifecycle

app = typer.Typer(add_completion=False, help="Movie recommender lifecycle commands.")


def _broker_is_reachable(bootstrap_servers: str) -> bool:
    host, port_text = bootstrap_servers.rsplit(":", maxsplit=1)
    try:
        with socket.create_connection((host, int(port_text)), timeout=0.5):
            return True
    except (OSError, ValueError):
        return False


def _ensure_kafka(bootstrap_servers: str) -> None:
    if _broker_is_reachable(bootstrap_servers):
        return
    try:
        subprocess.run(["docker", "compose", "up", "-d", "kafka"], check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise typer.BadParameter(
            "Kafka is unreachable and Docker Compose could not start the official broker"
        ) from error

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if _broker_is_reachable(bootstrap_servers):
            return
        time.sleep(0.5)
    raise typer.BadParameter(f"Kafka did not become reachable at {bootstrap_servers}")


@app.command("fast")
def fast(
    output: Annotated[Path, typer.Option(help="Fresh output directory.")] = Path("artifacts/fast"),
    bootstrap_servers: Annotated[
        str, typer.Option(help="Kafka bootstrap address.")
    ] = "localhost:9092",
) -> None:
    """Run the fixture → Kafka → artifact → FastAPI-ready report path."""

    _ensure_kafka(bootstrap_servers)
    result = run_fast_lifecycle(
        output_dir=output,
        kafka=ConfluentKafkaBoundary(bootstrap_servers),
    )
    typer.echo(f"Serving Artifact: {result.artifact_path}")
    typer.echo(f"Active pointer: {result.active_pointer_path}")
    typer.echo(f"Static report: {result.report_path}")


@app.command("rolling")
def rolling(
    output: Annotated[Path, typer.Option(help="Rolling output/checkpoint directory.")] = Path(
        "artifacts/rolling"
    ),
    bootstrap_servers: Annotated[
        str, typer.Option(help="Kafka bootstrap address.")
    ] = "localhost:9092",
) -> None:
    """Run the rolling lifecycle, portfolio report, and three-mode warm CPU benchmark."""

    _ensure_kafka(bootstrap_servers)
    result = run_rolling_lifecycle(
        output_dir=output,
        kafka=ConfluentKafkaBoundary(bootstrap_servers),
    )
    typer.echo(f"Active 100% Serving Artifact: {result.artifact_path}")
    typer.echo(f"Active pointer: {result.active_pointer_path}")
    typer.echo(f"Latency benchmark: {result.latency_benchmark_path}")
    typer.echo(f"Static report: {result.report_path}")


@app.command("full")
def full(
    output: Annotated[Path, typer.Option(help="Full lifecycle output directory.")] = Path(
        "artifacts/full"
    ),
    cache: Annotated[Path, typer.Option(help="Ignored MovieLens cache directory.")] = Path(
        "var/datasets"
    ),
    bootstrap_servers: Annotated[
        str, typer.Option(help="Kafka bootstrap address.")
    ] = "localhost:9092",
) -> None:
    """Run all MovieLens 20M ratings through the six-stage lifecycle."""

    source = prepare_movielens_20m(cache)
    _ensure_kafka(bootstrap_servers)
    result = run_rolling_lifecycle(
        output_dir=output,
        kafka=ConfluentKafkaBoundary(bootstrap_servers),
        source_events=source,
        evaluation_cohort_limit=5_000,
        fusion_negative_rows_per_query=20,
        multivae_config=MultVAEConfig(epochs=1, batch_size=256),
        multivae_device_preference="auto",
        lightgcn_config=LightGCNConfig(epochs=1, batch_size=65_536),
        lightgcn_device_preference="auto",
        phase_observer=lambda phase, percentage: typer.echo(
            f"full stage {percentage}%: {phase}", err=True
        ),
    )
    typer.echo(f"Active 100% Serving Artifact: {result.artifact_path}")
    typer.echo(f"Latency benchmark: {result.latency_benchmark_path}")
    typer.echo(f"Static report: {result.report_path}")


@app.command("known-user-benchmark")
def known_user_benchmark(
    output: Annotated[
        Path, typer.Option(help="Path for the structured Known-User benchmark report.")
    ] = Path("artifacts/paper-known-user/report.json"),
    cache: Annotated[Path, typer.Option(help="Ignored MovieLens cache directory.")] = Path(
        "var/datasets"
    ),
    seed: Annotated[int, typer.Option(help="Deterministic per-Subject split seed.")] = 42,
    inner_seed: Annotated[
        int | None, typer.Option(help="Deterministic inner interaction fold seed.")
    ] = None,
    inner_fold_count: Annotated[
        int, typer.Option(help="Disjoint 10% inner interaction folds (1 to 5).")
    ] = 1,
    pool_limit: Annotated[
        int, typer.Option(help="Candidates per retriever before fusion (1 to 200).")
    ] = 200,
) -> None:
    """Fit the four-retriever Known-User hybrid and report validation only."""

    source = prepare_movielens_20m(cache)
    benchmark = run_known_user_hybrid_benchmark(
        source,
        config=KnownUserHybridConfig(
            seed=seed,
            inner_seed=inner_seed,
            inner_fold_count=inner_fold_count,
            pool_limit=pool_limit,
        ),
    )
    report_path = write_known_user_hybrid_report(benchmark, output)
    typer.echo(f"Known-User validation report: {report_path}")


@app.command("history-only-benchmark")
def history_only_benchmark(
    output: Annotated[
        Path, typer.Option(help="Path for the structured History-Only benchmark report.")
    ] = Path("artifacts/paper-history-only/report.json"),
    cache: Annotated[Path, typer.Option(help="Ignored MovieLens cache directory.")] = Path(
        "var/datasets"
    ),
    seed: Annotated[int, typer.Option(help="Deterministic Subject and fold-in split seed.")] = 42,
    validation_subject_count: Annotated[
        int,
        typer.Option(help="Disjoint validation Subject count; MovieLens 20M default is 10,000."),
    ] = 10_000,
    test_subject_count: Annotated[
        int, typer.Option(help="Disjoint test Subject count; MovieLens 20M default is 10,000.")
    ] = 10_000,
    evaluate_test: Annotated[
        bool, typer.Option(help="Reveal test metrics only after the configuration is frozen.")
    ] = False,
) -> None:
    """Run the separate Mult-VAE History-Only paper-style benchmark."""

    config = HistoryOnlyBenchmarkConfig(
        seed=seed,
        validation_subject_count=validation_subject_count,
        test_subject_count=test_subject_count,
    )
    source = prepare_movielens_20m(cache)
    benchmark = run_history_only_benchmark(source, config=config, evaluate_test=evaluate_test)
    report_path = write_history_only_benchmark_report(benchmark, output)
    typer.echo(f"History-Only benchmark report: {report_path}")


@app.command("serve")
def serve(
    artifact: Annotated[
        Path, typer.Option(help="Artifact store or immutable Serving Artifact directory.")
    ],
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Serve a previously exported artifact through FastAPI."""

    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(artifact), host=host, port=port)


def main() -> None:
    app()
