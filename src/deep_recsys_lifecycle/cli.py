from __future__ import annotations

import socket
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Annotated, cast

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
from .paper_fit_cache import FitCache
from .paper_known_user_fusion import (
    KnownUserHybridConfig,
    run_known_user_hybrid_benchmark,
    write_known_user_hybrid_report,
)
from .paper_pool_cache import PoolCache
from .paper_screening import PaperScreenWorkspace, ScreenMode, default_screen_runner
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
) -> None:
    """Fit the three-retriever History-Only hybrid and report validation only."""

    config = HistoryOnlyBenchmarkConfig(
        seed=seed,
        validation_subject_count=validation_subject_count,
        test_subject_count=test_subject_count,
    )
    source = prepare_movielens_20m(cache)
    benchmark = run_history_only_benchmark(source, config=config, evaluate_test=False)
    report_path = write_history_only_benchmark_report(benchmark, output)
    typer.echo(f"History-Only benchmark report: {report_path}")


def _paper_mode(value: str) -> ScreenMode:
    if value not in {"known_user", "history_only"}:
        raise typer.BadParameter("mode must be known_user or history_only")
    return cast(ScreenMode, value)


@app.command("paper-screen-plan")
def paper_screen_plan(
    root: Annotated[Path, typer.Option(help="New write-once screening directory.")],
    source_sha256: Annotated[str, typer.Option(help="Exact MovieLens source checksum.")],
) -> None:
    """Preregister the validation-only one-factor plan before opening outcomes."""

    workspace = PaperScreenWorkspace.create(root, source_sha256=source_sha256)
    typer.echo(f"Screen plan: {workspace.root / 'plan.json'}")


@app.command("paper-screen-baseline")
def paper_screen_baseline(
    root: Annotated[Path, typer.Option(help="Existing screening directory.")],
    mode: Annotated[str, typer.Option(help="known_user or history_only")],
) -> None:
    """Register a mode's fixed baseline before running it."""

    run_id = PaperScreenWorkspace(root).register_baseline(_paper_mode(mode))
    typer.echo(f"Registered {run_id}")


@app.command("paper-screen-variant")
def paper_screen_variant(
    root: Annotated[Path, typer.Option(help="Existing screening directory.")],
    mode: Annotated[str, typer.Option(help="known_user or history_only")],
    axis: Annotated[str, typer.Option(help="Axis name from plan.json.")],
    alternative: Annotated[int, typer.Option(help="Preregistered alternative: 0 or 1.")],
    reference: Annotated[str, typer.Option(help="Completed same-mode reference run ID.")],
    run_id: Annotated[str, typer.Option(help="Unique run ID.")],
) -> None:
    """Register one exact factor value before viewing its validation outcome."""

    workspace = PaperScreenWorkspace(root)
    resolved_mode = _paper_mode(mode)
    configuration = workspace.planned_configuration(
        mode=resolved_mode,
        axis_name=axis,
        alternative_index=alternative,
        reference_run_id=reference,
    )
    workspace.register_variant(
        run_id=run_id,
        mode=resolved_mode,
        axis_name=axis,
        alternative_index=alternative,
        reference_run_id=reference,
        configuration=configuration,
    )
    typer.echo(f"Registered {run_id}")


@app.command("paper-screen-run")
def paper_screen_run(
    root: Annotated[Path, typer.Option(help="Existing screening directory.")],
    run_id: Annotated[str, typer.Option(help="Registered run ID.")],
    cache: Annotated[Path, typer.Option(help="MovieLens dataset directory.")] = Path(
        "var/datasets"
    ),
    reference_report: Annotated[
        Path | None, typer.Option(help="Archived #47 report, required for a full-data baseline.")
    ] = None,
) -> None:
    """Run the complete training pipeline and record validation evidence only."""

    workspace = PaperScreenWorkspace(root)
    source = prepare_movielens_20m(cache)
    record = workspace.run_registered(
        run_id,
        source,
        runner=partial(
            default_screen_runner,
            fit_cache=FitCache(root / "fit-cache"),
            pool_cache=PoolCache(root / "pool-cache"),
        ),
        reference_report_path=reference_report,
    )
    typer.echo(f"Validation run: {workspace.root / 'runs' / (str(record['run_id']) + '.json')}")


@app.command("paper-screen-skip")
def paper_screen_skip(
    root: Annotated[Path, typer.Option(help="Existing screening directory.")],
    mode: Annotated[str, typer.Option(help="known_user or history_only")],
    axis: Annotated[str, typer.Option(help="Preregistered axis name.")],
    reason: Annotated[str, typer.Option(help="Measured-loss reason for skipping.")],
) -> None:
    """Record why an axis was not run before freezing the selection."""

    PaperScreenWorkspace(root).skip_axis(_paper_mode(mode), axis, reason)
    typer.echo(f"Skipped {mode}/{axis}")


@app.command("paper-screen-freeze")
def paper_screen_freeze(
    root: Annotated[Path, typer.Option(help="Existing screening directory.")],
) -> None:
    """Select by paired validation evidence and durably freeze both modes."""

    workspace = PaperScreenWorkspace(root)
    workspace.select_and_freeze()
    typer.echo(f"Selection freeze: {workspace.root / 'freeze.json'}")


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
