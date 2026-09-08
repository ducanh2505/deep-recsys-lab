"""The ``recsys`` command-line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from recsys.artifacts import load_artifact
from recsys.artifacts.runtimes import RUNTIME_REGISTRY
from recsys.conf.schema import PlatformConfig, load_config
from recsys.core.hashing import normalize
from recsys.core.io import write_json
from recsys.core.paths import WorkspacePaths
from recsys.datasets import prepare_from_config
from recsys.datasets.store import load_dataset
from recsys.experiments import evaluate_artifact, run_pipeline, train_artifact
from recsys.experiments.runner import create_run_dir, record_run_context
from recsys.fusion import FUSION_REGISTRY
from recsys.models import MODEL_REGISTRY
from recsys.reranking import RERANKER_REGISTRY
from recsys.retrieval import RETRIEVER_REGISTRY


@dataclass(slots=True)
class CLIState:
    workspace_root: Path
    config_path: Path | None
    overrides: list[str]


app = typer.Typer(help="Train, evaluate, compose, and serve recommendation systems.")
data_app = typer.Typer(help="Prepare and inspect interaction datasets.")
artifact_app = typer.Typer(help="Inspect and verify immutable model artifacts.")
plugins_app = typer.Typer(help="Discover built-in extension points.")
app.add_typer(data_app, name="data")
app.add_typer(artifact_app, name="artifact")
app.add_typer(plugins_app, name="plugins")


def _state(context: typer.Context) -> CLIState:
    value = context.find_root().obj
    if not isinstance(value, CLIState):
        raise RuntimeError("CLI state is unavailable")
    return value


def _config(context: typer.Context, extra: list[str] | None = None) -> PlatformConfig:
    state = _state(context)
    return load_config(
        external=state.config_path,
        overrides=[*state.overrides, *(extra or [])],
        workspace_root=state.workspace_root,
    )


def _print(value: Any) -> None:
    typer.echo(json.dumps(normalize(value), ensure_ascii=False, indent=2, sort_keys=True))


@app.callback()
def main(
    context: typer.Context,
    workspace_root: Annotated[
        Path,
        typer.Option("--workspace-root", help="Root for configuration and generated var/ state."),
    ] = Path.cwd(),
    config_path: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False, help="Optional external YAML layer."),
    ] = None,
    override: Annotated[
        list[str] | None,
        typer.Option("--set", "-o", help="Hydra dotlist override; may be repeated."),
    ] = None,
) -> None:
    context.obj = CLIState(workspace_root.resolve(), config_path, override or [])


@data_app.command(
    "prepare", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def data_prepare(context: typer.Context) -> None:
    """Validate and materialize a deterministic prepared dataset."""

    config = _config(context, list(context.args))
    paths = WorkspacePaths.from_value(config.workspace_root).ensure()
    data = prepare_from_config(config.dataset, paths)
    _print({"dataset_digest": data.dataset_digest, "path": data.root, "shape": data.shape})


@data_app.command("inspect")
def data_inspect(path: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Verify and summarize an existing prepared dataset."""

    data = load_dataset(path)
    _print(data.manifest)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(
    context: typer.Context,
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", exists=True, file_okay=False, help="Prepared dataset directory."),
    ] = None,
) -> None:
    """Train one configured model plugin and write an immutable artifact."""

    config = _config(context, list(context.args))
    paths = WorkspacePaths.from_value(config.workspace_root).ensure()
    data = (
        load_dataset(dataset) if dataset is not None else prepare_from_config(config.dataset, paths)
    )
    run_dir = create_run_dir(paths)
    record_run_context(config, paths, run_dir)
    artifact = train_artifact(config, data, paths, run_dir)
    _print({"artifact_id": artifact.manifest.artifact_id, "path": artifact.root, "run": run_dir})


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def evaluate(
    context: typer.Context,
    artifact: Annotated[Path, typer.Option("--artifact", exists=True, file_okay=False)],
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", exists=True, file_okay=False, help="Prepared dataset directory."),
    ] = None,
) -> None:
    """Evaluate an artifact against a prepared or configured dataset."""

    config = _config(context, list(context.args))
    paths = WorkspacePaths.from_value(config.workspace_root).ensure()
    data = (
        load_dataset(dataset) if dataset is not None else prepare_from_config(config.dataset, paths)
    )
    loaded = load_artifact(artifact)
    run_dir = create_run_dir(paths)
    record_run_context(config, paths, run_dir)
    metrics = evaluate_artifact(loaded, data, top_k=config.evaluation.top_k)
    write_json(
        run_dir / "artifact.json",
        {"artifact_id": loaded.manifest.artifact_id, "path": loaded.root},
    )
    write_json(run_dir / "metrics.json", metrics)
    _print({"artifact_id": loaded.manifest.artifact_id, "metrics": metrics, "run": run_dir})


@app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run_command(
    context: typer.Context,
    run_dir: Annotated[
        Path | None,
        typer.Option(help="Existing or new run directory; completed stages are resumed."),
    ] = None,
) -> None:
    """Prepare, train, and evaluate with run-local state and provenance."""

    _print(run_pipeline(_config(context, list(context.args)), run_dir=run_dir))


@artifact_app.command("inspect")
def artifact_inspect(path: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Verify and print an artifact manifest."""

    artifact = load_artifact(path)
    _print(artifact.manifest.model_dump(mode="json", by_alias=True))


@artifact_app.command("verify")
def artifact_verify(path: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Fail closed unless all artifact checksums and contracts are valid."""

    artifact = load_artifact(path)
    _print({"artifact_id": artifact.manifest.artifact_id, "status": "verified"})


@plugins_app.command("list")
def plugins_list() -> None:
    """List built-in models, retrievers, fusion strategies, and runtimes."""

    _print(
        {
            "models": MODEL_REGISTRY.names(),
            "retrievers": RETRIEVER_REGISTRY.names(),
            "fusion": FUSION_REGISTRY.names(),
            "runtimes": RUNTIME_REGISTRY.names(),
            "rerankers": RERANKER_REGISTRY.names(),
        }
    )


@app.command()
def serve(
    context: typer.Context,
    artifact: Annotated[Path, typer.Option("--artifact", exists=True, file_okay=False)],
    host: Annotated[str | None, typer.Option(help="Bento bind host.")] = None,
    port: Annotated[int | None, typer.Option(help="Bento bind port.")] = None,
) -> None:
    """Start the optional Bento adapter from an explicit artifact path."""

    config = _config(context)
    load_artifact(artifact)
    environment = os.environ.copy()
    environment["RECSYS_ARTIFACT"] = str(artifact.resolve())
    environment["RECSYS_MAX_BODY_BYTES"] = str(config.serving.max_body_bytes)
    command = [
        sys.executable,
        "-m",
        "bentoml",
        "serve",
        "recsys.serving.bento:RecommendationService",
        "--host",
        host or config.serving.host,
        "--port",
        str(port or config.serving.port),
    ]
    raise typer.Exit(subprocess.run(command, env=environment, check=False).returncode)


if __name__ == "__main__":  # pragma: no cover
    app()
