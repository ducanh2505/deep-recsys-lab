"""Generic train, evaluate, and combined pipeline workflows."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recsys.artifacts import LoadedArtifact, load_artifact, write_artifact
from recsys.conf.schema import PlatformConfig
from recsys.core.hashing import digest_value
from recsys.core.io import write_json
from recsys.core.paths import WorkspacePaths
from recsys.core.provenance import capture_provenance
from recsys.core.types import QueryMode
from recsys.datasets import PreparedDataset, ordered_train_events, prepare_from_config
from recsys.datasets.schema import HistoryInteraction
from recsys.datasets.store import load_dataset
from recsys.evaluation import ranking_metrics
from recsys.hybrid import fit_hybrid
from recsys.models import fit_model
from recsys.models.base import TrainingContext
from recsys.serving import RecommendationEngine, RecommendationRequest

from .state import RunState


def create_run_dir(paths: WorkspacePaths) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = paths.runs / f"{stamp}-{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def record_run_context(config: PlatformConfig, paths: WorkspacePaths, run_dir: Path) -> None:
    write_json(run_dir / "config.json", asdict(config))
    write_json(run_dir / "provenance.json", capture_provenance(paths.root))


def train_artifact(
    config: PlatformConfig,
    data: PreparedDataset,
    paths: WorkspacePaths,
    run_dir: Path,
) -> LoadedArtifact:
    scratch = run_dir / "scratch"
    if config.model.name == "hybrid":
        spec = fit_hybrid(
            data,
            config.retrieval,
            config.fusion,
            config.reranker,
            config.training,
            scratch,
        )
    else:
        spec = fit_model(
            config.model.name,
            TrainingContext(data, config.training, config.model.parameters, scratch),
        )
    artifact_config = {
        "model": asdict(config.model),
        "training": asdict(config.training),
        "retrieval": asdict(config.retrieval) if config.model.name == "hybrid" else None,
        "fusion": asdict(config.fusion) if config.model.name == "hybrid" else None,
        "reranker": asdict(config.reranker) if config.model.name == "hybrid" else None,
    }
    artifact = write_artifact(paths.models, spec, data, artifact_config)
    write_json(
        run_dir / "artifact.json",
        {"artifact_id": artifact.manifest.artifact_id, "path": artifact.root},
    )
    return artifact


def evaluate_artifact(
    artifact: Path | LoadedArtifact,
    data: PreparedDataset,
    *,
    top_k: int,
) -> dict[str, float | int]:
    loaded = load_artifact(artifact) if isinstance(artifact, Path) else artifact
    if loaded.manifest.dataset_digest != data.dataset_digest:
        raise ValueError("artifact and evaluation dataset digests differ")
    engine = RecommendationEngine(loaded)
    rankings: list[list[int]] = []
    truths: list[set[int]] = []
    item_lookup = data.item_index
    history_by_user: dict[int, list[HistoryInteraction]] = {}
    for row in ordered_train_events(data).itertuples(index=False):
        timestamp = getattr(row, "timestamp", None)
        if timestamp is not None and timestamp != timestamp:
            timestamp = None
        history_by_user.setdefault(int(row.recsys_user_index), []).append(
            HistoryInteraction(
                item_id=row.item_id,
                value=float(row.value),
                timestamp=timestamp,
            )
        )
    for user_index, user_id in enumerate(data.user_ids):
        relevant = set(
            data.test.indices[
                data.test.indptr[user_index] : data.test.indptr[user_index + 1]
            ].tolist()
        )
        if not relevant:
            continue
        if QueryMode.KNOWN_USER in loaded.manifest.capabilities:
            request = RecommendationRequest(user_id=user_id, top_k=top_k)
        else:
            request = RecommendationRequest(
                interactions=history_by_user[user_index],
                top_k=top_k,
            )
        response = engine.recommend(request)
        rankings.append([item_lookup[value.item_id] for value in response.recommendations])
        truths.append(relevant)
    return ranking_metrics(rankings, truths, k=top_k)


def run_pipeline(config: PlatformConfig, *, run_dir: Path | None = None) -> dict[str, Any]:
    paths = WorkspacePaths.from_value(config.workspace_root).ensure()
    run_dir = create_run_dir(paths) if run_dir is None else run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState.open(run_dir / "state.json")
    config_digest = digest_value(asdict(config))
    recorded_digest = state.value.get("config_digest")
    if recorded_digest is not None and recorded_digest != config_digest:
        raise ValueError("run directory was created with a different normalized configuration")
    state.value["config_digest"] = config_digest
    record_run_context(config, paths, run_dir)
    data = state.stage(
        "prepare",
        lambda: prepare_from_config(config.dataset, paths),
        encode=lambda value: {"path": value.root, "dataset_digest": value.dataset_digest},
        decode=lambda value: load_dataset(Path(value["path"])),
    )
    artifact = state.stage(
        "train",
        lambda: train_artifact(config, data, paths, run_dir),
        encode=lambda value: {"path": value.root, "artifact_id": value.manifest.artifact_id},
        decode=lambda value: load_artifact(Path(value["path"])),
    )
    metrics = state.stage(
        "evaluate",
        lambda: evaluate_artifact(artifact, data, top_k=config.evaluation.top_k),
        encode=lambda value: value,
        decode=lambda value: dict(value),
    )
    write_json(run_dir / "metrics.json", metrics)
    return {
        "run": run_dir,
        "dataset_digest": data.dataset_digest,
        "artifact_id": artifact.manifest.artifact_id,
        "artifact": artifact.root,
        "metrics": metrics,
    }
