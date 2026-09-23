from __future__ import annotations

import gc
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any
from uuid import uuid4

from .artifact import (
    ArtifactStore,
    ServingArtifact,
    canonical_json_bytes,
    deterministic_dataset_checksum,
    resolve_source_revision,
    sha256_bytes,
    validate_manifest,
)
from .benchmark import (
    detect_runtime_environment,
    run_latency_benchmark,
)
from .evaluation import (
    EvaluationReport,
    build_evaluation_cohort,
    evaluate_retrievers,
    history_segment_metrics,
)
from .event_store import AppendOnlyEventStore, DataSnapshot
from .fixture import load_movielens_fixture
from .fusion import (
    FUSION_BANKS,
    FusionFeatureBuilder,
    FusionQueryMode,
    FusionTrainingRow,
    build_lhf_training_rows,
    rank_lhf_union,
    train_lhf_classifier,
)
from .itemknn import fit_itemknn
from .kafka import KafkaBoundary
from .lightgcn import LightGCNConfig, fit_lightgcn
from .models import RatingEvent
from .movielens import MovieLens20MSource
from .multivae import MultVAEConfig, fit_multivae
from .popularity import fit_popularity
from .positive import derive_positive_interactions
from .report import write_rolling_report
from .retriever import CandidateRetriever
from .serving import ItemKNNRetriever, LightGCNRetriever, MultVAERetriever
from .temporal import split_temporal_events
from .validation import validate_smoke_queries

ROLLING_STAGES = (50, 60, 70, 80, 90, 100)
FUSION_MODES: tuple[FusionQueryMode, ...] = ("known_user", "history_only")
PhaseObserver = Callable[[str, int], None]


class RollingLifecycleError(ValueError):
    """Raised when a rolling run cannot safely resume or advance."""


@dataclass(frozen=True, slots=True)
class RollingStageResult:
    percentage: int
    snapshot_event_count: int
    source_event_count: int
    snapshot_fingerprint: str
    artifact_id: str
    artifact_path: Path
    active: bool
    evaluation_window: tuple[int, int] | None
    evaluation: Mapping[str, Any] | None
    timings: Mapping[str, float]
    lhf_training: Mapping[str, Mapping[str, Any]]

    @property
    def future_quality_claim(self) -> bool:
        return self.evaluation_window is not None and self.evaluation is not None


@dataclass(frozen=True, slots=True)
class RollingLifecycleResult:
    output_dir: Path
    event_store_dir: Path
    artifact_store_path: Path
    active_pointer_path: Path
    checkpoint_path: Path
    report_path: Path
    latency_benchmark_path: Path
    stages: tuple[RollingStageResult, ...]
    active_artifact_id: str
    source_event_count: int
    dataset_checksum: str
    configuration_checksum: str
    source_revision: str

    @property
    def artifact_path(self) -> Path:
        """Return the final active artifact for callers that use the fast result seam."""

        return self.artifact_store_path / self.active_artifact_id


@dataclass(frozen=True, slots=True)
class _CompletedStage:
    record: dict[str, Any]
    artifact: ServingArtifact | None
    manifest: dict[str, Any]
    evaluation_record: dict[str, Any] | None


def run_rolling_lifecycle(
    output_dir: Path,
    kafka: KafkaBoundary,
    source_events: Sequence[RatingEvent] | None = None,
    topic: str | None = None,
    *,
    source_revision: str | None = None,
    smoke_validator: Callable[[ServingArtifact], None] | None = None,
    random_seed: int = 42,
    evaluation_cohort_limit: int = 1_000,
    fusion_negative_rows_per_query: int | None = None,
    multivae_config: MultVAEConfig | None = None,
    multivae_device_preference: str = "cpu",
    lightgcn_config: LightGCNConfig | None = None,
    lightgcn_device_preference: str = "cpu",
    phase_observer: PhaseObserver | None = None,
) -> RollingLifecycleResult:
    """Run the six prequential stages without ingesting a future window early.

    The checkpoint is deliberately a small local JSON ledger.  Stage snapshots, evaluation
    records, and immutable Serving Artifacts are separate files so a completed stage can be
    validated and reused without mutating its bytes.
    """

    if lightgcn_device_preference not in {"cpu", "auto", "mps"}:
        raise ValueError("lightgcn_device_preference must be one of: cpu, auto, mps")
    if multivae_device_preference not in {"cpu", "auto", "mps"}:
        raise ValueError("multivae_device_preference must be one of: cpu, auto, mps")
    if isinstance(evaluation_cohort_limit, bool) or evaluation_cohort_limit < 1:
        raise ValueError("evaluation_cohort_limit must be positive")
    if fusion_negative_rows_per_query is not None and (
        isinstance(fusion_negative_rows_per_query, bool)
        or fusion_negative_rows_per_query < 1
    ):
        raise ValueError("fusion_negative_rows_per_query must be positive")
    full_source = isinstance(source_events, MovieLens20MSource)
    ordered_events: Sequence[RatingEvent]
    if isinstance(source_events, MovieLens20MSource):
        ordered_events = source_events
    else:
        raw_events = tuple(source_events) if source_events is not None else load_movielens_fixture()
        ordered_events = _canonical_source_events(raw_events)
    if not ordered_events:
        raise ValueError("the rolling lifecycle needs at least one source Rating Event")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_store = ArtifactStore(output_dir)
    source_checksum = (
        source_events.dataset_checksum
        if isinstance(source_events, MovieLens20MSource)
        else deterministic_dataset_checksum(ordered_events)
    )
    resolved_revision = source_revision or resolve_source_revision()
    resolved_lightgcn_config = lightgcn_config or LightGCNConfig(seed=random_seed)
    resolved_multivae_config = multivae_config or MultVAEConfig()
    configuration = _rolling_configuration(
        profile="full" if full_source else "rolling",
        random_seed=random_seed,
        evaluation_cohort_limit=evaluation_cohort_limit,
        fusion_negative_rows_per_query=fusion_negative_rows_per_query,
        multivae_config=resolved_multivae_config,
        multivae_device_preference=multivae_device_preference,
        lightgcn_config=resolved_lightgcn_config,
        lightgcn_device_preference=lightgcn_device_preference,
    )
    configuration_checksum = sha256_bytes(canonical_json_bytes(configuration))
    checkpoint_path = output_dir / "rolling_checkpoint.json"

    ledger, kafka_topic, kafka_group = _load_or_create_ledger(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        source_event_count=len(ordered_events),
        source_checksum=source_checksum,
        configuration=configuration,
        configuration_checksum=configuration_checksum,
        source_revision=resolved_revision,
        requested_topic=topic,
    )

    event_store_dir = output_dir / "event_store"
    event_store = AppendOnlyEventStore(event_store_dir)
    smoke = smoke_validator or validate_smoke_queries
    stage_results: list[RollingStageResult] = []
    completed: dict[int, _CompletedStage] = {}

    for percentage in ROLLING_STAGES:
        expected_snapshot = _snapshot_for_percentage(ordered_events, percentage)
        prior = _load_completed_stage(
            percentage=percentage,
            ledger=ledger,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            artifact_store=artifact_store,
            expected_snapshot=expected_snapshot,
            source_event_count=len(ordered_events),
            source_checksum=source_checksum,
            configuration_checksum=configuration_checksum,
            random_seed=random_seed,
            source_revision=resolved_revision,
            ordered_events=ordered_events,
        )
        if prior is not None:
            completed[percentage] = prior
            stage_results.append(_stage_result_from_record(prior.record, output_dir))
            continue

        _observe(phase_observer, "stage_start", percentage)
        repairing_after_downstream = any(
            int(stage_key) > percentage
            for stage_key in ledger.get("stages", {})
            if isinstance(stage_key, str) and stage_key.isdigit()
        )
        _ensure_snapshot_prefix_not_ahead(
            event_store,
            expected_snapshot,
            ordered_events,
            percentage,
            allow_ahead=repairing_after_downstream,
        )
        target_count = expected_snapshot.event_count
        if full_source:
            del expected_snapshot
        _observe(
            phase_observer,
            "ingest_initial" if percentage == 50 else "ingest_stage",
            percentage,
        )
        if not repairing_after_downstream:
            _ensure_ingested_prefix(
                kafka=kafka,
                topic=kafka_topic,
                group_id=kafka_group,
                event_store=event_store,
                source_events=ordered_events,
                target_count=target_count,
            )
        snapshot = _snapshot_prefix_from_store(event_store, ordered_events, percentage)
        _write_stage_snapshot(output_dir, percentage, snapshot)

        interactions = derive_positive_interactions(snapshot.events)
        training_started = monotonic()
        retrievers = _fit_retrievers(
            snapshot,
            interactions,
            random_seed=random_seed,
            multivae_config=resolved_multivae_config,
            multivae_device_preference=multivae_device_preference,
            lightgcn_config=resolved_lightgcn_config,
            lightgcn_device_preference=lightgcn_device_preference,
        )
        prior_rows, prior_sources, prior_best = _prior_validation_context(completed)
        fusions, best_single_names = _fit_stage_fusions(
            percentage=percentage,
            snapshot=snapshot,
            interactions=interactions,
            random_seed=random_seed,
            evaluation_cohort_limit=evaluation_cohort_limit,
            fusion_negative_rows_per_query=fusion_negative_rows_per_query,
            multivae_config=resolved_multivae_config,
            multivae_device_preference=multivae_device_preference,
            lightgcn_config=resolved_lightgcn_config,
            lightgcn_device_preference=lightgcn_device_preference,
            prior_rows=prior_rows,
            prior_sources=prior_sources,
            prior_best=prior_best,
        )
        training_seconds = monotonic() - training_started

        boundary = _evaluation_boundary(percentage, len(ordered_events))
        stage_configuration = _stage_configuration(
            percentage=percentage,
            boundary=boundary,
            configuration=configuration,
            validation_sources=_flatten_sources(prior_sources)
            if percentage != 50
            else _fusion_sources(fusions),
        )
        manifest_metadata = {
            "rolling_configuration_sha256": configuration_checksum,
            "evaluation_boundary": boundary,
            "rolling_stage": {
                "stage_percentage": percentage,
                "snapshot_event_count": snapshot.event_count,
                "source_event_count": len(ordered_events),
                "dataset_checksum": source_checksum,
                "configuration_sha256": configuration_checksum,
                "active": percentage == 100,
            },
        }

        published, artifact = _export_stage_artifact(
            artifact_store=artifact_store,
            percentage=percentage,
            snapshot=snapshot,
            interactions=interactions,
            source_events=ordered_events,
            source_checksum=source_checksum,
            random_seed=random_seed,
            source_revision=resolved_revision,
            training_seconds=training_seconds,
            retrievers=retrievers,
            fusions=fusions,
            stage_configuration=stage_configuration,
            manifest_metadata=manifest_metadata,
            smoke_validator=smoke,
            phase_observer=phase_observer,
        )

        evaluation_record: dict[str, Any] | None = None
        evaluation_summary: Mapping[str, Any] | None = None
        evaluation_started = monotonic()
        if percentage < 100:
            _observe(phase_observer, "evaluate", percentage)
            future_events = _future_window(ordered_events, percentage)
            evaluation_record, evaluation_summary = _evaluate_published_artifact(
                artifact=artifact,
                snapshot=snapshot,
                future_events=future_events,
                best_single_names=best_single_names,
                evaluation_cohort_limit=evaluation_cohort_limit,
                fusion_negative_rows_per_query=fusion_negative_rows_per_query,
                random_seed=random_seed,
            )
            evaluation_record["evaluation_seconds"] = monotonic() - evaluation_started
            evaluation_path, evaluation_sha256 = _write_evaluation_record(
                output_dir, percentage, evaluation_record
            )
        else:
            evaluation_path = None
            evaluation_sha256 = None

        evaluation_seconds = monotonic() - evaluation_started
        timings = {
            "training": float(training_seconds),
            "export": float(artifact.manifest["timings"].get("export", 0.0)),
            "smoke": float(artifact.manifest["timings"].get("smoke", 0.0)),
            "evaluation": float(evaluation_seconds) if percentage < 100 else 0.0,
            "ingest": 0.0,
        }
        stage_record = _make_stage_record(
            percentage=percentage,
            snapshot=snapshot,
            artifact=artifact,
            artifact_path=published,
            source_event_count=len(ordered_events),
            source_checksum=source_checksum,
            configuration_checksum=configuration_checksum,
            random_seed=random_seed,
            source_revision=resolved_revision,
            boundary=boundary,
            evaluation_summary=evaluation_summary,
            evaluation_path=evaluation_path,
            evaluation_sha256=evaluation_sha256,
            timings=timings,
        )
        record_path, record_sha256 = _write_stage_record(output_dir, percentage, stage_record)
        _record_completed_stage(
            ledger=ledger,
            checkpoint_path=checkpoint_path,
            percentage=percentage,
            record_path=record_path,
            record_sha256=record_sha256,
        )
        completed_stage = _CompletedStage(
            record=stage_record,
            artifact=None,
            manifest=artifact.manifest,
            evaluation_record=evaluation_record,
        )
        completed[percentage] = completed_stage
        stage_results.append(_stage_result_from_record(stage_record, output_dir))

        if full_source:
            del retrievers, fusions, interactions, snapshot, artifact
            del prior_rows, prior_sources, prior_best
            if percentage < 100:
                del future_events
            gc.collect()

        if percentage < 100 and not repairing_after_downstream:
            _observe(phase_observer, "ingest_next", percentage)
            ingest_started = monotonic()
            _ensure_ingested_prefix(
                kafka=kafka,
                topic=kafka_topic,
                group_id=kafka_group,
                event_store=event_store,
                source_events=ordered_events,
                target_count=_snapshot_for_percentage(ordered_events, percentage + 10).event_count,
            )
            timings["ingest"] = monotonic() - ingest_started

    stage_results.sort(key=lambda stage: stage.percentage)
    active_artifact_id = stage_results[-1].artifact_id
    if _active_pointer_id(artifact_store) != active_artifact_id:
        raise RollingLifecycleError(
            "rolling lifecycle completed but active pointer does not target the 100% artifact"
        )
    active_artifact = artifact_store.load_active()
    benchmark_path = output_dir / "latency_benchmark.json"
    try:
        latency_benchmark = run_latency_benchmark(active_artifact)
    except Exception as error:
        latency_benchmark = _failed_latency_benchmark_record(
            artifact=active_artifact,
            reason=f"benchmark failed: {type(error).__name__}: {error}",
        )
    benchmark_path.write_text(
        json.dumps(latency_benchmark, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_path = output_dir / "report.html"
    write_rolling_report(
        report_path,
        stages=[
            _report_stage_payload(stage, completed[stage.percentage]) for stage in stage_results
        ],
        source_event_count=len(ordered_events),
        dataset_checksum=source_checksum,
        active_artifact_id=active_artifact_id,
        source_revision=resolved_revision,
        configuration=configuration,
        latency_benchmark=latency_benchmark,
    )
    return RollingLifecycleResult(
        output_dir=output_dir,
        event_store_dir=event_store_dir,
        artifact_store_path=artifact_store.root,
        active_pointer_path=artifact_store.active_pointer_path,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        latency_benchmark_path=benchmark_path,
        stages=tuple(stage_results),
        active_artifact_id=active_artifact_id,
        source_event_count=len(ordered_events),
        dataset_checksum=source_checksum,
        configuration_checksum=configuration_checksum,
        source_revision=resolved_revision,
    )


def _canonical_source_events(events: Iterable[RatingEvent]) -> tuple[RatingEvent, ...]:
    unique: dict[str, RatingEvent] = {}
    for event in events:
        previous = unique.get(event.event_id)
        if previous is not None and previous != event:
            raise RollingLifecycleError(
                f"source contains conflicting payloads for Rating Event {event.event_id!r}"
            )
        unique[event.event_id] = event
    return tuple(sorted(unique.values(), key=lambda event: (event.event_time, event.event_id)))


def _rolling_configuration(
    *,
    profile: str,
    random_seed: int,
    evaluation_cohort_limit: int,
    fusion_negative_rows_per_query: int | None,
    multivae_config: MultVAEConfig,
    multivae_device_preference: str,
    lightgcn_config: LightGCNConfig,
    lightgcn_device_preference: str,
) -> dict[str, Any]:
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("rolling random_seed must be an integer")
    return {
        "profile": profile,
        "stage_percentages": list(ROLLING_STAGES),
        "future_window_percentage": 10,
        "implicit_signal_threshold": 4.0,
        "random_seed": random_seed,
        "evaluation_cohort_limit": evaluation_cohort_limit,
        "fusion_negative_rows_per_query": fusion_negative_rows_per_query,
        "multivae_device_preference": multivae_device_preference,
        "multivae_config": multivae_config.to_dict(),
        "lightgcn_device_preference": lightgcn_device_preference,
        "lightgcn_config": lightgcn_config.to_dict(),
    }


def _load_or_create_ledger(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    source_event_count: int,
    source_checksum: str,
    configuration: Mapping[str, Any],
    configuration_checksum: str,
    source_revision: str,
    requested_topic: str | None,
) -> tuple[dict[str, Any], str, str]:
    if checkpoint_path.exists():
        loaded_ledger = _read_json_object(checkpoint_path)
        _validate_ledger_checksum(loaded_ledger)
        if loaded_ledger.get("source_event_count") != source_event_count:
            raise RollingLifecycleError("rolling source dataset checksum/event count mismatch")
        if loaded_ledger.get("source_dataset_checksum") != source_checksum:
            raise RollingLifecycleError("rolling source dataset checksum mismatch")
        if loaded_ledger.get("configuration_sha256") != configuration_checksum:
            raise RollingLifecycleError("rolling configuration checksum mismatch")
        if loaded_ledger.get("source_revision") != source_revision:
            raise RollingLifecycleError("rolling code revision mismatch")
        topic = loaded_ledger.get("kafka_topic")
        group = loaded_ledger.get("kafka_group_id")
        if not isinstance(topic, str) or not isinstance(group, str):
            raise RollingLifecycleError("rolling checkpoint has invalid Kafka identity")
        if requested_topic is not None and requested_topic != topic:
            raise RollingLifecycleError("rolling Kafka topic differs from the saved checkpoint")
        if loaded_ledger.get("configuration") != dict(configuration):
            raise RollingLifecycleError("rolling configuration differs from the saved checkpoint")
        return loaded_ledger, topic, group

    existing = [path for path in output_dir.iterdir() if path.name != "rolling_checkpoint.json"]
    if existing:
        raise RollingLifecycleError(
            "rolling checkpoint is missing; refusing to mix an existing output directory"
        )
    topic = requested_topic or f"movie-recsys-rolling-{uuid4().hex[:12]}"
    group = f"{topic}-consumer"
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "source_event_count": source_event_count,
        "source_dataset_checksum": source_checksum,
        "configuration": dict(configuration),
        "configuration_sha256": configuration_checksum,
        "random_seed": configuration["random_seed"],
        "source_revision": source_revision,
        "kafka_topic": topic,
        "kafka_group_id": group,
        "stages": {},
        "last_completed_stage": None,
    }
    _write_ledger(checkpoint_path, ledger)
    return ledger, topic, group


def _write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    payload = dict(ledger)
    payload["ledger_sha256"] = _checksum_without_key(payload, "ledger_sha256")
    _write_json_atomic(path, payload)
    ledger.clear()
    ledger.update(payload)


def _validate_ledger_checksum(ledger: Mapping[str, Any]) -> None:
    expected = ledger.get("ledger_sha256")
    if not isinstance(expected, str) or _checksum_without_key(ledger, "ledger_sha256") != expected:
        raise RollingLifecycleError("rolling checkpoint checksum is invalid")
    if ledger.get("schema_version") != 1 or not isinstance(ledger.get("stages"), Mapping):
        raise RollingLifecycleError("rolling checkpoint has an invalid schema")


def _record_completed_stage(
    *,
    ledger: dict[str, Any],
    checkpoint_path: Path,
    percentage: int,
    record_path: Path,
    record_sha256: str,
) -> None:
    stages = dict(ledger.get("stages", {}))
    stages[str(percentage)] = {
        "status": "completed",
        "record_path": str(record_path.relative_to(checkpoint_path.parent)),
        "record_sha256": record_sha256,
    }
    ledger["stages"] = stages
    ledger["last_completed_stage"] = percentage
    _write_ledger(checkpoint_path, ledger)


def _load_completed_stage(
    *,
    percentage: int,
    ledger: dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    artifact_store: ArtifactStore,
    expected_snapshot: DataSnapshot,
    source_event_count: int,
    source_checksum: str,
    configuration_checksum: str,
    random_seed: int,
    source_revision: str,
    ordered_events: Sequence[RatingEvent],
) -> _CompletedStage | None:
    entry = ledger.get("stages", {}).get(str(percentage))
    if not isinstance(entry, Mapping) or entry.get("status") != "completed":
        return None
    try:
        record_path = _safe_relative_path(output_dir, entry.get("record_path"))
        raw_bytes = record_path.read_bytes()
        record = _read_json_object_bytes(raw_bytes)
        if entry.get("record_sha256") != sha256_bytes(raw_bytes):
            raise RollingLifecycleError("stage record checksum is invalid")
        if record.get("record_sha256") != _checksum_without_key(record, "record_sha256"):
            raise RollingLifecycleError("stage record self-checksum is invalid")
        _validate_stage_record_identity(
            record,
            percentage=percentage,
            expected_snapshot=expected_snapshot,
            source_event_count=source_event_count,
            source_checksum=source_checksum,
            configuration_checksum=configuration_checksum,
            random_seed=random_seed,
            source_revision=source_revision,
        )
        snapshot_path = _safe_relative_path(output_dir, record.get("snapshot_path"))
        snapshot_bytes = snapshot_path.read_bytes()
        if record.get("snapshot_sha256") != sha256_bytes(snapshot_bytes):
            raise RollingLifecycleError("stage snapshot checksum is invalid")
        snapshot_value = _read_json_object_bytes(snapshot_bytes)
        if snapshot_value.get("fingerprint") != expected_snapshot.fingerprint:
            raise RollingLifecycleError("stage snapshot fingerprint is invalid")
        artifact_id = record.get("artifact_id")
        if not isinstance(artifact_id, str):
            raise RollingLifecycleError("stage record has no artifact identity")
        artifact_path = _artifact_path_for_id(artifact_store, artifact_id)
        manifest = _load_artifact_manifest_without_native_models(artifact_path)
        artifact = None
        _validate_completed_artifact(
            manifest,
            artifact_id=artifact_id,
            record=record,
            percentage=percentage,
            expected_snapshot=expected_snapshot,
            source_event_count=source_event_count,
            source_checksum=source_checksum,
            configuration_checksum=configuration_checksum,
            source_revision=source_revision,
        )
        if record.get("artifact_sha256") != _directory_checksum(artifact_path):
            raise RollingLifecycleError("stage artifact checksum is invalid")

        evaluation_record: dict[str, Any] | None = None
        if percentage < 100:
            evaluation_path = _safe_relative_path(output_dir, record.get("evaluation_path"))
            evaluation_bytes = evaluation_path.read_bytes()
            if record.get("evaluation_sha256") != sha256_bytes(evaluation_bytes):
                raise RollingLifecycleError("stage evaluation record checksum is invalid")
            evaluation_record = _read_json_object_bytes(evaluation_bytes)
            _validate_evaluation_record(
                evaluation_record,
                percentage=percentage,
                snapshot=expected_snapshot,
                future_events=_future_window(ordered_events, percentage),
            )
        elif record.get("evaluation") is not None or record.get("evaluation_path") is not None:
            raise RollingLifecycleError("100% stage contains an unsupported future evaluation")
        if percentage == 100:
            if _active_pointer_id(artifact_store) != artifact_id:
                raise RollingLifecycleError("100% stage is complete but is not active")
        return _CompletedStage(
            record=record,
            artifact=artifact,
            manifest=manifest,
            evaluation_record=evaluation_record,
        )
    except Exception:
        _quarantine_incomplete_stage(
            percentage=percentage,
            output_dir=output_dir,
            artifact_store=artifact_store,
            ledger=ledger,
            checkpoint_path=checkpoint_path,
            entry=entry,
        )
        return None


def _validate_stage_record_identity(
    record: Mapping[str, Any],
    *,
    percentage: int,
    expected_snapshot: DataSnapshot,
    source_event_count: int,
    source_checksum: str,
    configuration_checksum: str,
    random_seed: int,
    source_revision: str,
) -> None:
    expected = {
        "status": "completed",
        "stage_percentage": percentage,
        "source_event_count": source_event_count,
        "snapshot_event_count": expected_snapshot.event_count,
        "snapshot_fingerprint": expected_snapshot.fingerprint,
        "dataset_checksum": source_checksum,
        "configuration_sha256": configuration_checksum,
        "random_seed": random_seed,
        "source_revision": source_revision,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RollingLifecycleError(f"stage record {key} does not match the run")


def _validate_completed_artifact(
    manifest: Mapping[str, Any],
    *,
    artifact_id: str,
    record: Mapping[str, Any],
    percentage: int,
    expected_snapshot: DataSnapshot,
    source_event_count: int,
    source_checksum: str,
    configuration_checksum: str,
    source_revision: str,
) -> None:
    rolling = manifest.get("rolling_stage")
    if (
        manifest.get("lifecycle_stage") != f"rolling-{percentage}"
        or manifest.get("data_cutoff_percentage") != percentage
        or manifest.get("source_event_count") != source_event_count
        or manifest.get("event_count") != expected_snapshot.event_count
        or manifest.get("data_snapshot_fingerprint") != expected_snapshot.fingerprint
        or manifest.get("source_revision") != source_revision
        or manifest.get("rolling_configuration_sha256") != configuration_checksum
        or manifest.get("dataset_checksum") != source_checksum
        or not isinstance(rolling, Mapping)
        or rolling.get("stage_percentage") != percentage
        or rolling.get("active") != (percentage == 100)
    ):
        raise RollingLifecycleError("stage artifact provenance does not match the run")
    if record.get("artifact_id") != artifact_id:
        raise RollingLifecycleError("stage record and artifact identities differ")


def _artifact_path_for_id(store: ArtifactStore, artifact_id: str) -> Path:
    pure_id = PurePosixPath(artifact_id)
    if (
        not artifact_id
        or "\\" in artifact_id
        or pure_id.is_absolute()
        or pure_id.parts != (artifact_id,)
    ):
        raise RollingLifecycleError("stage record contains an invalid artifact id")
    path = store.root / artifact_id
    if not path.is_dir():
        raise RollingLifecycleError(f"stage artifact is missing: {artifact_id}")
    return path


def _load_artifact_manifest_without_native_models(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise RollingLifecycleError("stage artifact manifest is missing")
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RollingLifecycleError("stage artifact manifest is unreadable") from error
    typed_manifest = validate_manifest(manifest_value)
    manifest = typed_manifest.to_dict()
    model_descriptor = next(
        (
            descriptor
            for descriptor in typed_manifest.payload_inventory
            if descriptor.path == "model.json"
        ),
        None,
    )
    if model_descriptor is None or model_descriptor.sha256 != typed_manifest.model_sha256:
        raise RollingLifecycleError("stage artifact model payload inventory is invalid")
    for descriptor in typed_manifest.payload_inventory:
        payload_path = _safe_bundle_payload_path(path, descriptor.path)
        if (
            not payload_path.is_file()
            or sha256_bytes(payload_path.read_bytes()) != descriptor.sha256
        ):
            raise RollingLifecycleError(
                f"stage artifact payload checksum is invalid: {descriptor.path}"
            )
    try:
        model_value = json.loads((path / "model.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RollingLifecycleError("stage artifact model payload is unreadable") from error
    if not isinstance(model_value, Mapping):
        raise RollingLifecycleError("stage artifact model payload is invalid")
    if model_value.get("payload_schema_version") != 4:
        raise RollingLifecycleError("rolling stage artifact must use payload schema 4")
    for key in ("popularity", "itemknn", "multivae", "lightgcn", "fusion"):
        if key not in model_value or model_value[key] is None:
            raise RollingLifecycleError(f"stage artifact payload is missing {key}")
    return manifest


def _safe_bundle_payload_path(root: Path, value: str) -> Path:
    pure_path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or pure_path.is_absolute()
        or "." in pure_path.parts
        or ".." in pure_path.parts
    ):
        raise RollingLifecycleError("stage artifact payload path is invalid")
    target = (root / Path(*pure_path.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise RollingLifecycleError("stage artifact payload path escapes its bundle") from error
    return target


def _validate_evaluation_record(
    record: Mapping[str, Any],
    *,
    percentage: int,
    snapshot: DataSnapshot,
    future_events: Sequence[RatingEvent],
) -> None:
    if (
        record.get("schema_version") != 1
        or record.get("stage_percentage") != percentage
        or record.get("snapshot_fingerprint") != snapshot.fingerprint
        or record.get("evaluated_before_ingest") is not True
    ):
        raise RollingLifecycleError("stage evaluation record has invalid provenance")
    boundary = record.get("future_window")
    if not isinstance(boundary, Mapping):
        raise RollingLifecycleError("stage evaluation record has an invalid Future Window")
    if "event_ids_sha256" in boundary:
        valid_window = (
            boundary.get("event_count") == len(future_events)
            and boundary.get("event_ids_sha256") == _event_ids_sha256(future_events)
        )
    else:
        valid_window = boundary.get("event_ids") == [
            event.event_id for event in future_events
        ]
    if not valid_window:
        raise RollingLifecycleError("stage evaluation record has an invalid Future Window")
    rows = record.get("validation_rows")
    if not isinstance(rows, Mapping) or not isinstance(rows.get("known_user"), list):
        raise RollingLifecycleError("stage evaluation record is missing Known-User rows")
    if not isinstance(rows.get("history_only"), list):
        raise RollingLifecycleError("stage evaluation record is missing History-Only rows")
    pools = record.get("validation_pools")
    if not isinstance(pools, Mapping) or not isinstance(pools.get("known_user"), Mapping):
        raise RollingLifecycleError("stage evaluation record is missing Known-User pools")
    if not isinstance(pools.get("history_only"), Mapping):
        raise RollingLifecycleError("stage evaluation record is missing History-Only pools")


def _quarantine_incomplete_stage(
    *,
    percentage: int,
    output_dir: Path,
    artifact_store: ArtifactStore,
    ledger: dict[str, Any],
    checkpoint_path: Path,
    entry: Mapping[str, Any],
) -> None:
    record_value = entry.get("record_path")
    try:
        record_path = _safe_relative_path(output_dir, record_value)
    except RollingLifecycleError:
        record_path = None
    record: dict[str, Any] = {}
    if record_path is not None and record_path.is_file():
        try:
            record = _read_json_object(record_path)
        except Exception:
            record = {}
    if record_path is not None:
        _quarantine_path(record_path, output_dir)
    # The artifact ID is not trusted until the record has been validated.  Quarantine every
    # immutable directory named by a valid record, while never moving the active 100% bundle.
    artifact_id = record.get("artifact_id")
    if isinstance(artifact_id, str):
        artifact_path = artifact_store.root / artifact_id
        if not _active_pointer_targets(artifact_store, artifact_id):
            _quarantine_path(artifact_path, output_dir)
    stages = dict(ledger.get("stages", {}))
    stages.pop(str(percentage), None)
    ledger["stages"] = stages
    _write_ledger(checkpoint_path, ledger)


def _active_pointer_targets(store: ArtifactStore, artifact_id: str) -> bool:
    return _active_pointer_id(store) == artifact_id


def _active_pointer_id(store: ArtifactStore) -> str | None:
    try:
        value = json.loads(store.active_pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    artifact_id = value.get("artifact_id") if isinstance(value, Mapping) else None
    return artifact_id if isinstance(artifact_id, str) else None


def _ensure_snapshot_prefix_not_ahead(
    store: AppendOnlyEventStore,
    expected_snapshot: DataSnapshot,
    ordered_events: Sequence[RatingEvent],
    percentage: int,
    *,
    allow_ahead: bool = False,
) -> None:
    if isinstance(ordered_events, MovieLens20MSource):
        actual_ids = set(store.iter_event_ids())
        if allow_ahead:
            unmatched = actual_ids.copy()
            for event in ordered_events:
                unmatched.discard(event.event_id)
            if unmatched:
                raise RollingLifecycleError(
                    "Event Store contains a Rating Event outside the source dataset"
                )
        else:
            expected_ids = {event.event_id for event in expected_snapshot.events}
            if not actual_ids <= expected_ids:
                raise RollingLifecycleError(
                    f"Future Window for stage {percentage}% was ingested before its evaluation"
                )
        return
    actual_ids = {event.event_id for event in store.materialize_snapshot().events}
    expected_ids = {event.event_id for event in expected_snapshot.events}
    source_ids = {event.event_id for event in ordered_events}
    if not actual_ids <= source_ids:
        raise RollingLifecycleError(
            "Event Store contains a Rating Event outside the source dataset"
        )
    if not allow_ahead and not actual_ids <= expected_ids:
        raise RollingLifecycleError(
            f"Future Window for stage {percentage}% was ingested before its evaluation"
        )


def _ensure_ingested_prefix(
    *,
    kafka: KafkaBoundary,
    topic: str,
    group_id: str,
    event_store: AppendOnlyEventStore,
    source_events: Sequence[RatingEvent],
    target_count: int,
) -> None:
    if isinstance(source_events, MovieLens20MSource):
        current_ids = set(event_store.iter_event_ids())
        unmatched_ids = current_ids.copy()
        for index in range(target_count):
            unmatched_ids.discard(source_events[index].event_id)
        if unmatched_ids:
            raise RollingLifecycleError("Event Store is ahead of the requested rolling boundary")
        del unmatched_ids
        pending: list[RatingEvent] = []
        for index in range(target_count):
            event = source_events[index]
            if event.event_id not in current_ids:
                pending.append(event)
            if len(pending) == 50_000:
                kafka.publish(topic, pending)
                kafka.consume_to_store(
                    topic, group_id, len(pending), event_store, timeout_seconds=300.0
                )
                pending = []
        if pending:
            kafka.publish(topic, pending)
            kafka.consume_to_store(
                topic, group_id, len(pending), event_store, timeout_seconds=300.0
            )
        return
    target_events = tuple(source_events[:target_count])
    current = event_store.materialize_snapshot()
    current_ids = {event.event_id for event in current.events}
    target_ids = {event.event_id for event in target_events}
    source_ids = {event.event_id for event in source_events}
    if not current_ids <= source_ids:
        raise RollingLifecycleError(
            "Event Store contains a Rating Event outside the source dataset"
        )
    if not current_ids <= target_ids:
        raise RollingLifecycleError("Event Store is ahead of the requested rolling boundary")
    missing = tuple(event for event in target_events if event.event_id not in current_ids)
    if not missing:
        return
    kafka.publish(topic, missing)
    kafka.consume_to_store(
        topic=topic,
        group_id=group_id,
        expected_count=len(missing),
        store=event_store,
    )
    after = event_store.materialize_snapshot()
    after_ids = {event.event_id for event in after.events}
    if after_ids != target_ids:
        raise RollingLifecycleError(
            f"Kafka ingestion did not materialize the exact {target_count}-event boundary"
        )


def _snapshot_prefix_from_store(
    store: AppendOnlyEventStore,
    source_events: Sequence[RatingEvent],
    percentage: int,
) -> DataSnapshot:
    if isinstance(source_events, MovieLens20MSource):
        actual = store.materialize_snapshot()
        target_count = len(source_events) * percentage // 100
        if actual.event_count != target_count:
            raise RollingLifecycleError("materialized stage snapshot has the wrong event count")
        for index, event in enumerate(actual.events):
            if event.event_id != source_events[index].event_id:
                raise RollingLifecycleError(
                    "Event Store is missing a source event at the stage boundary"
                )
        return actual
    expected = _snapshot_for_percentage(source_events, percentage)
    actual = store.materialize_snapshot()
    actual_by_id = {event.event_id: event for event in actual.events}
    try:
        events = tuple(actual_by_id[event.event_id] for event in expected.events)
    except KeyError as error:
        raise RollingLifecycleError(
            "Event Store is missing a source event at the stage boundary"
        ) from error
    if len(events) != expected.event_count:
        raise RollingLifecycleError("materialized stage snapshot has the wrong event count")
    return DataSnapshot(events=events, source_batch_count=actual.source_batch_count)


def _snapshot_for_percentage(events: Sequence[RatingEvent], percentage: int) -> DataSnapshot:
    boundary = len(events) * percentage // 100
    return DataSnapshot(events=tuple(events[:boundary]), source_batch_count=1)


def _future_window(events: Sequence[RatingEvent], percentage: int) -> tuple[RatingEvent, ...]:
    start = len(events) * percentage // 100
    end = len(events) * (percentage + 10) // 100
    return tuple(events[start:end])


def _evaluation_boundary(percentage: int, source_event_count: int) -> dict[str, Any]:
    snapshot_count = source_event_count * percentage // 100
    if percentage == 100:
        return {
            "snapshot_cutoff_percentage": 100,
            "snapshot_event_count": snapshot_count,
            "future_window_start_percentage": None,
            "future_window_end_percentage": None,
            "future_window_event_count": 0,
            "evaluated_before_future_window_ingest": None,
            "future_quality_claim": False,
        }
    future_count = source_event_count * (percentage + 10) // 100 - snapshot_count
    return {
        "snapshot_cutoff_percentage": percentage,
        "snapshot_event_count": snapshot_count,
        "future_window_start_percentage": percentage,
        "future_window_end_percentage": percentage + 10,
        "future_window_event_count": future_count,
        "evaluated_before_future_window_ingest": True,
        "future_quality_claim": True,
    }


def _fit_retrievers(
    snapshot: DataSnapshot,
    interactions: Sequence[Any],
    *,
    random_seed: int,
    multivae_config: MultVAEConfig,
    multivae_device_preference: str,
    lightgcn_config: LightGCNConfig,
    lightgcn_device_preference: str,
) -> dict[str, CandidateRetriever]:
    return {
        "popularity": fit_popularity(snapshot, interactions),
        "itemknn": fit_itemknn(snapshot, interactions).to_serving(),
        "multivae": fit_multivae(
            snapshot,
            interactions,
            config=multivae_config,
            seed=random_seed,
            device_preference=multivae_device_preference,
        ),
        "lightgcn": fit_lightgcn(
            snapshot,
            interactions,
            config=lightgcn_config,
            seed=random_seed,
            device_preference=lightgcn_device_preference,
        ),
    }


def _fit_stage_fusions(
    *,
    percentage: int,
    snapshot: DataSnapshot,
    interactions: Sequence[Any],
    random_seed: int,
    evaluation_cohort_limit: int,
    fusion_negative_rows_per_query: int | None,
    multivae_config: MultVAEConfig,
    multivae_device_preference: str,
    lightgcn_config: LightGCNConfig,
    lightgcn_device_preference: str,
    prior_rows: Mapping[str, Sequence[FusionTrainingRow]],
    prior_sources: Mapping[str, Sequence[Mapping[str, Any]]],
    prior_best: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    builders = {
        mode: FusionFeatureBuilder.from_snapshot(
            snapshot,
            interactions,
            retriever_bank=FUSION_BANKS[mode],
        )
        for mode in FUSION_MODES
    }
    rows: dict[str, tuple[FusionTrainingRow, ...]] = {
        mode: tuple(prior_rows.get(mode, ())) for mode in builders
    }
    sources: dict[str, tuple[Mapping[str, Any], ...]] = {
        mode: tuple(prior_sources.get(mode, ())) for mode in builders
    }
    best: dict[str, str] = {str(mode): prior_best.get(mode, "popularity") for mode in builders}

    if percentage == 50:
        inner_rows, inner_sources, inner_best = _build_inner_validation(
            snapshot=snapshot,
            random_seed=random_seed,
            evaluation_cohort_limit=evaluation_cohort_limit,
            fusion_negative_rows_per_query=fusion_negative_rows_per_query,
            multivae_config=multivae_config,
            multivae_device_preference=multivae_device_preference,
            lightgcn_config=lightgcn_config,
            lightgcn_device_preference=lightgcn_device_preference,
        )
        rows = inner_rows
        sources = inner_sources
        best = inner_best

    event_ids_by_mode = {mode: _source_event_ids(sources[mode]) for mode in builders}
    fusions: dict[str, Any] = {}
    for mode in builders:
        source_snapshot_fingerprint = snapshot.fingerprint
        if percentage == 50 and sources[mode]:
            candidate_fingerprint = sources[mode][0].get("snapshot_fingerprint")
            if isinstance(candidate_fingerprint, str):
                source_snapshot_fingerprint = candidate_fingerprint
        fusions[mode] = train_lhf_classifier(
            query_mode=mode,
            retriever_bank=FUSION_BANKS[mode],
            feature_names=builders[mode].feature_names,
            rows=rows[mode],
            source_snapshot_fingerprint=source_snapshot_fingerprint,
            seed=random_seed,
            validation_event_ids=event_ids_by_mode[mode],
            validation_sources=sources[mode],
        )
    return fusions, best


def _build_inner_validation(
    *,
    snapshot: DataSnapshot,
    random_seed: int,
    evaluation_cohort_limit: int,
    fusion_negative_rows_per_query: int | None,
    multivae_config: MultVAEConfig,
    multivae_device_preference: str,
    lightgcn_config: LightGCNConfig,
    lightgcn_device_preference: str,
) -> tuple[
    dict[str, tuple[FusionTrainingRow, ...]],
    dict[str, tuple[Mapping[str, Any], ...]],
    dict[str, str],
]:
    empty_rows: dict[str, tuple[FusionTrainingRow, ...]] = {
        "known_user": (),
        "history_only": (),
    }
    empty_sources: dict[str, tuple[Mapping[str, Any], ...]] = {
        "known_user": (),
        "history_only": (),
    }
    empty_best: dict[str, str] = {
        "known_user": "popularity",
        "history_only": "popularity",
    }
    if snapshot.event_count < 2:
        return empty_rows, empty_sources, empty_best
    inner_split = split_temporal_events(
        snapshot.events,
        source_batch_count=snapshot.source_batch_count,
        snapshot_percentage=80,
        future_percentage=100,
    )
    inner_snapshot = inner_split.data_snapshot
    inner_interactions = derive_positive_interactions(inner_snapshot.events)
    try:
        inner_retrievers = _fit_retrievers(
            inner_snapshot,
            inner_interactions,
            random_seed=random_seed,
            multivae_config=multivae_config,
            multivae_device_preference=multivae_device_preference,
            lightgcn_config=lightgcn_config,
            lightgcn_device_preference=lightgcn_device_preference,
        )
    except ValueError:
        return empty_rows, empty_sources, empty_best

    rows: dict[str, tuple[FusionTrainingRow, ...]] = {}
    sources: dict[str, tuple[Mapping[str, Any], ...]] = {}
    best: dict[str, str] = {}
    inner_future = inner_split.future_window_events
    inner_catalog = {event.movie_id for event in inner_snapshot.events}
    for mode in FUSION_MODES:
        cohort = build_evaluation_cohort(
            inner_snapshot,
            inner_future,
            query_mode=mode,
            max_subjects=evaluation_cohort_limit,
            seed=random_seed,
        )
        bank_retrievers = {name: inner_retrievers[name] for name in FUSION_BANKS[mode]}
        report = evaluate_retrievers(
            bank_retrievers, cohort,
            candidate_catalog=inner_catalog,
        )
        builder = FusionFeatureBuilder.from_snapshot(
            inner_snapshot,
            inner_interactions,
            retriever_bank=FUSION_BANKS[mode],
        )
        rows[mode] = build_lhf_training_rows(
            cohort,
            _training_pools(report),
            builder,
            max_negative_rows_per_query=fusion_negative_rows_per_query,
            seed=random_seed,
        )
        source = {
            "source_stage_percentage": 50,
            "validation_kind": "inner_snapshot",
            "snapshot_fingerprint": inner_snapshot.fingerprint,
            "future_window_start_percentage": 80,
            "future_window_end_percentage": 100,
            "future_window_event_ids": [
                query.gold_event_id for query in cohort
            ] if snapshot.event_count > 100_000 else [
                event.event_id for event in inner_future
            ],
            "future_window_event_ids_sha256": _event_ids_sha256(inner_future),
            "future_window_event_count": len(inner_future),
            "row_count": len(rows[mode]),
        }
        sources[mode] = (source,)
        best[mode] = report.best_single_name or "popularity"
    return rows, sources, best


def _training_pools(report: EvaluationReport) -> dict[int | str, dict[str, tuple[Any, ...]]]:
    return {
        query.pool_key: {
            name: report.retrievers[name].pools[query.pool_key] for name in report.retrievers
        }
        for query in report.cohort
    }


def _fused_pools(
    report: EvaluationReport,
    fusion: Any,
    builder: FusionFeatureBuilder,
) -> dict[int | str, tuple[Any, ...]]:
    return {
        query.pool_key: rank_lhf_union(
            fusion,
            builder,
            {name: report.retrievers[name].pools[query.pool_key] for name in fusion.retriever_bank},
            query.history,
        )
        for query in report.cohort
    }


def _prior_validation_context(
    completed: Mapping[int, _CompletedStage],
) -> tuple[
    dict[str, tuple[FusionTrainingRow, ...]],
    dict[str, tuple[Mapping[str, Any], ...]],
    dict[str, str],
]:
    rows: dict[str, list[FusionTrainingRow]] = {"known_user": [], "history_only": []}
    sources: dict[str, list[Mapping[str, Any]]] = {"known_user": [], "history_only": []}
    best = {"known_user": "popularity", "history_only": "popularity"}
    for percentage in sorted(completed):
        evaluation = completed[percentage].evaluation_record
        if evaluation is None:
            continue
        raw_rows = evaluation.get("validation_rows", {})
        raw_sources = evaluation.get("validation_sources", {})
        for mode in rows:
            rows[mode].extend(_rows_from_dict(raw_rows.get(mode, [])))
            raw_mode_sources = raw_sources.get(mode, [])
            if isinstance(raw_mode_sources, list):
                sources[mode].extend(
                    dict(item) for item in raw_mode_sources if isinstance(item, Mapping)
                )
            raw_best = evaluation.get("best_single", {})
            if isinstance(raw_best, Mapping) and isinstance(raw_best.get(mode), str):
                best[mode] = str(raw_best[mode])
    return (
        {mode: tuple(value) for mode, value in rows.items()},
        {mode: tuple(value) for mode, value in sources.items()},
        best,
    )


def _source_event_ids(sources: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for source in sources:
        values = source.get("future_window_event_ids", [])
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value not in seen:
                seen.add(value)
                result.append(value)
    return tuple(result)


def _event_ids_sha256(events: Sequence[RatingEvent]) -> str:
    digest = sha256()
    for event in events:
        digest.update(event.event_id.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fusion_sources(fusions: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    for fusion in fusions.values():
        raw = fusion.training_metadata.get("validation_sources", [])
        if isinstance(raw, list):
            for source in raw:
                if isinstance(source, Mapping) and dict(source) not in [
                    dict(item) for item in sources
                ]:
                    sources.append(dict(source))
    return sources


def _flatten_sources(
    sources_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for sources in sources_by_mode.values():
        for source in sources:
            if dict(source) not in [dict(existing) for existing in result]:
                result.append(dict(source))
    return result


def _stage_configuration(
    *,
    percentage: int,
    boundary: Mapping[str, Any],
    configuration: Mapping[str, Any],
    validation_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "profile": configuration["profile"],
        "future_percentage": percentage + 10 if percentage < 100 else 100,
        "stage_percentage": percentage,
        "stage_percentages": list(ROLLING_STAGES),
        "evaluation_boundary": dict(boundary),
        "fusion_training_boundary": (
            "inner_validation_within_snapshot"
            if percentage == 50
            else "completed_prior_stage_validation_only"
        ),
        "fusion_validation_stage_percentages": [
            source.get("source_stage_percentage")
            for source in validation_sources
            if isinstance(source.get("source_stage_percentage"), int)
        ],
        "rolling_run_configuration": dict(configuration),
    }


def _export_stage_artifact(
    *,
    artifact_store: ArtifactStore,
    percentage: int,
    snapshot: DataSnapshot,
    interactions: Sequence[Any],
    source_events: Sequence[RatingEvent],
    source_checksum: str,
    random_seed: int,
    source_revision: str,
    training_seconds: float,
    retrievers: Mapping[str, CandidateRetriever],
    fusions: Mapping[str, Any],
    stage_configuration: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    smoke_validator: Callable[[ServingArtifact], None],
    phase_observer: PhaseObserver | None,
) -> tuple[Path, ServingArtifact]:
    itemknn = retrievers.get("itemknn")
    multivae = retrievers.get("multivae")
    lightgcn = retrievers.get("lightgcn")
    if not isinstance(itemknn, ItemKNNRetriever):
        raise RollingLifecycleError("rolling training did not produce ItemKNN serving state")
    if not isinstance(multivae, MultVAERetriever):
        raise RollingLifecycleError("rolling training did not produce Mult-VAE serving state")
    if not isinstance(lightgcn, LightGCNRetriever):
        raise RollingLifecycleError("rolling training did not produce LightGCN serving state")

    staging = artifact_store.new_staging_path()
    try:
        built = ServingArtifact.build(
            path=staging,
            snapshot=snapshot,
            interactions=tuple(interactions),
            source_events=source_events,
            source_dataset_checksum=source_checksum,
            source_event_count=len(source_events),
            lifecycle_stage=f"rolling-{percentage}",
            data_cutoff_percentage=percentage,
            random_seed=random_seed,
            timings={"training": training_seconds},
            evaluation_metrics={},
            source_revision=source_revision,
            itemknn_model=itemknn,
            multivae_model=multivae,
            lightgcn_model=lightgcn,
            known_user_fusion=fusions["known_user"],
            history_only_fusion=fusions["history_only"],
            configuration=dict(stage_configuration),
            manifest_metadata=dict(manifest_metadata),
        )
        built.save()
        if stage_configuration.get("profile") == "full":
            del built
            gc.collect()
        _observe(phase_observer, "export", percentage)
        staged = ServingArtifact.load(staging)
        smoke_started = monotonic()
        smoke_validator(staged)
        smoke_seconds = monotonic() - smoke_started
        if stage_configuration.get("profile") == "full":
            del staged
            gc.collect()
        staged = artifact_store.update_staging_timings(staging, {"smoke": smoke_seconds})
        _observe(phase_observer, "smoke", percentage)
        final_path = artifact_store.root / staged.artifact_id
        if final_path.exists():
            if _active_pointer_targets(artifact_store, staged.artifact_id):
                raise RollingLifecycleError(
                    "an incomplete rolling stage is already the active artifact; refusing reuse"
                )
            _quarantine_path(final_path, artifact_store.root)
        published = artifact_store.publish(staging)
        if stage_configuration.get("profile") == "full":
            staged.path = published
            if percentage == 100:
                artifact_store._write_active_pointer(staged)
            return published, staged
        loaded = artifact_store.load(published)
        if percentage == 100:
            artifact_store._activate_loaded(published)
        return published, loaded
    except BaseException:
        if staging.exists():
            artifact_store.discard_staging(staging)
        raise


def _evaluate_published_artifact(
    *,
    artifact: ServingArtifact,
    snapshot: DataSnapshot,
    future_events: Sequence[RatingEvent],
    best_single_names: Mapping[str, str],
    evaluation_cohort_limit: int,
    fusion_negative_rows_per_query: int | None,
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    large_snapshot = snapshot.event_count > 100_000
    retrievers = _artifact_retrievers(artifact)
    interactions = derive_positive_interactions(snapshot.events)
    known_cohort = build_evaluation_cohort(
        snapshot, future_events, query_mode="known_user",
        max_subjects=evaluation_cohort_limit, seed=random_seed,
    )
    history_cohort = build_evaluation_cohort(
        snapshot, future_events, query_mode="history_only",
        max_subjects=evaluation_cohort_limit, seed=random_seed,
    )
    empty_cohort = build_evaluation_cohort(
        snapshot, future_events, query_mode="empty_history",
        max_subjects=evaluation_cohort_limit, seed=random_seed,
    )
    known_base = evaluate_retrievers(
        retrievers,
        known_cohort,
        selected_best_single_name=best_single_names.get("known_user", "popularity"),
        best_single_source="inner_validation"
        if artifact.manifest["data_cutoff_percentage"] == 50
        else "prior_validation",
        candidate_catalog=artifact.model.catalog,
    )
    history_retrievers = {name: retrievers[name] for name in FUSION_BANKS["history_only"]}
    history_base = evaluate_retrievers(
        history_retrievers,
        history_cohort,
        selected_best_single_name=best_single_names.get("history_only", "popularity"),
        best_single_source="inner_validation"
        if artifact.manifest["data_cutoff_percentage"] == 50
        else "prior_validation",
        candidate_catalog=artifact.model.catalog,
    )
    known_builder = FusionFeatureBuilder.from_snapshot(
        snapshot, interactions, retriever_bank=FUSION_BANKS["known_user"]
    )
    history_builder = FusionFeatureBuilder.from_snapshot(
        snapshot, interactions, retriever_bank=FUSION_BANKS["history_only"]
    )
    if artifact.known_user_fusion is None or artifact.history_only_fusion is None:
        raise RollingLifecycleError("published rolling artifact is missing its LHF payload")
    known_lhf_pools = _fused_pools(known_base, artifact.known_user_fusion, known_builder)
    history_lhf_pools = _fused_pools(history_base, artifact.history_only_fusion, history_builder)
    known = evaluate_retrievers(
        retrievers,
        known_cohort,
        selected_best_single_name=known_base.best_single_name or "popularity",
        best_single_source=known_base.best_single_source,
        lhf_pools=known_lhf_pools,
        candidate_catalog=artifact.model.catalog,
    )
    history = evaluate_retrievers(
        history_retrievers,
        history_cohort,
        selected_best_single_name=history_base.best_single_name or "popularity",
        best_single_source=history_base.best_single_source,
        lhf_pools=history_lhf_pools,
        candidate_catalog=artifact.model.catalog,
    )
    empty = evaluate_retrievers(
        {"popularity": artifact.model},
        empty_cohort,
        selected_best_single_name="popularity",
        best_single_source="defined_fallback",
        candidate_catalog=artifact.model.catalog,
    )
    known_validation_pools = _training_pools(known_base)
    history_validation_pools = _training_pools(history_base)
    validation_rows = {
        "known_user": _rows_to_dict(
            build_lhf_training_rows(
                known_cohort,
                known_validation_pools,
                known_builder,
                max_negative_rows_per_query=fusion_negative_rows_per_query,
                seed=random_seed,
            )
        ),
        "history_only": _rows_to_dict(
            build_lhf_training_rows(
                history_cohort,
                history_validation_pools,
                history_builder,
                max_negative_rows_per_query=fusion_negative_rows_per_query,
                seed=random_seed,
            )
        ),
    }
    validation_pools = (
        {"known_user": {}, "history_only": {}}
        if large_snapshot
        else {
            "known_user": _pools_to_dict(known_validation_pools),
            "history_only": _pools_to_dict(history_validation_pools),
        }
    )
    future_window_digest = _event_ids_sha256(future_events)
    sources = {
        mode: [
            {
                "source_stage_percentage": int(artifact.manifest["data_cutoff_percentage"]),
                "validation_kind": "future_window",
                "snapshot_fingerprint": snapshot.fingerprint,
                "future_window_start_percentage": int(artifact.manifest["data_cutoff_percentage"]),
                "future_window_end_percentage": int(artifact.manifest["data_cutoff_percentage"])
                + 10,
                "future_window_event_ids": [
                    query.gold_event_id for query in known_cohort
                ] if large_snapshot else [event.event_id for event in future_events],
                "future_window_event_ids_sha256": future_window_digest,
                "future_window_event_count": len(future_events),
                "row_count": len(validation_rows[mode]),
            }
        ]
        for mode in ("known_user", "history_only")
    }
    summary = {
        "known_user": _evaluation_summary(known),
        "history_only": _evaluation_summary(history),
        "empty_history": _evaluation_summary(empty),
    }
    record = {
        "schema_version": 1,
        "stage_percentage": int(artifact.manifest["data_cutoff_percentage"]),
        "snapshot_fingerprint": snapshot.fingerprint,
        "future_window": {
            "start_percentage": int(artifact.manifest["data_cutoff_percentage"]),
            "end_percentage": int(artifact.manifest["data_cutoff_percentage"]) + 10,
            "event_count": len(future_events),
            **(
                {"event_ids_sha256": future_window_digest}
                if large_snapshot
                else {"event_ids": [event.event_id for event in future_events]}
            ),
        },
        "evaluated_before_ingest": True,
        "best_single": {
            "known_user": known.best_single_name,
            "history_only": history.best_single_name,
        },
        "evaluation": summary,
        "validation_rows": validation_rows,
        "validation_pools": validation_pools,
        "validation_sources": sources,
    }
    return record, summary


def _artifact_retrievers(artifact: ServingArtifact) -> dict[str, CandidateRetriever]:
    if artifact.itemknn is None or artifact.multivae is None or artifact.lightgcn is None:
        raise RollingLifecycleError("published rolling artifact is missing a retriever payload")
    return {
        "popularity": artifact.model,
        "itemknn": artifact.itemknn,
        "multivae": artifact.multivae,
        "lightgcn": artifact.lightgcn,
    }


def _evaluation_summary(report: EvaluationReport) -> dict[str, Any]:
    oracle_metrics = report.oracle_union.metrics if report.oracle_union is not None else None
    return {
        "query_mode": report.query_mode,
        "cohort_size": report.cohort.size,
        "best_single_name": report.best_single_name,
        "best_single_source": report.best_single_source,
        "best_single": report.best_single.metrics.to_dict() if report.best_single else {},
        "oracle_union": report.oracle_union.metrics.to_dict() if report.oracle_union else {},
        "oracle_union_coverage": report.oracle_union_coverage,
        "retrieval_failure_count": (
            report.cohort.size - oracle_metrics.covered_query_count
            if oracle_metrics is not None
            else None
        ),
        "oracle_headroom_denominator": report.oracle_headroom_denominator,
        "oracle_headroom_realized": (
            report.oracle_headroom_realized if report.oracle_headroom_realized is not None else -1.0
        ),
        "retrievers": {
            name: evaluation.metrics.to_dict() for name, evaluation in report.retrievers.items()
        },
        "rrf": report.rrf.metrics.to_dict(),
        "history_segments": (
            {} if report.query_mode == "empty_history" else history_segment_metrics(report)
        ),
        "lhf": report.lhf.metrics.to_dict() if report.lhf else {},
    }


def _failed_latency_benchmark_record(
    *,
    artifact: ServingArtifact,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "failed",
        "artifact_id": artifact.artifact_id,
        "artifact_cutoff_percentage": artifact.manifest.get("data_cutoff_percentage"),
        "artifact_loaded_before_measurement": True,
        "serving_device": artifact.manifest.get("serving_device", "cpu"),
        "top_n": 10,
        "concurrency": 1,
        "warmup_count": 0,
        "requested_sample_count": 0,
        "percentile_method": "statistics.quantiles(method='inclusive')",
        "method": "benchmark did not complete; no latency claim is made",
        "environment": detect_runtime_environment(),
        "modes": {
            mode: {
                "measurement_status": "failed",
                "slo_status": "not_measured",
                "warmup_count": 0,
                "warmup_success_count": 0,
                "requested_sample_count": 0,
                "successful_sample_count": 0,
                "failed_sample_count": 0,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "throughput_rps": None,
                "slo_target_p95_ms": target,
                "reason": reason,
            }
            for mode, target in (
                ("known_user", 200.0),
                ("history_only", 200.0),
                ("empty_history", 20.0),
            )
        },
    }


def _rows_to_dict(rows: Sequence[FusionTrainingRow]) -> list[dict[str, Any]]:
    return [
        {
            "query_key": row.query_key,
            "movie_id": row.movie_id,
            "features": list(row.features),
            "label": row.label,
        }
        for row in rows
    ]


def _pools_to_dict(
    pools: Mapping[int | str, Mapping[str, Sequence[Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        str(query_key): {
            name: [candidate.to_dict() for candidate in candidates]
            for name, candidates in query_pools.items()
        }
        for query_key, query_pools in pools.items()
    }


def _rows_from_dict(value: object) -> tuple[FusionTrainingRow, ...]:
    if not isinstance(value, list):
        raise RollingLifecycleError("validation rows must be a list")
    rows: list[FusionTrainingRow] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise RollingLifecycleError("validation row must be an object")
        key = raw.get("query_key")
        movie_id = raw.get("movie_id")
        features = raw.get("features")
        label = raw.get("label")
        if (
            not isinstance(key, (int, str))
            or isinstance(key, bool)
            or not isinstance(movie_id, int)
            or isinstance(movie_id, bool)
            or not isinstance(features, list)
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float)) for item in features
            )
            or not isinstance(label, int)
            or isinstance(label, bool)
            or label not in {0, 1}
        ):
            raise RollingLifecycleError("validation row has an invalid schema")
        rows.append(
            FusionTrainingRow(
                query_key=key,
                movie_id=movie_id,
                features=tuple(float(item) for item in features),
                label=label,
            )
        )
    return tuple(rows)


def _make_stage_record(
    *,
    percentage: int,
    snapshot: DataSnapshot,
    artifact: ServingArtifact,
    artifact_path: Path,
    source_event_count: int,
    source_checksum: str,
    configuration_checksum: str,
    random_seed: int,
    source_revision: str,
    boundary: Mapping[str, Any],
    evaluation_summary: Mapping[str, Any] | None,
    evaluation_path: Path | None,
    evaluation_sha256: str | None,
    timings: Mapping[str, float],
) -> dict[str, Any]:
    store_root = artifact_path.parent
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "stage_percentage": percentage,
        "source_event_count": source_event_count,
        "snapshot_event_count": snapshot.event_count,
        "snapshot_fingerprint": snapshot.fingerprint,
        "dataset_checksum": source_checksum,
        "configuration_sha256": configuration_checksum,
        "random_seed": random_seed,
        "source_revision": source_revision,
        "snapshot_path": str(
            (store_root / "snapshots" / f"stage-{percentage:03d}.json").relative_to(store_root)
        ),
        "snapshot_sha256": "",
        "artifact_id": artifact.artifact_id,
        "artifact_path": str(artifact_path.relative_to(store_root)),
        "artifact_sha256": _directory_checksum(artifact_path),
        "evaluation_boundary": dict(boundary),
        "evaluation": dict(evaluation_summary) if evaluation_summary is not None else None,
        "evaluation_path": (
            str(evaluation_path.relative_to(store_root)) if evaluation_path is not None else None
        ),
        "evaluation_sha256": evaluation_sha256,
        "timings": dict(timings),
        "active": percentage == 100,
        "lhf_training": {
            mode: dict(artifact.manifest.get("fusion_training", {}).get(mode, {}))
            for mode in ("known_user", "history_only")
        },
    }
    snapshot_path = store_root / record["snapshot_path"]
    record["snapshot_sha256"] = sha256_bytes(snapshot_path.read_bytes())
    record["record_sha256"] = _checksum_without_key(record, "record_sha256")
    return record


def _write_stage_snapshot(output_dir: Path, percentage: int, snapshot: DataSnapshot) -> Path:
    path = output_dir / "snapshots" / f"stage-{percentage:03d}.json"
    if path.exists():
        try:
            existing = _read_json_object(path)
            if existing.get("fingerprint") == snapshot.fingerprint:
                return path
        except Exception:
            pass
        _quarantine_path(path, output_dir)
    _write_json_exclusive(path, snapshot.to_dict())
    return path


def _write_evaluation_record(
    output_dir: Path,
    percentage: int,
    record: Mapping[str, Any],
) -> tuple[Path, str]:
    path = output_dir / "evaluations" / f"stage-{percentage:03d}.json"
    if path.exists():
        _quarantine_path(path, output_dir)
    payload = dict(record)
    _write_json_exclusive(path, payload)
    return path, sha256_bytes(path.read_bytes())


def _write_stage_record(
    output_dir: Path,
    percentage: int,
    record: Mapping[str, Any],
) -> tuple[Path, str]:
    path = output_dir / "stages" / f"stage-{percentage:03d}.json"
    if path.exists():
        _quarantine_path(path, output_dir)
    _write_json_exclusive(path, record)
    return path, sha256_bytes(path.read_bytes())


def _stage_result_from_record(record: Mapping[str, Any], output_dir: Path) -> RollingStageResult:
    boundary = record.get("evaluation_boundary")
    window: tuple[int, int] | None = None
    if isinstance(boundary, Mapping):
        start = boundary.get("future_window_start_percentage")
        end = boundary.get("future_window_end_percentage")
        if isinstance(start, int) and isinstance(end, int):
            window = (start, end)
    artifact_id = str(record["artifact_id"])
    evaluation = record.get("evaluation")
    return RollingStageResult(
        percentage=int(record["stage_percentage"]),
        snapshot_event_count=int(record["snapshot_event_count"]),
        source_event_count=int(record["source_event_count"]),
        snapshot_fingerprint=str(record["snapshot_fingerprint"]),
        artifact_id=artifact_id,
        artifact_path=output_dir / artifact_id,
        active=bool(record["active"]),
        evaluation_window=window,
        evaluation=evaluation if isinstance(evaluation, Mapping) else None,
        timings={
            str(key): float(value)
            for key, value in dict(record.get("timings", {})).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        lhf_training={
            str(mode): dict(metadata)
            for mode, metadata in dict(record.get("lhf_training", {})).items()
            if isinstance(metadata, Mapping)
        },
    )


def _report_stage_payload(stage: RollingStageResult, completed: _CompletedStage) -> dict[str, Any]:
    return {
        "percentage": stage.percentage,
        "snapshot_event_count": stage.snapshot_event_count,
        "source_event_count": stage.source_event_count,
        "snapshot_fingerprint": stage.snapshot_fingerprint,
        "artifact_id": stage.artifact_id,
        "active": stage.active,
        "evaluation_window": list(stage.evaluation_window) if stage.evaluation_window else None,
        "evaluation": stage.evaluation,
        "timings": dict(stage.timings),
        "lhf_training": dict(stage.lhf_training),
        "artifact_manifest": completed.manifest,
    }


def _observe(observer: PhaseObserver | None, phase: str, percentage: int) -> None:
    if observer is not None:
        observer(phase, percentage)


def _read_json_object(path: Path) -> dict[str, Any]:
    return _read_json_object_bytes(path.read_bytes())


def _read_json_object_bytes(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise RollingLifecycleError("rolling JSON record must be an object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _checksum_without_key(value: Mapping[str, Any], key: str) -> str:
    return sha256_bytes(
        canonical_json_bytes({name: child for name, child in value.items() if name != key})
    )


def _directory_checksum(path: Path) -> str:
    digest = sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_relative_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise RollingLifecycleError("rolling record contains an invalid relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RollingLifecycleError("rolling record path escapes its output directory") from error
    return path


def _quarantine_path(path: Path, root: Path) -> None:
    if not path.exists():
        return
    quarantine = root / ".rolling-quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    destination = quarantine / f"{path.name}-{uuid4().hex}"
    os.replace(path, destination)
