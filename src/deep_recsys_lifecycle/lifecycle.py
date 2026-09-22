from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from .artifact import ArtifactStore, ServingArtifact
from .evaluation import EvaluationReport, build_evaluation_cohort, evaluate_retrievers
from .event_store import AppendOnlyEventStore
from .fixture import load_movielens_fixture
from .fusion import (
    FUSION_BANKS,
    FusionFeatureBuilder,
    build_lhf_training_rows,
    rank_lhf_union,
    train_lhf_classifier,
)
from .itemknn import fit_itemknn
from .kafka import KafkaBoundary
from .lightgcn import LightGCNConfig, fit_lightgcn
from .models import RatingEvent
from .multivae import fit_multivae
from .popularity import fit_popularity
from .positive import derive_positive_interactions
from .report import write_static_report
from .retriever import CandidateRetriever
from .serving import LightGCNRetriever, MultVAERetriever
from .temporal import TemporalSplit, split_temporal_events
from .validation import validate_smoke_queries


@dataclass(frozen=True, slots=True)
class FastLifecycleResult:
    output_dir: Path
    event_store_dir: Path
    snapshot_path: Path
    artifact_path: Path
    artifact_id: str
    artifact_store_path: Path
    active_pointer_path: Path
    report_path: Path
    future_window_path: Path
    snapshot_fingerprint: str
    replay_snapshot_fingerprint: str
    source_event_count: int
    positive_interaction_count: int
    temporal_split: TemporalSplit
    evaluation: EvaluationReport
    history_only_evaluation: EvaluationReport
    inner_validation_snapshot_fingerprint: str
    inner_validation_event_ids: tuple[str, ...]
    multivae: MultVAERetriever
    lightgcn: LightGCNRetriever


def run_fast_lifecycle(
    output_dir: Path,
    kafka: KafkaBoundary,
    source_events: Sequence[RatingEvent] | None = None,
    topic: str | None = None,
    *,
    source_revision: str | None = None,
    smoke_validator: Callable[[ServingArtifact], None] | None = None,
    random_seed: int = 42,
    lightgcn_config: LightGCNConfig | None = None,
    # The fast showcase uses CPU by default so a host-reported-but-unstable MPS benchmark
    # cannot crash the lifecycle process.  Direct LightGCN callers retain the auto probe seam.
    lightgcn_device_preference: str = "cpu",
    lightgcn_mps_probe: Callable[[], bool] | None = None,
    lightgcn_device_benchmark: Callable[[str], float] | None = None,
) -> FastLifecycleResult:
    """Run ingest → evaluate → export → validate → smoke → activate → report."""

    events = tuple(source_events) if source_events is not None else load_movielens_fixture()
    if not events:
        raise ValueError("the fast lifecycle needs at least one source Rating Event")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_store = ArtifactStore(output_dir)
    run_id = uuid4().hex[:12]
    kafka_topic = topic or f"movie-recsys-fast-{run_id}"
    store_dir = output_dir / "event_store"
    store = AppendOnlyEventStore(store_dir)

    kafka.publish(kafka_topic, events)
    kafka.consume_to_store(
        topic=kafka_topic,
        group_id=f"{run_id}-initial",
        expected_count=len(events),
        store=store,
    )
    initial_snapshot = store.materialize_snapshot()

    kafka.publish(kafka_topic, events)
    kafka.consume_to_store(
        topic=kafka_topic,
        group_id=f"{run_id}-replay",
        expected_count=len(events),
        store=store,
    )
    replay_snapshot = store.materialize_snapshot()
    if initial_snapshot.fingerprint != replay_snapshot.fingerprint:
        raise RuntimeError("replaying Rating Events changed the materialized Data Snapshot")

    temporal_split = split_temporal_events(
        replay_snapshot.events, source_batch_count=replay_snapshot.source_batch_count
    )
    data_snapshot = temporal_split.data_snapshot
    snapshot_path = output_dir / "data_snapshot.json"
    data_snapshot.write_json(snapshot_path)
    future_window_path = output_dir / "future_window.json"
    future_window_path.write_text(
        json.dumps(
            {
                "event_count": len(temporal_split.future_window_events),
                "events": [event.to_dict() for event in temporal_split.future_window_events],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    training_started = monotonic()
    snapshot_interactions = derive_positive_interactions(data_snapshot.events)

    # LHF supervision is deliberately nested inside the 50% Data Snapshot.  Base retrievers
    # are fitted on the prefix, pools/features are built there, and only naturally covered
    # inner-validation positives become labels.
    inner_split = split_temporal_events(
        data_snapshot.events,
        source_batch_count=data_snapshot.source_batch_count,
        snapshot_percentage=80,
        future_percentage=100,
    )
    inner_snapshot = inner_split.data_snapshot
    inner_interactions = derive_positive_interactions(inner_snapshot.events)
    inner_retrievers = _fit_retrievers(
        inner_snapshot,
        inner_interactions,
        random_seed=random_seed,
        lightgcn_config=lightgcn_config,
    )
    inner_known_cohort = build_evaluation_cohort(
        inner_snapshot,
        inner_split.future_window_events,
        query_mode="known_user",
    )
    inner_history_cohort = build_evaluation_cohort(
        inner_snapshot,
        inner_split.future_window_events,
        query_mode="history_only",
    )
    inner_known_base = evaluate_retrievers(
        inner_retrievers,
        inner_known_cohort,
    )
    inner_history_base = evaluate_retrievers(
        {name: inner_retrievers[name] for name in FUSION_BANKS["history_only"]},
        inner_history_cohort,
    )
    inner_known_builder = FusionFeatureBuilder.from_snapshot(
        inner_snapshot,
        inner_interactions,
        retriever_bank=FUSION_BANKS["known_user"],
    )
    inner_history_builder = FusionFeatureBuilder.from_snapshot(
        inner_snapshot,
        inner_interactions,
        retriever_bank=FUSION_BANKS["history_only"],
    )
    known_training_pools = _training_pools(inner_known_base)
    history_training_pools = _training_pools(inner_history_base)
    inner_validation_event_ids = tuple(event.event_id for event in inner_split.future_window_events)
    known_rows = build_lhf_training_rows(
        inner_known_cohort,
        known_training_pools,
        inner_known_builder,
    )
    history_rows = build_lhf_training_rows(
        inner_history_cohort,
        history_training_pools,
        inner_history_builder,
    )
    known_fusion = train_lhf_classifier(
        query_mode="known_user",
        retriever_bank=FUSION_BANKS["known_user"],
        feature_names=inner_known_builder.feature_names,
        rows=known_rows,
        source_snapshot_fingerprint=inner_snapshot.fingerprint,
        validation_event_ids=inner_validation_event_ids,
        seed=random_seed,
    )
    history_fusion = train_lhf_classifier(
        query_mode="history_only",
        retriever_bank=FUSION_BANKS["history_only"],
        feature_names=inner_history_builder.feature_names,
        rows=history_rows,
        source_snapshot_fingerprint=inner_snapshot.fingerprint,
        validation_event_ids=inner_validation_event_ids,
        seed=random_seed,
    )

    # Refit every base retriever on the complete 50% snapshot before touching the 50–60%
    # Future Window.  No Future Window value enters the fitted state or feature builder.
    popularity = fit_popularity(data_snapshot, snapshot_interactions)
    itemknn = fit_itemknn(data_snapshot, snapshot_interactions)
    # Keep the deterministic fast showcase on CPU; direct vertical-slice callers retain the
    # MPS preference/fallback seam, while this process must not depend on unstable host MPS ops.
    multivae = fit_multivae(
        data_snapshot,
        snapshot_interactions,
        seed=random_seed,
        device_preference="cpu",
    )
    lightgcn = fit_lightgcn(
        data_snapshot,
        snapshot_interactions,
        config=lightgcn_config or LightGCNConfig(seed=random_seed),
        seed=random_seed,
        device_preference=lightgcn_device_preference,
        mps_probe=lightgcn_mps_probe,
        device_benchmark=lightgcn_device_benchmark,
    )
    cohort = build_evaluation_cohort(data_snapshot, temporal_split.future_window_events)
    retrievers: dict[str, CandidateRetriever] = {
        "popularity": popularity,
        "itemknn": itemknn,
        "multivae": multivae,
        "lightgcn": lightgcn,
    }
    history_retrievers: dict[str, CandidateRetriever] = {
        name: retrievers[name] for name in FUSION_BANKS["history_only"]
    }
    evaluation_cohort = cohort
    history_only_cohort = build_evaluation_cohort(
        data_snapshot,
        temporal_split.future_window_events,
        query_mode="history_only",
    )
    known_base_evaluation = evaluate_retrievers(
        retrievers,
        evaluation_cohort,
        selected_best_single_name=inner_known_base.best_single_name,
        best_single_source="inner_validation",
    )
    history_base_evaluation = evaluate_retrievers(
        history_retrievers,
        history_only_cohort,
        selected_best_single_name=inner_history_base.best_single_name,
        best_single_source="inner_validation",
    )
    full_known_builder = FusionFeatureBuilder.from_snapshot(
        data_snapshot,
        snapshot_interactions,
        retriever_bank=FUSION_BANKS["known_user"],
    )
    full_history_builder = FusionFeatureBuilder.from_snapshot(
        data_snapshot,
        snapshot_interactions,
        retriever_bank=FUSION_BANKS["history_only"],
    )
    known_lhf_pools = _fused_pools(
        known_base_evaluation,
        known_fusion,
        full_known_builder,
    )
    history_lhf_pools = _fused_pools(
        history_base_evaluation,
        history_fusion,
        full_history_builder,
    )
    evaluation = evaluate_retrievers(
        retrievers,
        evaluation_cohort,
        selected_best_single_name=inner_known_base.best_single_name,
        best_single_source="inner_validation",
        lhf_pools=known_lhf_pools,
    )
    history_only_evaluation = evaluate_retrievers(
        history_retrievers,
        history_only_cohort,
        selected_best_single_name=inner_history_base.best_single_name,
        best_single_source="inner_validation",
        lhf_pools=history_lhf_pools,
    )
    training_seconds = monotonic() - training_started
    evaluation_metrics: dict[str, Any] = {
        name: retriever.metrics.to_dict() for name, retriever in evaluation.retrievers.items()
    }
    evaluation_metrics["rrf"] = evaluation.rrf.metrics.to_dict()
    evaluation_metrics["oracle_union_coverage"] = evaluation.oracle_union_coverage
    evaluation_metrics["lhf"] = evaluation.lhf.metrics.to_dict() if evaluation.lhf else {}
    evaluation_metrics["best_single"] = (
        evaluation.best_single.metrics.to_dict() if evaluation.best_single is not None else {}
    )
    evaluation_metrics["known_user"] = _evaluation_summary(evaluation)
    evaluation_metrics["history_only"] = _evaluation_summary(history_only_evaluation)

    staging_path = artifact_store.new_staging_path()
    try:
        artifact = ServingArtifact.build(
            path=staging_path,
            snapshot=data_snapshot,
            interactions=snapshot_interactions,
            source_events=events,
            data_cutoff_percentage=50,
            random_seed=random_seed,
            timings={"training": training_seconds},
            evaluation_metrics=evaluation_metrics,
            source_revision=source_revision,
            itemknn_model=itemknn.to_serving(),
            multivae_model=multivae,
            lightgcn_model=lightgcn,
            known_user_fusion=known_fusion,
            history_only_fusion=history_fusion,
            configuration={
                "fusion_training_boundary": "inner_validation_before_future_window",
                "inner_validation_snapshot_fingerprint": inner_snapshot.fingerprint,
                "inner_validation_event_ids": list(inner_validation_event_ids),
            },
        )
        artifact.save()
        staged_artifact = ServingArtifact.load(staging_path)
        smoke_started = monotonic()
        (smoke_validator or validate_smoke_queries)(staged_artifact)
        smoke_seconds = monotonic() - smoke_started
        staged_artifact = artifact_store.update_staging_timings(
            staging_path,
            {"smoke": smoke_seconds},
        )
        published_artifact_path = artifact_store.publish(staging_path)
        loaded_artifact = artifact_store._activate_loaded(published_artifact_path)
    except BaseException:
        if staging_path.exists():
            artifact_store.discard_staging(staging_path)
        raise

    report_path = output_dir / "report.html"
    write_static_report(
        report_path,
        data_snapshot,
        loaded_artifact,
        temporal_split=temporal_split,
        evaluation=evaluation,
        history_only_evaluation=history_only_evaluation,
    )
    return FastLifecycleResult(
        output_dir=output_dir,
        event_store_dir=store_dir,
        snapshot_path=snapshot_path,
        artifact_path=published_artifact_path,
        artifact_id=loaded_artifact.artifact_id,
        artifact_store_path=artifact_store.root,
        active_pointer_path=artifact_store.active_pointer_path,
        report_path=report_path,
        future_window_path=future_window_path,
        snapshot_fingerprint=data_snapshot.fingerprint,
        replay_snapshot_fingerprint=data_snapshot.fingerprint,
        source_event_count=len(events),
        positive_interaction_count=len(snapshot_interactions),
        temporal_split=temporal_split,
        evaluation=evaluation,
        history_only_evaluation=history_only_evaluation,
        inner_validation_snapshot_fingerprint=inner_snapshot.fingerprint,
        inner_validation_event_ids=inner_validation_event_ids,
        multivae=multivae,
        lightgcn=lightgcn,
    )


def _fit_retrievers(
    snapshot: Any,
    interactions: Sequence[Any],
    *,
    random_seed: int,
    lightgcn_config: LightGCNConfig | None,
) -> dict[str, CandidateRetriever]:
    popularity = fit_popularity(snapshot, interactions)
    itemknn = fit_itemknn(snapshot, interactions)
    multivae = fit_multivae(
        snapshot,
        interactions,
        seed=random_seed,
        device_preference="cpu",
    )
    lightgcn = fit_lightgcn(
        snapshot,
        interactions,
        config=lightgcn_config or LightGCNConfig(seed=random_seed),
        seed=random_seed,
        device_preference="cpu",
    )
    return {
        "popularity": popularity,
        "itemknn": itemknn,
        "multivae": multivae,
        "lightgcn": lightgcn,
    }


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
    pools: dict[int | str, tuple[Any, ...]] = {}
    for query in report.cohort:
        query_pools = {
            name: report.retrievers[name].pools[query.pool_key] for name in fusion.retriever_bank
        }
        pools[query.pool_key] = rank_lhf_union(
            fusion,
            builder,
            query_pools,
            query.history,
        )
    return pools


def _evaluation_summary(report: EvaluationReport) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "best_single": report.best_single.metrics.to_dict()
        if report.best_single is not None
        else {},
        "oracle_union": report.oracle_union.metrics.to_dict()
        if report.oracle_union is not None
        else {},
        "oracle_headroom_denominator": report.oracle_headroom_denominator,
        # Metric manifests are numeric-only; -1 marks the explicitly undefined zero-denominator
        # case, while the HTML report carries the human-readable explanation.
        "oracle_headroom_realized": report.oracle_headroom_realized
        if report.oracle_headroom_realized is not None
        else -1.0,
        "oracle_headroom_undefined": float(report.oracle_headroom_realized is None),
        "rrf": report.rrf.metrics.to_dict(),
        "retrievers": {
            name: evaluation.metrics.to_dict() for name, evaluation in report.retrievers.items()
        },
    }
    if report.lhf is not None:
        summary["lhf"] = report.lhf.metrics.to_dict()
    return summary
