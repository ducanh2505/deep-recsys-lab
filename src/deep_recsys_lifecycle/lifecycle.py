from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .artifact import ServingArtifact
from .evaluation import EvaluationReport, build_evaluation_cohort, evaluate_retrievers
from .event_store import AppendOnlyEventStore
from .fixture import load_movielens_fixture
from .itemknn import fit_itemknn
from .kafka import KafkaBoundary
from .models import RatingEvent
from .popularity import fit_popularity
from .positive import derive_positive_interactions
from .report import write_static_report
from .temporal import TemporalSplit, split_temporal_events


@dataclass(frozen=True, slots=True)
class FastLifecycleResult:
    output_dir: Path
    event_store_dir: Path
    snapshot_path: Path
    artifact_path: Path
    report_path: Path
    future_window_path: Path
    snapshot_fingerprint: str
    replay_snapshot_fingerprint: str
    source_event_count: int
    positive_interaction_count: int
    temporal_split: TemporalSplit
    evaluation: EvaluationReport


def run_fast_lifecycle(
    output_dir: Path,
    kafka: KafkaBoundary,
    source_events: Sequence[RatingEvent] | None = None,
    topic: str | None = None,
) -> FastLifecycleResult:
    """Run the deterministic snapshot, evaluation, and Popularity serving path."""

    events = tuple(source_events) if source_events is not None else load_movielens_fixture()
    if not events:
        raise ValueError("the fast lifecycle needs at least one source Rating Event")

    output_dir.mkdir(parents=True, exist_ok=True)
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
    snapshot_interactions = derive_positive_interactions(data_snapshot.events)
    popularity = fit_popularity(data_snapshot, snapshot_interactions)
    itemknn = fit_itemknn(data_snapshot, snapshot_interactions)
    cohort = build_evaluation_cohort(data_snapshot, temporal_split.future_window_events)
    evaluation = evaluate_retrievers(
        {"popularity": popularity, "itemknn": itemknn},
        cohort,
    )
    artifact_path = output_dir / "serving_artifact"
    artifact = ServingArtifact.build(
        path=artifact_path,
        snapshot=data_snapshot,
        interactions=snapshot_interactions,
    )
    artifact.save()
    loaded_artifact = ServingArtifact.load(artifact_path)

    report_path = output_dir / "report.html"
    write_static_report(
        report_path,
        data_snapshot,
        loaded_artifact,
        temporal_split=temporal_split,
        evaluation=evaluation,
    )
    return FastLifecycleResult(
        output_dir=output_dir,
        event_store_dir=store_dir,
        snapshot_path=snapshot_path,
        artifact_path=artifact_path,
        report_path=report_path,
        future_window_path=future_window_path,
        snapshot_fingerprint=data_snapshot.fingerprint,
        replay_snapshot_fingerprint=data_snapshot.fingerprint,
        source_event_count=len(events),
        positive_interaction_count=len(snapshot_interactions),
        temporal_split=temporal_split,
        evaluation=evaluation,
    )
