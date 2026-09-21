from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .artifact import ServingArtifact
from .event_store import AppendOnlyEventStore
from .fixture import load_movielens_fixture
from .kafka import KafkaBoundary
from .models import RatingEvent
from .positive import derive_positive_interactions
from .report import write_static_report


@dataclass(frozen=True, slots=True)
class FastLifecycleResult:
    output_dir: Path
    event_store_dir: Path
    snapshot_path: Path
    artifact_path: Path
    report_path: Path
    snapshot_fingerprint: str
    replay_snapshot_fingerprint: str
    source_event_count: int
    positive_interaction_count: int


def run_fast_lifecycle(
    output_dir: Path,
    kafka: KafkaBoundary,
    source_events: Sequence[RatingEvent] | None = None,
    topic: str | None = None,
) -> FastLifecycleResult:
    """Run the complete deterministic Popularity path through the Kafka seam."""

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

    snapshot_path = output_dir / "data_snapshot.json"
    replay_snapshot.write_json(snapshot_path)
    interactions = derive_positive_interactions(replay_snapshot.events)
    artifact_path = output_dir / "serving_artifact"
    artifact = ServingArtifact.build(
        path=artifact_path,
        snapshot=replay_snapshot,
        interactions=interactions,
    )
    artifact.save()
    loaded_artifact = ServingArtifact.load(artifact_path)

    report_path = output_dir / "report.html"
    write_static_report(report_path, replay_snapshot, loaded_artifact)
    return FastLifecycleResult(
        output_dir=output_dir,
        event_store_dir=store_dir,
        snapshot_path=snapshot_path,
        artifact_path=artifact_path,
        report_path=report_path,
        snapshot_fingerprint=initial_snapshot.fingerprint,
        replay_snapshot_fingerprint=replay_snapshot.fingerprint,
        source_event_count=len(events),
        positive_interaction_count=len(interactions),
    )
