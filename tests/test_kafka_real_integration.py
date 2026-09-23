import socket
from pathlib import Path
from uuid import uuid4

import pytest

from deep_recsys_lifecycle.event_store import AppendOnlyEventStore
from deep_recsys_lifecycle.fixture import load_movielens_fixture
from deep_recsys_lifecycle.kafka import ConfluentKafkaBoundary
from deep_recsys_lifecycle.models import RatingEvent


def _local_kafka_is_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 9092), timeout=0.2):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _local_kafka_is_reachable(),
    reason="start the official Apache Kafka Compose service for this integration test",
)
def test_real_kafka_replay_is_consumed_and_deduplicated(tmp_path: Path) -> None:
    events = load_movielens_fixture()
    topic = f"movie-recsys-test-{uuid4().hex}"
    kafka = ConfluentKafkaBoundary("localhost:9092")
    store = AppendOnlyEventStore(tmp_path / "event-store")
    first_group = f"first-{uuid4().hex}"

    kafka.publish(topic, events)
    kafka.consume_to_store(topic, first_group, len(events), store)
    first_snapshot = store.materialize_snapshot()

    next_event = RatingEvent.from_movielens(99, 999, 5.0, 10_000)
    kafka.publish(topic, (next_event,))
    kafka.consume_to_store(topic, first_group, 1, store)
    assert len(store.batch_paths) == 2
    assert store.read_all()[-1] == next_event

    kafka.consume_to_store(topic, f"replay-{uuid4().hex}", len(events) + 1, store)
    replay_snapshot = store.materialize_snapshot()

    assert len(store.batch_paths) == 3
    assert len(store.read_all()) == 2 * (len(events) + 1)
    assert {event.event_id for event in first_snapshot.events}.issubset(
        {event.event_id for event in replay_snapshot.events}
    )
    assert replay_snapshot.event_count == len(events) + 1
