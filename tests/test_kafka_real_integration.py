import socket
from pathlib import Path
from uuid import uuid4

import pytest

from deep_recsys_lifecycle.event_store import AppendOnlyEventStore
from deep_recsys_lifecycle.fixture import load_movielens_fixture
from deep_recsys_lifecycle.kafka import ConfluentKafkaBoundary


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

    kafka.publish(topic, events)
    kafka.consume_to_store(topic, f"first-{uuid4().hex}", len(events), store)
    first_snapshot = store.materialize_snapshot()

    kafka.publish(topic, events)
    kafka.consume_to_store(topic, f"replay-{uuid4().hex}", len(events), store)
    replay_snapshot = store.materialize_snapshot()

    assert replay_snapshot.fingerprint == first_snapshot.fingerprint
    assert replay_snapshot.event_count == len(events)
