import json
from hashlib import sha256
from pathlib import Path

import pytest

from deep_recsys_lifecycle.event_store import AppendOnlyEventStore, DataSnapshot
from deep_recsys_lifecycle.fixture import load_movielens_fixture
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
from deep_recsys_lifecycle.models import RatingEvent


def test_replaying_events_appends_a_batch_but_not_a_second_snapshot_event(
    tmp_path: Path,
) -> None:
    events = load_movielens_fixture()
    kafka = InMemoryKafkaBoundary()
    store = AppendOnlyEventStore(tmp_path / "event-store")

    kafka.publish("ratings", events)
    kafka.consume_to_store("ratings", "first-consumer", len(events), store)
    first_snapshot = store.materialize_snapshot()

    kafka.publish("ratings", events)
    kafka.consume_to_store("ratings", "replay-consumer", len(events), store)
    replay_snapshot = store.materialize_snapshot()

    assert len(store.batch_paths) == 2
    assert len(store.read_all()) == len(events) * 2
    assert replay_snapshot == first_snapshot
    assert replay_snapshot.event_count == first_snapshot.event_count == len(events)
    assert replay_snapshot.fingerprint == first_snapshot.fingerprint
    assert {event.event_id for event in replay_snapshot.events} == {
        event.event_id for event in events
    }
    assert {event.event_id: event.rating for event in replay_snapshot.events} == {
        event.event_id: event.rating for event in events
    }


def test_large_snapshot_sidecar_keeps_fingerprint_without_embedding_events(
    tmp_path: Path,
) -> None:
    event = load_movielens_fixture()[0]
    snapshot = DataSnapshot.from_events((event,) * 100_001)
    expected = sha256(json.dumps(
        [event.to_dict()] * 100_001, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    path = tmp_path / "snapshot.json"

    snapshot.write_json(path)

    sidecar = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot.fingerprint == expected
    assert sidecar["fingerprint"] == expected
    assert sidecar["event_count"] == 100_001
    assert sidecar["events_embedded"] is False
    assert "events" not in sidecar


def test_snapshot_materialization_rejects_conflicting_event_payloads(tmp_path: Path) -> None:
    event = load_movielens_fixture()[0]
    conflict = RatingEvent(
        event_id=event.event_id,
        subject_id=event.subject_id,
        movie_id=event.movie_id,
        rating=event.rating - 1.0,
        event_time=event.event_time,
        source=event.source,
    )
    store = AppendOnlyEventStore(tmp_path / "event-store")
    store.append_batch([event])
    store.append_batch([conflict])

    with pytest.raises(ValueError, match="conflicting payloads"):
        store.materialize_snapshot()
