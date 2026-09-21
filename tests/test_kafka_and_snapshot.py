from pathlib import Path

from deep_recsys_lifecycle.event_store import AppendOnlyEventStore
from deep_recsys_lifecycle.fixture import load_movielens_fixture
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary


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
