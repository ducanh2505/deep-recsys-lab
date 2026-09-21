from __future__ import annotations

import json
from collections.abc import Sequence
from time import monotonic
from typing import Any, Protocol

from .event_store import AppendOnlyEventStore
from .models import RatingEvent


class KafkaBoundary(Protocol):
    """The small Kafka seam shared by the real broker and deterministic tests."""

    def publish(self, topic: str, events: Sequence[RatingEvent]) -> None: ...

    def consume_to_store(
        self,
        topic: str,
        group_id: str,
        expected_count: int,
        store: AppendOnlyEventStore,
        timeout_seconds: float = 30.0,
    ) -> None: ...


class InMemoryKafkaBoundary:
    """A deterministic Kafka-shaped boundary for lifecycle tests."""

    def __init__(self) -> None:
        self._topics: dict[str, list[RatingEvent]] = {}
        self._committed_offsets: dict[tuple[str, str], int] = {}

    def publish(self, topic: str, events: Sequence[RatingEvent]) -> None:
        self._topics.setdefault(topic, []).extend(events)

    def consume_to_store(
        self,
        topic: str,
        group_id: str,
        expected_count: int,
        store: AppendOnlyEventStore,
        timeout_seconds: float = 30.0,
    ) -> None:
        del timeout_seconds
        start = self._committed_offsets.get((topic, group_id), 0)
        available = self._topics.get(topic, [])[start:]
        if len(available) < expected_count:
            raise TimeoutError(
                f"Kafka topic {topic!r} has {len(available)} messages; expected {expected_count}"
            )

        batch = available[:expected_count]
        store.append_batch(batch)
        # At-least-once: the offset advances only after the append succeeds.
        self._committed_offsets[(topic, group_id)] = start + expected_count


class ConfluentKafkaBoundary:
    """A real single-broker Kafka boundary with manual post-append commits."""

    def __init__(self, bootstrap_servers: str) -> None:
        self.bootstrap_servers = bootstrap_servers

    def publish(self, topic: str, events: Sequence[RatingEvent]) -> None:
        try:
            from confluent_kafka import Producer
        except ImportError as error:  # pragma: no cover - dependency is project-managed
            raise RuntimeError("confluent-kafka is required for the real Kafka boundary") from error

        producer = Producer({"bootstrap.servers": self.bootstrap_servers})
        delivery_errors: list[str] = []

        def delivery_callback(error: Any, _message: Any) -> None:
            if error is not None:
                delivery_errors.append(str(error))

        for event in events:
            producer.produce(
                topic=topic,
                key=event.event_id,
                value=json.dumps(event.to_dict(), sort_keys=True),
                callback=delivery_callback,
            )

        remaining = producer.flush(30.0)
        if remaining:
            raise TimeoutError(f"Kafka producer still has {remaining} message(s) in flight")
        if delivery_errors:
            raise RuntimeError(f"Kafka delivery failed: {delivery_errors[0]}")

    def consume_to_store(
        self,
        topic: str,
        group_id: str,
        expected_count: int,
        store: AppendOnlyEventStore,
        timeout_seconds: float = 30.0,
    ) -> None:
        try:
            from confluent_kafka import Consumer, KafkaError
        except ImportError as error:  # pragma: no cover - dependency is project-managed
            raise RuntimeError("confluent-kafka is required for the real Kafka boundary") from error

        consumer = Consumer(
            {
                "bootstrap.servers": self.bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        consumer.subscribe([topic])
        messages: list[RatingEvent] = []
        deadline = monotonic() + timeout_seconds
        try:
            while len(messages) < expected_count:
                if monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for {expected_count} Kafka messages on {topic!r}"
                    )
                message = consumer.poll(min(1.0, max(0.01, deadline - monotonic())))
                if message is None:
                    continue
                message_error = message.error()
                if message_error is not None:
                    if message_error.code() == KafkaError._PARTITION_EOF:
                        continue
                    raise RuntimeError(f"Kafka consumption failed: {message_error}")
                raw_value = message.value()
                if not isinstance(raw_value, bytes):
                    raise TypeError("Kafka Rating Event payload must be bytes")
                payload = json.loads(raw_value.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise TypeError("Kafka Rating Event payload must be a JSON object")
                messages.append(RatingEvent.from_dict(payload))

            # Commit only after the append. A crash before this point causes replay,
            # which the deterministic Event ID makes safe at snapshot materialization.
            store.append_batch(messages)
            consumer.commit(asynchronous=False)
        finally:
            consumer.close()
