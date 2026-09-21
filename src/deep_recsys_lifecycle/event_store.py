from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .models import RatingEvent


@dataclass(frozen=True, slots=True)
class DataSnapshot:
    """The deduplicated Rating Events materialized from append-only batches."""

    events: tuple[RatingEvent, ...]
    source_batch_count: int = field(compare=False)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @classmethod
    def from_events(
        cls, events: tuple[RatingEvent, ...] | list[RatingEvent], source_batch_count: int = 1
    ) -> DataSnapshot:
        """Create a snapshot from an already ordered event partition."""

        return cls(events=tuple(events), source_batch_count=source_batch_count)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            [event.to_dict() for event in self.events],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "source_batch_count": self.source_batch_count,
            "fingerprint": self.fingerprint,
            "events": [event.to_dict() for event in self.events],
        }

    def write_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class AppendOnlyEventStore:
    """Persist consumed Rating Events as numbered, append-only Parquet batches."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def batch_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self.root.glob("batch-*.parquet")))

    def append_batch(self, events: tuple[RatingEvent, ...] | list[RatingEvent]) -> Path:
        if not events:
            raise ValueError("an Event Store batch must contain at least one Rating Event")

        next_number = len(self.batch_paths) + 1
        path = self.root / f"batch-{next_number:06d}.parquet"
        while path.exists():
            next_number += 1
            path = self.root / f"batch-{next_number:06d}.parquet"

        table = pa.Table.from_pylist([event.to_dict() for event in events])
        pq.write_table(table, path)
        return path

    def read_all(self) -> tuple[RatingEvent, ...]:
        events: list[RatingEvent] = []
        for batch_path in self.batch_paths:
            rows = pq.read_table(batch_path).to_pylist()
            events.extend(RatingEvent.from_dict(row) for row in rows)
        return tuple(events)

    def materialize_snapshot(self) -> DataSnapshot:
        unique: dict[str, RatingEvent] = {}
        for event in self.read_all():
            previous = unique.get(event.event_id)
            if previous is not None and previous != event:
                raise ValueError(f"event id {event.event_id!r} has conflicting payloads")
            unique[event.event_id] = event

        ordered = tuple(
            sorted(unique.values(), key=lambda event: (event.event_time, event.event_id))
        )
        return DataSnapshot(events=ordered, source_batch_count=len(self.batch_paths))
