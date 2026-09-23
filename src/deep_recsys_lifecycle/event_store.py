from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .models import RatingEvent


@dataclass(frozen=True, slots=True)
class DataSnapshot:
    """The deduplicated Rating Events materialized from append-only batches."""

    events: tuple[RatingEvent, ...]
    source_batch_count: int = field(compare=False)
    _fingerprint: str | None = field(default=None, init=False, compare=False, repr=False)

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
        if self._fingerprint is not None:
            return self._fingerprint
        digest = sha256()
        digest.update(b"[")
        for index, event in enumerate(self.events):
            if index:
                digest.update(b",")
            digest.update(
                json.dumps(
                    event.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
        digest.update(b"]")
        value = digest.hexdigest()
        object.__setattr__(self, "_fingerprint", value)
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "source_batch_count": self.source_batch_count,
            "fingerprint": self.fingerprint,
            "events": [event.to_dict() for event in self.events],
        }

    def write_json(self, path: Path) -> None:
        # Large snapshots remain reproducible from the append-only Parquet Event Store.
        # Embedding millions of duplicate source rows in a JSON sidecar serves no audit purpose.
        value = self.to_dict() if self.event_count <= 100_000 else {
            "event_count": self.event_count,
            "source_batch_count": self.source_batch_count,
            "fingerprint": self.fingerprint,
            "events_embedded": False,
        }
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
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

    def iter_event_ids(self) -> Iterator[str]:
        """Scan only the identity column when checking a large persisted prefix."""

        for batch_path in self.batch_paths:
            parquet = pq.ParquetFile(batch_path)
            for batch in parquet.iter_batches(columns=["event_id"], batch_size=65_536):
                yield from batch.column(0).to_pylist()

    def materialize_snapshot(self) -> DataSnapshot:
        paths = self.batch_paths
        if not paths:
            return DataSnapshot(events=(), source_batch_count=0)
        connection = duckdb.connect()
        temp_dir = self.root / ".duckdb-temp"
        temp_dir.mkdir(exist_ok=True)
        parquet_paths = [str(path.resolve()) for path in paths]
        try:
            connection.execute("SET memory_limit = '6GB'")
            connection.execute("SET temp_directory = ?", [str(temp_dir.resolve())])
            conflict = connection.execute(
                """
                SELECT event_id
                FROM read_parquet(?)
                GROUP BY event_id
                HAVING count(DISTINCT subject_id) > 1
                    OR count(DISTINCT movie_id) > 1
                    OR count(DISTINCT rating) > 1
                    OR count(DISTINCT event_time) > 1
                    OR count(DISTINCT source) > 1
                LIMIT 1
                """,
                [parquet_paths],
            ).fetchone()
            if conflict is not None:
                raise ValueError(f"event id {conflict[0]!r} has conflicting payloads")
            batches = connection.execute(
                """
                SELECT
                    event_id,
                    min(subject_id) AS subject_id,
                    min(movie_id) AS movie_id,
                    min(rating) AS rating,
                    min(event_time) AS event_time,
                    min(source) AS source
                FROM read_parquet(?)
                GROUP BY event_id
                ORDER BY event_time, event_id
                """,
                [parquet_paths],
            ).to_arrow_reader(batch_size=65_536)
            events: list[RatingEvent] = []
            for batch in batches:
                rows = batch.to_pydict()
                events.extend(
                    RatingEvent(
                        event_id=event_id,
                        subject_id=subject_id,
                        movie_id=movie_id,
                        rating=rating,
                        event_time=event_time,
                        source=source,
                    )
                    for event_id, subject_id, movie_id, rating, event_time, source in zip(
                        rows["event_id"],
                        rows["subject_id"],
                        rows["movie_id"],
                        rows["rating"],
                        rows["event_time"],
                        rows["source"],
                        strict=True,
                    )
                )
            return DataSnapshot(events=tuple(events), source_batch_count=len(paths))
        finally:
            connection.close()
            shutil.rmtree(temp_dir, ignore_errors=True)
