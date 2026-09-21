from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .event_store import DataSnapshot
from .models import RatingEvent


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """A deterministic Data Snapshot and the Future Window evaluated against it."""

    ordered_events: tuple[RatingEvent, ...]
    snapshot_boundary: int
    future_boundary: int
    source_batch_count: int = 1

    @property
    def data_snapshot_events(self) -> tuple[RatingEvent, ...]:
        return self.ordered_events[: self.snapshot_boundary]

    @property
    def snapshot_events(self) -> tuple[RatingEvent, ...]:
        return self.data_snapshot_events

    @property
    def future_window_events(self) -> tuple[RatingEvent, ...]:
        return self.ordered_events[self.snapshot_boundary : self.future_boundary]

    @property
    def data_snapshot(self) -> DataSnapshot:
        return DataSnapshot(
            events=self.data_snapshot_events,
            source_batch_count=self.source_batch_count,
        )

    @property
    def future_window(self) -> tuple[RatingEvent, ...]:
        return self.future_window_events


def split_temporal_events(
    events: Iterable[RatingEvent],
    *,
    source_batch_count: int = 1,
    snapshot_percentage: int = 50,
    future_percentage: int = 60,
) -> TemporalSplit:
    """Sort Rating Events stably and cut the 50%/60% temporal boundaries."""

    if not 0 <= snapshot_percentage < future_percentage <= 100:
        raise ValueError("temporal percentages must satisfy 0 <= snapshot < future <= 100")
    if source_batch_count < 0:
        raise ValueError("source_batch_count cannot be negative")

    ordered_events = tuple(sorted(events, key=lambda event: (event.event_time, event.event_id)))
    event_count = len(ordered_events)
    snapshot_boundary = event_count * snapshot_percentage // 100
    future_boundary = event_count * future_percentage // 100
    return TemporalSplit(
        ordered_events=ordered_events,
        snapshot_boundary=snapshot_boundary,
        future_boundary=future_boundary,
        source_batch_count=source_batch_count,
    )
