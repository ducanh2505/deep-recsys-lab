from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any


def deterministic_event_id(
    subject_id: int,
    movie_id: int,
    rating: float,
    event_time: int,
    source: str = "movielens",
) -> str:
    """Return a stable identity for one source Rating Event."""

    canonical = "\x1f".join(
        (
            source,
            str(subject_id),
            str(movie_id),
            format(rating, ".17g"),
            str(event_time),
        )
    )
    return f"{source}:{sha256(canonical.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class RatingEvent:
    """An immutable MovieLens rating observation."""

    event_id: str
    subject_id: int
    movie_id: int
    rating: float
    event_time: int
    source: str = "movielens"

    @classmethod
    def from_movielens(
        cls,
        subject_id: int,
        movie_id: int,
        rating: float,
        event_time: int,
        source: str = "movielens",
    ) -> RatingEvent:
        return cls(
            event_id=deterministic_event_id(
                subject_id=subject_id,
                movie_id=movie_id,
                rating=rating,
                event_time=event_time,
                source=source,
            ),
            subject_id=subject_id,
            movie_id=movie_id,
            rating=float(rating),
            event_time=event_time,
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "subject_id": self.subject_id,
            "movie_id": self.movie_id,
            "rating": self.rating,
            "event_time": self.event_time,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RatingEvent:
        return cls(
            event_id=str(value["event_id"]),
            subject_id=int(value["subject_id"]),
            movie_id=int(value["movie_id"]),
            rating=float(value["rating"]),
            event_time=int(value["event_time"]),
            source=str(value.get("source", "movielens")),
        )


@dataclass(frozen=True, slots=True)
class PositiveInteraction:
    """A binary preference signal derived from a Rating Event."""

    event_id: str
    subject_id: int
    movie_id: int
    rating: float
    event_time: int

    @classmethod
    def from_event(cls, event: RatingEvent) -> PositiveInteraction:
        return cls(
            event_id=event.event_id,
            subject_id=event.subject_id,
            movie_id=event.movie_id,
            rating=event.rating,
            event_time=event.event_time,
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    movie_id: int
    score: int | float
    rank: int

    def to_dict(self) -> dict[str, int | float]:
        return {"movie_id": self.movie_id, "score": self.score, "rank": self.rank}
