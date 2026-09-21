from __future__ import annotations

import csv
from pathlib import Path

from .models import RatingEvent


def load_movielens_fixture(path: Path | None = None) -> tuple[RatingEvent, ...]:
    """Load the small deterministic MovieLens-shaped source fixture."""

    fixture_path = path or Path(__file__).with_name("fixtures") / "ratings.csv"
    with fixture_path.open(newline="", encoding="utf-8") as source:
        rows = csv.DictReader(source)
        return tuple(
            RatingEvent.from_movielens(
                subject_id=int(row["userId"]),
                movie_id=int(row["movieId"]),
                rating=float(row["rating"]),
                event_time=int(row["timestamp"]),
            )
            for row in rows
        )
