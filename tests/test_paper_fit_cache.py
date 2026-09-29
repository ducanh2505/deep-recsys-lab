from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_recsys_lifecycle.event_store import DataSnapshot
from deep_recsys_lifecycle.models import PositiveInteraction, RatingEvent
from deep_recsys_lifecycle.paper_fit_cache import FitCache
from deep_recsys_lifecycle.popularity import fit_popularity


def test_fit_cache_reuses_exact_source_and_invalidates_changed_fit_inputs(tmp_path: Path) -> None:
    events = tuple(
        RatingEvent.from_movielens(
            subject_id=subject_id, movie_id=movie_id, rating=5.0,
            event_time=subject_id * 100 + movie_id,
        )
        for subject_id in (1, 2)
        for movie_id in (1, 2, 3)
    )
    snapshot = DataSnapshot.from_events(events)
    interactions = tuple(PositiveInteraction.from_event(event) for event in events)
    cache = FitCache(tmp_path, max_bytes=1_000_000, min_free_bytes=0)
    fits = 0

    def fit():
        nonlocal fits
        fits += 1
        return fit_popularity(snapshot, interactions)

    options = {
        "source": "popularity",
        "snapshot": snapshot,
        "source_config": {"ranking": "positive_count"},
        "seed": 42,
        "device": "cpu",
        "code_fingerprint": "code-a",
        "fit": fit,
    }
    first = cache.get_or_fit(**options)
    replay = cache.get_or_fit(**options)
    assert first.candidate_pool(()) == replay.candidate_pool(())
    assert fits == 1
    cache.get_or_fit(**{**options, "source_config": {"ranking": "time_decay"}})
    cache.get_or_fit(**{**options, "code_fingerprint": "code-b"})
    assert fits == 3
    assert cache.stats == {"hits": 1, "misses": 3}

    stored = next(
        path.parent / "retriever.pkl"
        for path in (tmp_path / "entries").glob("*/manifest.json")
        if json.loads(path.read_text(encoding="utf-8"))["identity"]["source_config"]
        == {"ranking": "positive_count"}
        and json.loads(path.read_text(encoding="utf-8"))["identity"]["code_fingerprint"]
        == "code-a"
    )
    stored.write_bytes(stored.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        cache.get_or_fit(**options)
