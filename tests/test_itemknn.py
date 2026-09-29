from collections import defaultdict
from math import sqrt

import numpy as np
import pytest

from deep_recsys_lifecycle.event_store import DataSnapshot
from deep_recsys_lifecycle.itemknn import fit_itemknn
from deep_recsys_lifecycle.models import Candidate, RatingEvent
from deep_recsys_lifecycle.positive import derive_positive_interactions
from deep_recsys_lifecycle.serving import ItemKNNRetriever
from deep_recsys_lifecycle.sparse_scoring import SparseItemKNNScorer


def _snapshot() -> DataSnapshot:
    events = (
        RatingEvent.from_movielens(1, 1, 5.0, 1),
        RatingEvent.from_movielens(1, 2, 5.0, 2),
        RatingEvent.from_movielens(2, 1, 5.0, 3),
        RatingEvent.from_movielens(2, 3, 5.0, 4),
        RatingEvent.from_movielens(3, 2, 5.0, 5),
        RatingEvent.from_movielens(3, 3, 5.0, 6),
        RatingEvent.from_movielens(4, 4, 3.0, 7),
    )
    return DataSnapshot.from_events(events)


def test_itemknn_uses_binary_cosine_scores_and_movie_id_ties() -> None:
    snapshot = _snapshot()
    model = fit_itemknn(snapshot, derive_positive_interactions(snapshot.events))

    assert model.similarity(1, 2) == 0.5
    assert model.similarity(1, 3) == 0.5
    assert [candidate.movie_id for candidate in model.candidate_pool((1,), limit=3)] == [2, 3, 4]
    assert [candidate.score for candidate in model.candidate_pool((1,), limit=3)][:2] == [0.5, 0.5]


def test_itemknn_sums_similarity_over_query_history_and_excludes_observed_movies() -> None:
    snapshot = _snapshot()
    model = fit_itemknn(snapshot, derive_positive_interactions(snapshot.events))

    candidates = model.candidate_pool((1, 2), limit=3)

    assert [candidate.movie_id for candidate in candidates] == [3, 4]
    assert candidates[0].score == 1.0


def test_itemknn_cpu_serving_payload_round_trip_and_corruption_rejection() -> None:
    snapshot = _snapshot()
    model = fit_itemknn(snapshot, derive_positive_interactions(snapshot.events))

    serving = model.to_serving()
    loaded = ItemKNNRetriever.from_dict(serving.to_dict())

    assert loaded.candidate_pool((1, 2), limit=3) == serving.candidate_pool((1, 2), limit=3)

    payload = serving.to_dict()
    payload["catalog"] = [1, 1]
    try:
        ItemKNNRetriever.from_dict(payload)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("corrupt ItemKNN payload was accepted")


def test_sparse_itemknn_scoring_matches_binary_cosine_contract() -> None:
    snapshot = _snapshot()
    serving = fit_itemknn(snapshot, derive_positive_interactions(snapshot.events)).to_serving()
    scorer = SparseItemKNNScorer(serving.catalog, serving.item_subjects, serving.subject_histories)

    for history in ((1,), (1, 2), (1, 2, 3), (999,)):
        expected = serving.candidate_pool(history, limit=4)
        actual = scorer.candidate_pool(history, limit=4)
        assert [candidate.movie_id for candidate in actual] == [
            candidate.movie_id for candidate in expected
        ]
        assert [candidate.score for candidate in actual] == pytest.approx(
            [candidate.score for candidate in expected]
        )


def test_sparse_itemknn_fast_path_preserves_exact_scores_and_tie_order() -> None:
    catalog = tuple(range(1, 513))
    item_subjects: defaultdict[int, set[int]] = defaultdict(set)
    histories: dict[int, tuple[int, ...]] = {}
    for subject_id in range(1, 101):
        movies = tuple(sorted({1 + (subject_id * 37 + offset * 53) % 512 for offset in range(20)}))
        histories[subject_id] = movies
        for movie_id in movies:
            item_subjects[movie_id].add(subject_id)
    scorer = SparseItemKNNScorer(
        catalog,
        {movie_id: frozenset(subjects) for movie_id, subjects in item_subjects.items()},
        histories,
    )

    def original_pool(history: tuple[int, ...], limit: int) -> tuple[Candidate, ...]:
        query = np.zeros(len(catalog), dtype=np.float64)
        excluded = []
        for movie_id in set(history):
            index = scorer.movie_index.get(movie_id)
            if index is not None:
                excluded.append(index)
                count = scorer.movie_counts[index]
                if count:
                    query[index] = 1.0 / sqrt(float(count))
        weights = scorer.matrix @ query
        scores = np.asarray(scorer.matrix.T @ weights).ravel()
        scores /= np.sqrt(np.maximum(scorer.movie_counts, 1.0))
        if excluded:
            scores[np.asarray(excluded, dtype=np.int32)] = -np.inf
        order = np.lexsort((scorer.movie_ids, -scores))
        order = order[np.isfinite(scores[order])][:limit]
        return tuple(
            Candidate(movie_id=int(scorer.movie_ids[index]), score=float(scores[index]), rank=rank)
            for rank, index in enumerate(order, start=1)
        )

    for history in ((), (1,), (1, 1, 2, 999), histories[42], histories[87]):
        for limit in (1, 10, 50, 200, 0, -1):
            assert scorer.candidate_pool(history, limit) == original_pool(history, limit)
