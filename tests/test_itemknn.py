from deep_recsys_lifecycle.event_store import DataSnapshot
from deep_recsys_lifecycle.itemknn import fit_itemknn
from deep_recsys_lifecycle.models import RatingEvent
from deep_recsys_lifecycle.positive import derive_positive_interactions
from deep_recsys_lifecycle.serving import ItemKNNRetriever


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
