from deep_recsys_lifecycle.models import RatingEvent
from deep_recsys_lifecycle.positive import derive_positive_interactions


def test_only_ratings_at_or_above_four_become_positive_interactions() -> None:
    events = (
        RatingEvent.from_movielens(1, 10, 4.0, 1),
        RatingEvent.from_movielens(1, 11, 3.99, 2),
        RatingEvent.from_movielens(1, 12, 5.0, 3),
    )

    interactions = derive_positive_interactions(events)

    assert [(item.movie_id, item.rating) for item in interactions] == [(10, 4.0), (12, 5.0)]
    assert events[0].rating == 4.0
    assert events[1].rating == 3.99
