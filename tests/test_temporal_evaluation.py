from pathlib import Path

from deep_recsys_lifecycle.evaluation import build_evaluation_cohort
from deep_recsys_lifecycle.event_store import DataSnapshot
from deep_recsys_lifecycle.fixture import load_movielens_fixture
from deep_recsys_lifecycle.itemknn import fit_itemknn
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
from deep_recsys_lifecycle.lifecycle import run_fast_lifecycle
from deep_recsys_lifecycle.models import RatingEvent
from deep_recsys_lifecycle.popularity import fit_popularity
from deep_recsys_lifecycle.positive import derive_positive_interactions
from deep_recsys_lifecycle.temporal import split_temporal_events


def test_temporal_boundaries_sort_by_event_time_then_event_id() -> None:
    events = (
        RatingEvent.from_movielens(1, 12, 4.0, 2),
        RatingEvent.from_movielens(1, 11, 4.0, 1),
        RatingEvent.from_movielens(1, 10, 4.0, 1),
        RatingEvent.from_movielens(2, 10, 4.0, 3),
        RatingEvent.from_movielens(2, 11, 4.0, 4),
    )

    split = split_temporal_events(events)

    assert split.snapshot_boundary == 2
    assert split.future_boundary == 3
    assert split.ordered_events == tuple(
        sorted(events, key=lambda event: (event.event_time, event.event_id))
    )
    assert split.data_snapshot_events == split.ordered_events[:2]
    assert split.future_window_events == split.ordered_events[2:3]


def test_cohort_uses_first_future_positive_and_snapshot_history_only() -> None:
    snapshot_events = (RatingEvent.from_movielens(1, 10, 5.0, 1),)
    future_events = (
        RatingEvent.from_movielens(1, 11, 3.0, 2),
        RatingEvent.from_movielens(1, 12, 5.0, 3),
        RatingEvent.from_movielens(2, 13, 5.0, 4),
    )

    cohort = build_evaluation_cohort(DataSnapshot.from_events(snapshot_events), future_events)

    assert cohort.size == 1
    assert cohort.queries[0].subject_id == 1
    assert cohort.queries[0].history == (10,)
    assert cohort.queries[0].gold_movie_id == 12


def test_fast_lifecycle_reports_the_temporal_evaluation_stage(tmp_path: Path) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())

    split = result.temporal_split
    assert split.snapshot_boundary == result.source_event_count * 50 // 100
    assert split.future_boundary == result.source_event_count * 60 // 100
    assert len(split.data_snapshot_events) == split.snapshot_boundary
    assert len(split.future_window_events) == split.future_boundary - split.snapshot_boundary
    assert result.evaluation.cohort.size > 0

    report = result.report_path.read_text(encoding="utf-8")
    assert 'id="future-window"' in report
    assert 'id="evaluation-cohort"' in report
    assert 'id="popularity-metrics"' in report
    assert 'id="itemknn-metrics"' in report
    assert 'id="rrf-metrics"' in report
    assert 'id="oracle-union"' in report
    assert "Data Snapshot 50%" in report
    assert "Future Window 50–60%" in report
    assert "Coverage@200" in report
    assert "ConditionalRecall@10" in report
    assert "EndToEndRecall@10" in report
    assert "NDCG@10" in report
    assert "Diagnostic-only" in report


def test_future_window_changes_do_not_change_fitted_retriever_state() -> None:
    events = load_movielens_fixture()
    original_split = split_temporal_events(events)
    changed_events = events[: original_split.snapshot_boundary] + (
        RatingEvent.from_movielens(1, 999, 5.0, 1021),
        *events[original_split.snapshot_boundary + 1 :],
    )
    changed_split = split_temporal_events(changed_events)
    original_interactions = derive_positive_interactions(original_split.data_snapshot.events)
    changed_interactions = derive_positive_interactions(changed_split.data_snapshot.events)

    assert fit_popularity(original_split.data_snapshot, original_interactions) == fit_popularity(
        changed_split.data_snapshot, changed_interactions
    )
    assert fit_itemknn(original_split.data_snapshot, original_interactions) == fit_itemknn(
        changed_split.data_snapshot, changed_interactions
    )


def test_future_events_are_absent_from_snapshot_histories_and_item_statistics() -> None:
    events = load_movielens_fixture()
    split = split_temporal_events(events)
    changed_events = events[: split.snapshot_boundary] + (
        RatingEvent.from_movielens(1, 999, 5.0, 1021),
        *events[split.snapshot_boundary + 1 :],
    )
    changed_split = split_temporal_events(changed_events)
    interactions = derive_positive_interactions(changed_split.data_snapshot.events)
    popularity = fit_popularity(changed_split.data_snapshot, interactions)
    itemknn = fit_itemknn(changed_split.data_snapshot, interactions)

    assert 999 not in popularity.catalog
    assert 999 not in popularity.counts
    assert all(999 not in history for history in popularity.subject_histories.values())
    assert 999 not in itemknn.catalog
    assert 999 not in itemknn.item_subjects


def test_popularity_and_itemknn_share_cohort_gold_candidates_and_history_exclusion(
    tmp_path: Path,
) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    evaluation = result.evaluation

    popularity = evaluation.retrievers["popularity"]
    itemknn = evaluation.retrievers["itemknn"]
    assert (
        set(popularity.pools)
        == set(itemknn.pools)
        == {query.subject_id for query in evaluation.cohort}
    )
    for query in evaluation.cohort:
        assert {candidate.movie_id for candidate in popularity.pools[query.subject_id]}.isdisjoint(
            query.history
        )
        assert {candidate.movie_id for candidate in itemknn.pools[query.subject_id]}.isdisjoint(
            query.history
        )
