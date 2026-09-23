import math

from deep_recsys_lifecycle.evaluation import (
    EvaluationCohort,
    EvaluationQuery,
    compute_metrics,
    evaluate_retrievers,
    history_length_segment,
    oracle_union,
    oracle_union_coverage,
    rrf_fuse,
)
from deep_recsys_lifecycle.models import Candidate


def test_history_segments_use_declared_boundaries() -> None:
    assert [history_length_segment(value) for value in (1, 4, 5, 19, 20, 99)] == [
        "1-4",
        "1-4",
        "5-19",
        "5-19",
        "20+",
        "20+",
    ]


def test_metrics_match_a_hand_checkable_candidate_pool_fixture() -> None:
    cohort = EvaluationCohort(
        queries=(
            EvaluationQuery(subject_id=1, history=(10,), gold_movie_id=3),
            EvaluationQuery(subject_id=2, history=(11,), gold_movie_id=4),
            EvaluationQuery(subject_id=3, history=(12,), gold_movie_id=5),
        )
    )
    pools = {
        1: (Candidate(movie_id=1, score=1, rank=1), Candidate(movie_id=3, score=1, rank=2)),
        2: (Candidate(movie_id=4, score=1, rank=1),),
        3: (Candidate(movie_id=1, score=1, rank=1),),
    }

    metrics = compute_metrics(cohort, pools)

    assert metrics.coverage_at_200 == 2 / 3
    assert metrics.conditional_recall_at_10 == 1.0
    assert metrics.end_to_end_recall_at_10 == 2 / 3
    assert math.isclose(metrics.ndcg_at_10, (1 / math.log2(3) + 1) / 3)


def test_catalog_coverage_counts_distinct_top_ten_movies_over_snapshot_catalog() -> None:
    cohort = EvaluationCohort(
        queries=(
            EvaluationQuery(subject_id=1, history=(10,), gold_movie_id=3),
            EvaluationQuery(subject_id=2, history=(11,), gold_movie_id=4),
        )
    )
    pools = {
        1: (Candidate(movie_id=10, score=3, rank=1), Candidate(movie_id=3, score=2, rank=2)),
        2: (Candidate(movie_id=3, score=3, rank=1), Candidate(movie_id=4, score=2, rank=2)),
    }

    metrics = compute_metrics(cohort, pools, candidate_catalog={3, 4, 10, 11, 12})

    assert metrics.catalog_coverage_at_10 == 2 / 5
    assert metrics.to_dict()["CatalogCoverage@10"] == 2 / 5


def test_conditional_recall_is_explicitly_zero_when_no_query_is_covered() -> None:
    cohort = EvaluationCohort(
        queries=(EvaluationQuery(subject_id=1, history=(), gold_movie_id=99),)
    )

    metrics = compute_metrics(cohort, {1: ()})

    assert metrics.coverage_at_200 == 0.0
    assert metrics.conditional_recall_at_10 == 0.0
    assert metrics.end_to_end_recall_at_10 == 0.0
    assert metrics.ndcg_at_10 == 0.0


def test_metrics_remove_observed_candidates_before_assigning_final_ranks() -> None:
    cohort = EvaluationCohort(
        queries=(EvaluationQuery(subject_id=1, history=(10,), gold_movie_id=3),)
    )

    metrics = compute_metrics(
        cohort,
        {1: (Candidate(movie_id=10, score=9, rank=1), Candidate(movie_id=3, score=1, rank=2))},
    )

    assert metrics.coverage_at_200 == 1.0
    assert metrics.end_to_end_recall_at_10 == 1.0
    assert metrics.ndcg_at_10 == 1.0


def test_rrf_and_oracle_union_use_one_based_ranks_and_deterministic_ties() -> None:
    pools = {
        "popularity": (
            Candidate(movie_id=1, score=10, rank=1),
            Candidate(movie_id=2, score=9, rank=2),
        ),
        "itemknn": (
            Candidate(movie_id=2, score=0.9, rank=1),
            Candidate(movie_id=3, score=0.8, rank=2),
        ),
    }

    fused = rrf_fuse(pools)
    union = oracle_union(pools)

    assert [candidate.movie_id for candidate in fused] == [2, 1, 3]
    assert [candidate.movie_id for candidate in union] == [1, 2, 3]
    cohort = EvaluationCohort(
        queries=(
            EvaluationQuery(subject_id=1, history=(), gold_movie_id=3),
            EvaluationQuery(subject_id=2, history=(), gold_movie_id=8),
        )
    )
    assert oracle_union_coverage(cohort, {1: pools, 2: pools}) == 0.5

    tied = rrf_fuse(
        {
            "popularity": (Candidate(movie_id=7, score=1, rank=1),),
            "itemknn": (Candidate(movie_id=8, score=1, rank=1),),
        }
    )
    assert [candidate.movie_id for candidate in tied] == [7, 8]


def test_query_mode_report_uses_validation_best_single_and_realizes_oracle_headroom() -> None:
    class Probe:
        def __init__(self, name: str, pools: dict[int, tuple[Candidate, ...]]) -> None:
            self.name = name
            self._pools = pools

        def candidate_pool(self, _history: tuple[int, ...], limit: int = 200):
            return self._pools[1][:limit]

        def candidate_pool_for_subject(
            self, subject_id: int, _history: tuple[int, ...], limit: int = 200
        ):
            return self._pools[subject_id][:limit]

    cohort = EvaluationCohort(
        queries=(
            EvaluationQuery(subject_id=1, history=(), gold_movie_id=1),
            EvaluationQuery(subject_id=2, history=(), gold_movie_id=4),
        )
    )
    popularity = Probe(
        "popularity",
        {
            1: (Candidate(movie_id=1, score=1, rank=1),),
            2: (Candidate(movie_id=2, score=1, rank=1),),
        },
    )
    itemknn = Probe(
        "itemknn",
        {
            1: (Candidate(movie_id=3, score=1, rank=1),),
            2: (Candidate(movie_id=4, score=1, rank=1),),
        },
    )

    report = evaluate_retrievers(
        {"popularity": popularity, "itemknn": itemknn},
        cohort,
        selected_best_single_name="popularity",
        best_single_source="inner_validation",
        lhf_pools={
            1: (Candidate(movie_id=1, score=0.9, rank=1),),
            2: (Candidate(movie_id=4, score=0.9, rank=1),),
        },
    )

    assert report.best_single_name == "popularity"
    assert report.best_single_source == "inner_validation"
    assert report.best_single.metrics.coverage_at_200 == 0.5
    assert report.rrf.metrics.coverage_at_200 == 1.0
    assert report.lhf is not None
    assert report.lhf.metrics.coverage_at_200 == 1.0
    assert report.oracle_union_coverage == 1.0
    assert report.oracle_headroom_denominator == 0.5
    assert report.oracle_headroom_realized == 1.0


def test_oracle_headroom_explicitly_marks_zero_denominator_undefined() -> None:
    class Probe:
        name = "popularity"

        def candidate_pool(self, _history: tuple[int, ...], limit: int = 200):
            return (Candidate(movie_id=1, score=1, rank=1),)[:limit]

    cohort = EvaluationCohort(queries=(EvaluationQuery(subject_id=1, history=(), gold_movie_id=1),))
    report = evaluate_retrievers(
        {"popularity": Probe()},
        cohort,
        selected_best_single_name="popularity",
        lhf_pools={1: (Candidate(movie_id=1, score=1, rank=1),)},
    )

    assert report.oracle_headroom_denominator == 0.0
    assert report.oracle_headroom_realized is None
    assert report.to_dict()["oracle_headroom"]["undefined_reason"] == ("zero headroom denominator")
