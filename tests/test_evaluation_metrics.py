import math

from deep_recsys_lifecycle.evaluation import (
    EvaluationCohort,
    EvaluationQuery,
    compute_metrics,
    oracle_union,
    oracle_union_coverage,
    rrf_fuse,
)
from deep_recsys_lifecycle.models import Candidate


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
