from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .event_store import DataSnapshot
from .models import Candidate, PositiveInteraction, RatingEvent
from .positive import derive_positive_interactions, history_movie_ids
from .retriever import (
    MAX_CANDIDATE_POOL,
    CandidateRetriever,
    candidate_pool_for_query,
    validate_candidate_pool_limit,
)

EvaluationQueryMode = Literal["known_user", "history_only"]
HISTORY_LENGTH_SEGMENTS: tuple[str, ...] = ("1-4", "5-19", "20+")


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    """One snapshot-history query and its first Future Window Gold Candidate."""

    subject_id: int | None
    history: tuple[int, ...]
    gold_movie_id: int
    gold_event_id: str | None = None
    query_mode: EvaluationQueryMode = "known_user"
    query_id: str | None = None

    def __post_init__(self) -> None:
        if self.query_mode == "known_user" and self.subject_id is None:
            raise ValueError("Known-User evaluation queries require a Subject identity")
        if self.query_mode == "history_only":
            if self.subject_id is not None:
                raise ValueError(
                    "History-Only evaluation queries must not carry Subject identity; "
                    "Known-User queries are the identity-bearing mode"
                )
            if not self.query_id:
                raise ValueError("History-Only evaluation queries require a query_id")

    @property
    def mode(self) -> EvaluationQueryMode:
        return self.query_mode

    @property
    def pool_key(self) -> int | str:
        if self.query_mode == "known_user":
            if self.subject_id is None:  # pragma: no cover - guarded by __post_init__
                raise ValueError("Known-User query has no Subject identity")
            return self.subject_id
        if self.query_id is None:  # pragma: no cover - guarded by __post_init__
            raise ValueError("History-Only query has no query identity")
        return self.query_id

    @property
    def gold_candidate(self) -> int:
        return self.gold_movie_id


@dataclass(frozen=True, slots=True)
class EvaluationCohort:
    queries: tuple[EvaluationQuery, ...]

    @property
    def size(self) -> int:
        return len(self.queries)

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[EvaluationQuery]:
        return iter(self.queries)


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    query_count: int
    covered_query_count: int
    ranking_success_count: int
    coverage_at_200: float
    conditional_recall_at_10: float
    end_to_end_recall_at_10: float
    ndcg_at_10: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "query_count": self.query_count,
            "covered_query_count": self.covered_query_count,
            "ranking_success_count": self.ranking_success_count,
            "Coverage@200": self.coverage_at_200,
            "ConditionalRecall@10": self.conditional_recall_at_10,
            "EndToEndRecall@10": self.end_to_end_recall_at_10,
            "NDCG@10": self.ndcg_at_10,
        }


@dataclass(frozen=True, slots=True)
class RetrieverEvaluation:
    name: str
    pools: Mapping[int | str, tuple[Candidate, ...]]
    metrics: EvaluationMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "metrics": self.metrics.to_dict(),
            "pools": {
                str(subject_id): [candidate.to_dict() for candidate in candidates]
                for subject_id, candidates in self.pools.items()
            },
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    cohort: EvaluationCohort
    retrievers: Mapping[str, RetrieverEvaluation]
    rrf: RetrieverEvaluation
    oracle_union_coverage: float
    query_mode: EvaluationQueryMode = "known_user"
    best_single: RetrieverEvaluation | None = None
    best_single_name: str | None = None
    best_single_source: str = "cohort"
    lhf: RetrieverEvaluation | None = None
    oracle_union: RetrieverEvaluation | None = None
    oracle_headroom_denominator: float = 0.0
    oracle_headroom_realized: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "cohort_size": self.cohort.size,
            "query_mode": self.query_mode,
            "queries": [
                {
                    "subject_id": query.subject_id,
                    "query_id": query.query_id,
                    "history": list(query.history),
                    "gold_movie_id": query.gold_movie_id,
                    "gold_event_id": query.gold_event_id,
                    "query_mode": query.mode,
                }
                for query in self.cohort.queries
            ],
            "retrievers": {
                name: evaluation.to_dict() for name, evaluation in self.retrievers.items()
            },
            "rrf": self.rrf.to_dict(),
            "best_single": self.best_single.to_dict() if self.best_single is not None else None,
            "best_single_name": self.best_single_name,
            "best_single_source": self.best_single_source,
            "lhf": self.lhf.to_dict() if self.lhf is not None else None,
            "oracle_union": {
                "coverage_at_200": self.oracle_union_coverage,
                "diagnostic_only": True,
                "metrics": self.oracle_union.metrics.to_dict()
                if self.oracle_union is not None
                else None,
            },
            "oracle_headroom": {
                "denominator": self.oracle_headroom_denominator,
                "realized": self.oracle_headroom_realized,
                "undefined_reason": (
                    "zero headroom denominator" if self.oracle_headroom_realized is None else None
                ),
            },
        }


def build_evaluation_cohort(
    snapshot: DataSnapshot | Iterable[RatingEvent],
    future_window: Iterable[RatingEvent],
    *,
    threshold: float = 4.0,
    query_mode: EvaluationQueryMode = "known_user",
) -> EvaluationCohort:
    """Build one deterministic Subject cohort without adding future events to history."""

    snapshot_events = snapshot.events if isinstance(snapshot, DataSnapshot) else tuple(snapshot)
    ordered_snapshot_events = tuple(
        sorted(snapshot_events, key=lambda event: (event.event_time, event.event_id))
    )
    ordered_future_events = tuple(
        sorted(future_window, key=lambda event: (event.event_time, event.event_id))
    )
    snapshot_interactions = derive_positive_interactions(ordered_snapshot_events, threshold)
    future_interactions = derive_positive_interactions(ordered_future_events, threshold)

    histories: defaultdict[int, list[PositiveInteraction]] = defaultdict(list)
    for interaction in snapshot_interactions:
        histories[interaction.subject_id].append(interaction)
    first_future_positive: dict[int, PositiveInteraction] = {}
    for interaction in future_interactions:
        first_future_positive.setdefault(interaction.subject_id, interaction)

    if query_mode not in {"known_user", "history_only"}:
        raise ValueError("evaluation query mode must be Known-User or History-Only")
    queries: list[EvaluationQuery] = []
    for subject_id in sorted(set(histories) & set(first_future_positive)):
        history = history_movie_ids(histories[subject_id])
        gold = first_future_positive[subject_id]
        queries.append(
            EvaluationQuery(
                subject_id=subject_id if query_mode == "known_user" else None,
                history=history,
                gold_movie_id=gold.movie_id,
                gold_event_id=gold.event_id,
                query_mode=query_mode,
                query_id=(f"history-{len(queries)}" if query_mode == "history_only" else None),
            )
        )
    return EvaluationCohort(queries=tuple(queries))


def compute_metrics(
    cohort: EvaluationCohort,
    pools: Mapping[int | str, Sequence[Candidate]],
    *,
    pool_limit: int = MAX_CANDIDATE_POOL,
    ranking_cutoff: int = 10,
) -> EvaluationMetrics:
    """Compute retrieval coverage and final ranking quality from ordered pools."""

    validate_candidate_pool_limit(pool_limit)
    if ranking_cutoff < 1:
        raise ValueError("ranking_cutoff must be positive")

    covered = 0
    ranking_success = 0
    discounted_gain = 0.0
    for query in cohort.queries:
        pool = _unobserved_pool(pools.get(query.pool_key, ()), query.history, pool_limit)
        ranks: dict[int, int] = {}
        for rank, candidate in enumerate(pool, start=1):
            ranks.setdefault(candidate.movie_id, rank)
        gold_rank = ranks.get(query.gold_movie_id)
        if gold_rank is not None:
            covered += 1
            if gold_rank <= ranking_cutoff:
                ranking_success += 1
                discounted_gain += 1.0 / math.log2(gold_rank + 1)

    query_count = cohort.size
    coverage = covered / query_count if query_count else 0.0
    conditional_recall = ranking_success / covered if covered else 0.0
    end_to_end_recall = ranking_success / query_count if query_count else 0.0
    ndcg = discounted_gain / query_count if query_count else 0.0
    return EvaluationMetrics(
        query_count=query_count,
        covered_query_count=covered,
        ranking_success_count=ranking_success,
        coverage_at_200=coverage,
        conditional_recall_at_10=conditional_recall,
        end_to_end_recall_at_10=end_to_end_recall,
        ndcg_at_10=ndcg,
    )


def rrf_fuse(
    pools: Mapping[str, Sequence[Candidate]],
    *,
    top_n: int = MAX_CANDIDATE_POOL,
    rrf_k: int = 60,
) -> tuple[Candidate, ...]:
    """Fuse bounded retriever pools with deterministic Reciprocal Rank Fusion."""

    validate_candidate_pool_limit(top_n)
    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")
    scores: defaultdict[int, float] = defaultdict(float)
    for candidates in pools.values():
        for rank, candidate in enumerate(tuple(candidates)[:MAX_CANDIDATE_POOL], start=1):
            scores[candidate.movie_id] += 1.0 / (rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    return tuple(
        Candidate(movie_id=movie_id, score=score, rank=rank)
        for rank, (movie_id, score) in enumerate(ordered, start=1)
    )


def oracle_union(
    pools: Mapping[str, Sequence[Candidate]],
) -> tuple[Candidate, ...]:
    """Return the unbudgeted union, used only as a diagnostic coverage ceiling."""

    first_rank: dict[int, int] = {}
    for candidates in pools.values():
        for rank, candidate in enumerate(tuple(candidates)[:MAX_CANDIDATE_POOL], start=1):
            first_rank[candidate.movie_id] = min(first_rank.get(candidate.movie_id, rank), rank)
    ordered = sorted(first_rank.items(), key=lambda item: (item[1], item[0]))
    return tuple(
        Candidate(movie_id=movie_id, score=0.0, rank=rank)
        for rank, (movie_id, _first_rank) in enumerate(ordered, start=1)
    )


def oracle_union_coverage(
    cohort: EvaluationCohort,
    pools: Mapping[int | str, Mapping[str, Sequence[Candidate]] | Sequence[Candidate]],
) -> float:
    covered = 0
    for query in cohort.queries:
        query_pools = pools.get(query.pool_key, {})
        if isinstance(query_pools, Mapping):
            union_movie_ids = {
                candidate.movie_id
                for candidates in query_pools.values()
                for candidate in _unobserved_pool(candidates, query.history, MAX_CANDIDATE_POOL)
            }
        else:
            union_movie_ids = {
                candidate.movie_id
                for candidate in _unobserved_pool(query_pools, query.history, MAX_CANDIDATE_POOL)
            }
        if query.gold_movie_id in union_movie_ids:
            covered += 1
    return covered / cohort.size if cohort.size else 0.0


def evaluate_retrievers(
    retrievers: Mapping[str, CandidateRetriever],
    cohort: EvaluationCohort,
    *,
    pool_limit: int = MAX_CANDIDATE_POOL,
    ranking_cutoff: int = 10,
    selected_best_single_name: str | None = None,
    best_single_source: str = "cohort",
    lhf_pools: Mapping[int | str, Sequence[Candidate]] | None = None,
) -> EvaluationReport:
    """Evaluate one query-mode retriever bank and its fusion diagnostics."""

    validate_candidate_pool_limit(pool_limit)
    query_mode: EvaluationQueryMode = cohort.queries[0].mode if cohort.queries else "known_user"
    if any(query.mode != query_mode for query in cohort.queries):
        raise ValueError("an evaluation cohort cannot mix query modes")
    if query_mode == "history_only" and "lightgcn" in retrievers:
        raise ValueError("History-Only evaluation cannot use LightGCN")
    retriever_pools: dict[str, dict[int | str, tuple[Candidate, ...]]] = {}
    for name, retriever in retrievers.items():
        retriever_pools[name] = {
            query.pool_key: _unobserved_pool(
                candidate_pool_for_query(
                    retriever,
                    query.subject_id,
                    query.history,
                    limit=pool_limit,
                ),
                query.history,
                pool_limit,
            )
            for query in cohort.queries
        }

    evaluations = {
        name: RetrieverEvaluation(
            name=name,
            pools=pools,
            metrics=compute_metrics(
                cohort, pools, pool_limit=pool_limit, ranking_cutoff=ranking_cutoff
            ),
        )
        for name, pools in retriever_pools.items()
    }
    rrf_pools: dict[int | str, tuple[Candidate, ...]] = {}
    oracle_pools: dict[int | str, tuple[Candidate, ...]] = {}
    for query in cohort.queries:
        query_pools = {name: pools[query.pool_key] for name, pools in retriever_pools.items()}
        rrf_pools[query.pool_key] = rrf_fuse(query_pools, top_n=pool_limit)
        oracle_pools[query.pool_key] = oracle_union(query_pools)
    rrf_evaluation = RetrieverEvaluation(
        name="rrf",
        pools=rrf_pools,
        metrics=compute_metrics(
            cohort, rrf_pools, pool_limit=pool_limit, ranking_cutoff=ranking_cutoff
        ),
    )
    oracle_evaluation = RetrieverEvaluation(
        name="oracle_union",
        pools=oracle_pools,
        metrics=_compute_unbounded_metrics(cohort, oracle_pools, ranking_cutoff),
    )
    if selected_best_single_name is None:
        selected_best_single_name = _select_best_single_name(evaluations)
        resolved_source = best_single_source
    else:
        if selected_best_single_name not in evaluations:
            raise ValueError("selected best single retriever is not in the evaluation bank")
        resolved_source = best_single_source
    best_single = evaluations[selected_best_single_name]
    lhf_evaluation = (
        RetrieverEvaluation(
            name="lhf",
            pools={key: tuple(values) for key, values in lhf_pools.items()},
            metrics=compute_metrics(
                cohort, lhf_pools, pool_limit=pool_limit, ranking_cutoff=ranking_cutoff
            ),
        )
        if lhf_pools is not None
        else None
    )
    headroom_denominator = max(
        oracle_evaluation.metrics.coverage_at_200 - best_single.metrics.coverage_at_200,
        0.0,
    )
    headroom_realized = (
        None
        if lhf_evaluation is None or headroom_denominator == 0.0
        else (lhf_evaluation.metrics.coverage_at_200 - best_single.metrics.coverage_at_200)
        / headroom_denominator
    )
    return EvaluationReport(
        cohort=cohort,
        retrievers=evaluations,
        rrf=rrf_evaluation,
        oracle_union_coverage=oracle_evaluation.metrics.coverage_at_200,
        query_mode=query_mode,
        best_single=best_single,
        best_single_name=selected_best_single_name,
        best_single_source=resolved_source,
        lhf=lhf_evaluation,
        oracle_union=oracle_evaluation,
        oracle_headroom_denominator=headroom_denominator,
        oracle_headroom_realized=headroom_realized,
    )


def _select_best_single_name(evaluations: Mapping[str, RetrieverEvaluation]) -> str:
    if not evaluations:
        raise ValueError("cannot select a best single retriever from an empty bank")
    return max(
        evaluations,
        key=lambda name: (
            evaluations[name].metrics.coverage_at_200,
            evaluations[name].metrics.end_to_end_recall_at_10,
            evaluations[name].metrics.ndcg_at_10,
            # Mapping insertion order is the declared bank order; earlier wins exact ties.
            -tuple(evaluations).index(name),
        ),
    )


def _compute_unbounded_metrics(
    cohort: EvaluationCohort,
    pools: Mapping[int | str, Sequence[Candidate]],
    ranking_cutoff: int,
) -> EvaluationMetrics:
    covered = 0
    ranking_success = 0
    discounted_gain = 0.0
    for query in cohort.queries:
        query_pool = pools.get(query.pool_key, ())
        pool = _unobserved_pool(
            query_pool,
            query.history,
            limit=max(len(query_pool), 1),
        )
        gold_rank = next(
            (
                rank
                for rank, candidate in enumerate(pool, start=1)
                if candidate.movie_id == query.gold_movie_id
            ),
            None,
        )
        if gold_rank is not None:
            covered += 1
            if gold_rank <= ranking_cutoff:
                ranking_success += 1
                discounted_gain += 1.0 / math.log2(gold_rank + 1)
    query_count = cohort.size
    return EvaluationMetrics(
        query_count=query_count,
        covered_query_count=covered,
        ranking_success_count=ranking_success,
        coverage_at_200=covered / query_count if query_count else 0.0,
        conditional_recall_at_10=ranking_success / covered if covered else 0.0,
        end_to_end_recall_at_10=ranking_success / query_count if query_count else 0.0,
        ndcg_at_10=discounted_gain / query_count if query_count else 0.0,
    )


def _unobserved_pool(
    candidates: Sequence[Candidate], history: Sequence[int], limit: int
) -> tuple[Candidate, ...]:
    observed = set(history)
    seen_movie_ids: set[int] = set()
    unobserved: list[Candidate] = []
    for candidate in candidates:
        if candidate.movie_id in observed or candidate.movie_id in seen_movie_ids:
            continue
        seen_movie_ids.add(candidate.movie_id)
        unobserved.append(
            Candidate(movie_id=candidate.movie_id, score=candidate.score, rank=len(unobserved) + 1)
        )
        if len(unobserved) == limit:
            break
    return tuple(unobserved)


def history_length_segment(history_length: int) -> str:
    """Return the report's stable cold-user history-length bucket."""

    if history_length < 0:
        raise ValueError("history_length cannot be negative")
    if history_length <= 4:
        return "1-4"
    if history_length <= 19:
        return "5-19"
    return "20+"


def history_segment_metrics(report: EvaluationReport) -> dict[str, dict[str, object]]:
    """Return aggregate metrics for the three declared history-length segments.

    The output contains only cohort denominators and metric aggregates. Query identities,
    histories, pools, and Gold Candidates remain outside the report payload.
    """

    evaluations: list[tuple[str, RetrieverEvaluation, bool]] = [
        (name, evaluation, False) for name, evaluation in report.retrievers.items()
    ]
    if report.best_single is not None:
        evaluations.append(("best_single", report.best_single, False))
    evaluations.append(("rrf", report.rrf, False))
    if report.lhf is not None:
        evaluations.append(("lhf", report.lhf, False))
    if report.oracle_union is not None:
        evaluations.append(("oracle_union", report.oracle_union, True))
    result: dict[str, dict[str, object]] = {}
    for segment in HISTORY_LENGTH_SEGMENTS:
        queries = tuple(
            query
            for query in report.cohort
            if history_length_segment(len(query.history)) == segment
        )
        segment_cohort = EvaluationCohort(queries=queries)
        metrics: dict[str, dict[str, int | float]] = {}
        for name, evaluation, unbounded in evaluations:
            pools = {query.pool_key: evaluation.pools.get(query.pool_key, ()) for query in queries}
            segment_metrics = (
                _compute_unbounded_metrics(segment_cohort, pools, ranking_cutoff=10)
                if unbounded
                else compute_metrics(segment_cohort, pools, pool_limit=MAX_CANDIDATE_POOL)
            )
            metrics[name] = segment_metrics.to_dict()
        result[segment] = {
            "cohort_size": segment_cohort.size,
            "denominator": segment_cohort.size,
            "metrics": metrics,
        }
    return result
