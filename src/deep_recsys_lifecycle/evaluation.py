from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from .event_store import DataSnapshot
from .models import Candidate, PositiveInteraction, RatingEvent
from .positive import derive_positive_interactions, history_movie_ids
from .retriever import MAX_CANDIDATE_POOL, CandidateRetriever, validate_candidate_pool_limit


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    """One snapshot-history query and its first Future Window Gold Candidate."""

    subject_id: int
    history: tuple[int, ...]
    gold_movie_id: int
    gold_event_id: str | None = None

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
    pools: Mapping[int, tuple[Candidate, ...]]
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

    def to_dict(self) -> dict[str, object]:
        return {
            "cohort_size": self.cohort.size,
            "queries": [
                {
                    "subject_id": query.subject_id,
                    "history": list(query.history),
                    "gold_movie_id": query.gold_movie_id,
                    "gold_event_id": query.gold_event_id,
                }
                for query in self.cohort.queries
            ],
            "retrievers": {
                name: evaluation.to_dict() for name, evaluation in self.retrievers.items()
            },
            "rrf": self.rrf.to_dict(),
            "oracle_union": {
                "coverage_at_200": self.oracle_union_coverage,
                "diagnostic_only": True,
            },
        }


def build_evaluation_cohort(
    snapshot: DataSnapshot | Iterable[RatingEvent],
    future_window: Iterable[RatingEvent],
    *,
    threshold: float = 4.0,
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

    queries: list[EvaluationQuery] = []
    for subject_id in sorted(set(histories) & set(first_future_positive)):
        history = history_movie_ids(histories[subject_id])
        gold = first_future_positive[subject_id]
        queries.append(
            EvaluationQuery(
                subject_id=subject_id,
                history=history,
                gold_movie_id=gold.movie_id,
                gold_event_id=gold.event_id,
            )
        )
    return EvaluationCohort(queries=tuple(queries))


def compute_metrics(
    cohort: EvaluationCohort,
    pools: Mapping[int, Sequence[Candidate]],
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
        pool = _unobserved_pool(pools.get(query.subject_id, ()), query.history, pool_limit)
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
    pools: Mapping[int, Mapping[str, Sequence[Candidate]] | Sequence[Candidate]],
) -> float:
    covered = 0
    for query in cohort.queries:
        query_pools = pools.get(query.subject_id, {})
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
) -> EvaluationReport:
    """Evaluate every retriever, RRF, and Oracle Union over the same cohort."""

    validate_candidate_pool_limit(pool_limit)
    retriever_pools: dict[str, dict[int, tuple[Candidate, ...]]] = {}
    for name, retriever in retrievers.items():
        retriever_pools[name] = {
            query.subject_id: _unobserved_pool(
                retriever.candidate_pool(query.history, limit=pool_limit),
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
    rrf_pools: dict[int, tuple[Candidate, ...]] = {}
    union_inputs: dict[int, dict[str, tuple[Candidate, ...]]] = {}
    for query in cohort.queries:
        query_pools = {name: pools[query.subject_id] for name, pools in retriever_pools.items()}
        union_inputs[query.subject_id] = query_pools
        rrf_pools[query.subject_id] = rrf_fuse(query_pools, top_n=pool_limit)
    rrf_evaluation = RetrieverEvaluation(
        name="rrf",
        pools=rrf_pools,
        metrics=compute_metrics(
            cohort, rrf_pools, pool_limit=pool_limit, ranking_cutoff=ranking_cutoff
        ),
    )
    return EvaluationReport(
        cohort=cohort,
        retrievers=evaluations,
        rrf=rrf_evaluation,
        oracle_union_coverage=oracle_union_coverage(cohort, union_inputs),
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
