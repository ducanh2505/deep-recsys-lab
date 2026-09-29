from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from .event_store import DataSnapshot
from .fusion import (
    FUSION_BANKS,
    FusionFeatureBuilder,
    FusionTrainingRow,
    LearnedHybridFusion,
    build_candidate_union,
    build_lhf_training_rows,
    expected_feature_names,
    rank_lhf_union,
    train_lhf_classifier,
)
from .itemknn import fit_itemknn
from .lightgcn import LightGCNConfig, fit_lightgcn
from .models import Candidate, PositiveInteraction, RatingEvent
from .multivae import MultVAEConfig, fit_multivae
from .paper_benchmark import (
    K_CORE_MIN_DEGREE,
    METRIC_CUTOFFS,
    POSITIVE_RATING_THRESHOLD,
    TEST_FRACTION,
    VALIDATION_FRACTION_OF_TRAIN,
    KnownUserSubjectSplit,
    _filter_to_k_core,
    _membership_fingerprint,
    _package_source_fingerprint,
    _split_rank,
)
from .paper_diagnostics import PaperGoldDiagnostics
from .paper_fit_cache import FitCache
from .paper_measurements import peak_rss_bytes, run_identity, summarize_query_latency
from .paper_pool_adapter import PartitionPoolStream, partition_pools, retriever_state_sha256
from .paper_pool_cache import PoolCache, PoolQuery
from .popularity import fit_popularity
from .retriever import CandidateRetriever, candidate_pool_for_query, validate_candidate_pool_limit

if TYPE_CHECKING:
    from .paper_screening import FrozenTestAccess

_BANK = FUSION_BANKS["known_user"]
_INNER_HOLDOUT_FRACTION = 0.10


@dataclass(frozen=True, slots=True)
class KnownUserHybridConfig:
    seed: int = 42
    inner_seed: int | None = None
    inner_fold_count: int = 1
    pool_limit: int = 200
    max_negative_rows_per_query: int | None = 20
    multivae: MultVAEConfig = field(default_factory=lambda: MultVAEConfig(epochs=1, batch_size=256))
    lightgcn: LightGCNConfig = field(
        default_factory=lambda: LightGCNConfig(epochs=1, batch_size=65_536)
    )
    capture_training_rows: bool = False
    capture_query_evidence: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("split and fit seed must be an integer")
        if self.inner_seed is not None and (
            not isinstance(self.inner_seed, int) or isinstance(self.inner_seed, bool)
        ):
            raise ValueError("inner fold seed must be an integer")
        if isinstance(self.inner_fold_count, bool) or not 1 <= self.inner_fold_count <= 5:
            raise ValueError("inner_fold_count must be between 1 and 5")
        validate_candidate_pool_limit(self.pool_limit)
        if self.max_negative_rows_per_query is not None and (
            isinstance(self.max_negative_rows_per_query, bool)
            or self.max_negative_rows_per_query < 1
        ):
            raise ValueError("max_negative_rows_per_query must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "inner_seed": self.seed if self.inner_seed is None else self.inner_seed,
            "inner_fold_count": self.inner_fold_count,
            "pool_limit_per_source": self.pool_limit,
            "final_top_n": 100,
            "max_negative_rows_per_query": self.max_negative_rows_per_query,
            "multivae": self.multivae.to_dict(),
            "multivae_device_preference": "cpu",
            "lightgcn": self.lightgcn.to_dict(),
            "lightgcn_device_preference": "cpu",
            "capture_training_rows": self.capture_training_rows,
        }


@dataclass(frozen=True, slots=True)
class _GoldQuery:
    subject_id: int
    history: tuple[int, ...]
    gold_movie_ids: tuple[int, ...]
    raw_gold_movie_ids: tuple[int, ...] = ()

    @property
    def pool_key(self) -> int:
        return self.subject_id


@dataclass(frozen=True, slots=True)
class KnownUserHybridResult:
    """Validation evidence with a separately callable frozen-test seam."""

    report: Mapping[str, object]
    split_by_subject: Mapping[int, KnownUserSubjectSplit]
    training_catalog: frozenset[int]
    inner_training_rows: tuple[FusionTrainingRow, ...]
    fusion: LearnedHybridFusion
    validation_rankings: Mapping[int, tuple[int, ...]]
    validation_subject_metrics: Mapping[int, Mapping[str, float]]
    validation_query_evidence: Mapping[int, Mapping[str, object]]
    _test_movie_ids_by_subject: Mapping[int, tuple[int, ...]] = field(repr=False)
    _retrievers: Mapping[str, CandidateRetriever] = field(repr=False)
    _feature_builder: FusionFeatureBuilder = field(repr=False)
    _config: KnownUserHybridConfig = field(repr=False)

    def to_dict(self) -> dict[str, object]:
        return dict(self.report)


class _RankingMetrics:
    def __init__(self, catalog: frozenset[int]) -> None:
        self.catalog = catalog
        self.query_count = 0
        self.covered = 0
        self.recommended: set[int] = set()
        self.recall = {cutoff: 0.0 for cutoff in METRIC_CUTOFFS}
        self.ndcg = {cutoff: 0.0 for cutoff in METRIC_CUTOFFS}

    def add(self, gold: Sequence[int], ranking: Sequence[int]) -> None:
        if not gold:
            return
        self.query_count += 1
        ranked = tuple(ranking[:100])
        positions = {movie_id: rank for rank, movie_id in enumerate(ranked, 1)}
        self.recommended.update(ranked)
        if any(movie_id in positions for movie_id in gold):
            self.covered += 1
        for cutoff in METRIC_CUTOFFS:
            hits = [
                positions[movie_id] for movie_id in gold if positions.get(movie_id, 101) <= cutoff
            ]
            self.recall[cutoff] += len(hits) / len(gold)
            gain = sum(1.0 / math.log2(rank + 1) for rank in hits)
            ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(cutoff, len(gold)) + 1))
            self.ndcg[cutoff] += gain / ideal if ideal else 0.0

    def result(self) -> dict[str, int | float | None]:
        divisor = self.query_count or 1
        return {
            "query_count": self.query_count,
            **{f"Recall@{k}": self.recall[k] / divisor for k in METRIC_CUTOFFS},
            **{f"NDCG@{k}": self.ndcg[k] / divisor for k in METRIC_CUTOFFS},
            "QueryRetrievalCoverage@100": self.covered / divisor,
            "CatalogCoverage@100": (
                len(self.recommended) / len(self.catalog)
                if self.catalog and self.query_count
                else None
            ),
        }


class _UnionMetrics:
    def __init__(self, catalog: frozenset[int]) -> None:
        self.catalog = catalog
        self.query_count = 0
        self.covered = 0
        self.recall = 0.0
        self.movies: set[int] = set()

    def add(self, gold: Sequence[int], union: Sequence[int]) -> None:
        if not gold:
            return
        self.query_count += 1
        union_ids = set(union)
        self.movies.update(union_ids)
        hits = len(union_ids.intersection(gold))
        self.recall += hits / len(gold)
        self.covered += int(hits > 0)

    def result(self) -> dict[str, object]:
        divisor = self.query_count or 1
        return {
            "query_count": self.query_count,
            "OracleUnionRecall": self.recall / divisor,
            "OracleUnionQueryCoverage": self.covered / divisor,
            "OracleUnionCatalogCoverage": (
                len(self.movies) / len(self.catalog) if self.catalog and self.query_count else None
            ),
            "pool_budget": "unbounded union of four source pools",
            "final_order": False,
        }


def _sorted_events(events: Iterable[RatingEvent]) -> tuple[RatingEvent, ...]:
    return tuple(
        sorted(
            events,
            key=lambda event: (event.subject_id, event.movie_id, event.event_time, event.event_id),
        )
    )


def _fingerprint_event_ids(events: Iterable[RatingEvent]) -> str:
    digest = hashlib.sha256()
    for event in _sorted_events(events):
        digest.update(event.event_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _split(
    events: Iterable[RatingEvent], seed: int
) -> tuple[dict[int, KnownUserSubjectSplit], tuple[RatingEvent, ...], str]:
    by_pair: dict[tuple[int, int], RatingEvent] = {}
    event_count = 0
    digest_xor = 0
    digest_sum = 0
    for event in events:
        event_count += 1
        value = int.from_bytes(
            hashlib.sha256(
                "\x1f".join(
                    (
                        event.event_id,
                        str(event.subject_id),
                        str(event.movie_id),
                        format(event.rating, ".17g"),
                        str(event.event_time),
                        event.source,
                    )
                ).encode("utf-8")
            ).digest(),
            "big",
        )
        digest_xor ^= value
        digest_sum = (digest_sum + value) % (1 << 256)
        if event.rating < POSITIVE_RATING_THRESHOLD:
            continue
        key = (event.subject_id, event.movie_id)
        previous = by_pair.get(key)
        if previous is None or (event.event_time, event.event_id) < (
            previous.event_time,
            previous.event_id,
        ):
            by_pair[key] = event
    retained = _filter_to_k_core(by_pair, K_CORE_MIN_DEGREE)
    by_subject: dict[int, list[RatingEvent]] = {}
    for event in retained.values():
        by_subject.setdefault(event.subject_id, []).append(event)
    splits: dict[int, KnownUserSubjectSplit] = {}
    fit_events: list[RatingEvent] = []
    for subject_id in sorted(by_subject):
        values = by_subject[subject_id]
        for_test = sorted(
            values,
            key=lambda event: (
                _split_rank(seed, "test", subject_id, event.movie_id),
                event.movie_id,
            ),
        )
        test_count = max(1, math.ceil(len(values) * TEST_FRACTION))
        test_ids = {event.movie_id for event in for_test[:test_count]}
        outer = [event for event in values if event.movie_id not in test_ids]
        for_validation = sorted(
            outer,
            key=lambda event: (
                _split_rank(seed, "validation", subject_id, event.movie_id),
                event.movie_id,
            ),
        )
        validation_count = max(1, math.ceil(len(outer) * VALIDATION_FRACTION_OF_TRAIN))
        validation_ids = {event.movie_id for event in for_validation[:validation_count]}
        fitted = [event for event in outer if event.movie_id not in validation_ids]
        fit_events.extend(fitted)
        splits[subject_id] = KnownUserSubjectSplit(
            train_movie_ids=tuple(sorted(event.movie_id for event in fitted)),
            outer_train_movie_ids=tuple(sorted(event.movie_id for event in outer)),
            validation_movie_ids=tuple(sorted(validation_ids)),
            test_movie_ids=tuple(sorted(test_ids)),
        )
    fingerprint = hashlib.sha256(
        f"{event_count}:{digest_xor:064x}:{digest_sum:064x}".encode("ascii")
    ).hexdigest()
    return splits, _sorted_events(fit_events), fingerprint


def _inner_holdout(
    fit_events: Sequence[RatingEvent], seed: int, fold_index: int = 0
) -> tuple[tuple[RatingEvent, ...], tuple[RatingEvent, ...]]:
    by_subject: dict[int, list[RatingEvent]] = {}
    for event in fit_events:
        by_subject.setdefault(event.subject_id, []).append(event)
    inner_fit: list[RatingEvent] = []
    inner_holdout: list[RatingEvent] = []
    for subject_id, values in sorted(by_subject.items()):
        ordered = sorted(
            values,
            key=lambda event: (
                _split_rank(seed, "fusion-inner", subject_id, event.movie_id),
                event.movie_id,
            ),
        )
        fold_size = max(1, math.ceil(len(ordered) * _INNER_HOLDOUT_FRACTION))
        start = fold_index * fold_size
        holdout_count = min(fold_size, max(0, len(ordered) - 1 - start))
        held = {event.movie_id for event in ordered[start : start + holdout_count]}
        inner_holdout.extend(event for event in values if event.movie_id in held)
        inner_fit.extend(event for event in values if event.movie_id not in held)
    return _sorted_events(inner_fit), _sorted_events(inner_holdout)


def _fit_bank(
    events: Sequence[RatingEvent],
    config: KnownUserHybridConfig,
    *,
    fit_cache: FitCache | None = None,
    code_fingerprint: str = "",
) -> tuple[DataSnapshot, dict[str, CandidateRetriever]]:
    snapshot = DataSnapshot.from_events(tuple(events))
    interactions = tuple(PositiveInteraction.from_event(event) for event in snapshot.events)
    if not interactions:
        raise ValueError("Known-User hybrid requires non-empty training interactions")
    def source_fit(
        name: str,
        source_config: Mapping[str, object],
        fit: Callable[[], CandidateRetriever],
    ) -> CandidateRetriever:
        if fit_cache is None:
            return fit()
        return fit_cache.get_or_fit(
            source=name,
            snapshot=snapshot,
            source_config=source_config,
            seed=config.seed,
            device="cpu",
            code_fingerprint=code_fingerprint,
            fit=fit,
        )

    return snapshot, {
        "popularity": source_fit(
            "popularity", {"ranking": "positive_count"},
            lambda: fit_popularity(snapshot, interactions),
        ),
        "itemknn": source_fit(
            "itemknn", {"similarity": "binary_cosine"},
            lambda: fit_itemknn(snapshot, interactions).to_serving(),
        ),
        "multivae": source_fit(
            "multivae", config.multivae.to_dict(),
            lambda: fit_multivae(
                snapshot, interactions, config=config.multivae,
                seed=config.seed, device_preference="cpu",
            ),
        ),
        "lightgcn": source_fit(
            "lightgcn", config.lightgcn.to_dict(),
            lambda: fit_lightgcn(
                snapshot, interactions, config=config.lightgcn,
                seed=config.seed, device_preference="cpu",
            ),
        ),
    }


def _queries(
    splits: Mapping[int, KnownUserSubjectSplit],
    catalog: frozenset[int],
    partition: str,
    *,
    test_movie_ids_by_subject: Mapping[int, tuple[int, ...]] | None = None,
) -> tuple[_GoldQuery, ...]:
    result: list[_GoldQuery] = []
    for subject_id, split in sorted(splits.items()):
        if partition == "validation":
            held_out = split.validation_movie_ids
        elif partition == "test" and test_movie_ids_by_subject is not None:
            held_out = test_movie_ids_by_subject[subject_id]
        else:
            raise ValueError("test scoring requires the private frozen-test partition")
        gold = tuple(
            movie_id
            for movie_id in held_out
            if movie_id in catalog and movie_id not in split.train_movie_ids
        )
        result.append(_GoldQuery(subject_id, split.train_movie_ids, gold, held_out))
    return tuple(result)


def _pools_for(
    query: _GoldQuery,
    retrievers: Mapping[str, CandidateRetriever],
    limit: int,
) -> dict[str, tuple[Candidate, ...]]:
    return {
        name: candidate_pool_for_query(retrievers[name], query.subject_id, query.history, limit)
        for name in _BANK
    }


def _cached_partition(
    pool_cache: PoolCache | None,
    *,
    partition: str,
    cohort_fingerprint: str,
    snapshot: DataSnapshot,
    retrievers: Mapping[str, CandidateRetriever],
    queries: Sequence[_GoldQuery],
    config: KnownUserHybridConfig,
    code_fingerprint: str,
) -> PartitionPoolStream | None:
    if pool_cache is None:
        return None
    return partition_pools(
        pool_cache,
        mode="known_user",
        partition=partition,
        cohort_fingerprint=cohort_fingerprint,
        fit_fingerprint=snapshot.fingerprint,
        code_fingerprint=code_fingerprint,
        retrievers=retrievers,
        source_configs={
            "popularity": {"ranking": "positive_count"},
            "itemknn": {"similarity": "binary_cosine"},
            "multivae": config.multivae.to_dict(),
            "lightgcn": config.lightgcn.to_dict(),
        },
        model_fingerprints={
            name: retriever_state_sha256(retriever)
            for name, retriever in retrievers.items()
        },
        seed=config.seed,
        device_by_source={name: "cpu" for name in _BANK},
        checkpoint_by_source={
            name: "last" if name in {"multivae", "lightgcn"} else "not_applicable"
            for name in _BANK
        },
        catalog_movie_ids=frozenset(event.movie_id for event in snapshot.events),
        queries=tuple(
            PoolQuery(query.subject_id, query.subject_id, query.history)
            for query in queries
            if query.gold_movie_ids
        ),
        depth=config.pool_limit,
    )


def _evaluate(
    queries: Iterable[_GoldQuery],
    retrievers: Mapping[str, CandidateRetriever],
    fusion: LearnedHybridFusion,
    builder: FusionFeatureBuilder,
    catalog: frozenset[int],
    config: KnownUserHybridConfig,
    pool_stream: PartitionPoolStream | None = None,
) -> tuple[
    dict[str, object],
    dict[int, tuple[int, ...]],
    dict[int, Mapping[str, float]],
    dict[int, Mapping[str, object]],
]:
    sources = {name: _RankingMetrics(catalog) for name in _BANK}
    union_metrics = _UnionMetrics(catalog)
    final_metrics = _RankingMetrics(catalog)
    loss = PaperGoldDiagnostics(
        mode="known_user",
        source_names=_BANK,
        training_catalog=catalog,
        training_popularity=builder.item_popularity,
    )
    final_rankings: dict[int, tuple[int, ...]] = {}
    subject_metrics: dict[int, Mapping[str, float]] = {}
    evidence: dict[int, Mapping[str, object]] = {}
    scoring_samples_ms: list[float] = []
    cached_pools = iter(pool_stream) if pool_stream is not None else None
    for query in queries:
        if not query.gold_movie_ids:
            loss.observe(
                raw_gold_movie_ids=query.raw_gold_movie_ids,
                eligible_gold_movie_ids=(),
                history=query.history,
                pools={name: () for name in _BANK},
                fusion_input_movie_ids=(),
                final_ranking=(),
            )
            continue
        scoring_started = monotonic()
        if cached_pools is None:
            pools = _pools_for(query, retrievers, config.pool_limit)
        else:
            cached_query, pools = next(cached_pools)
            if (cached_query.key, cached_query.history) != (query.subject_id, query.history):
                raise ValueError("cached Known-User Query differs from validation Query")
        for name, pool in pools.items():
            sources[name].add(query.gold_movie_ids, tuple(item.movie_id for item in pool))
        union = build_candidate_union(pools, history=query.history)
        union_ids = tuple(item.movie_id for item in union)
        union_metrics.add(query.gold_movie_ids, union_ids)
        final = rank_lhf_union(fusion, builder, pools, query.history, top_n=100)
        scoring_samples_ms.append((monotonic() - scoring_started) * 1000)
        loss.observe(
            raw_gold_movie_ids=query.raw_gold_movie_ids,
            eligible_gold_movie_ids=query.gold_movie_ids,
            history=query.history,
            pools=pools,
            fusion_input_movie_ids=union_ids,
            final_ranking=final,
        )
        final_ids = tuple(item.movie_id for item in final)
        final_metrics.add(query.gold_movie_ids, final_ids)
        final_rankings[query.subject_id] = final_ids
        per_subject = _RankingMetrics(catalog)
        per_subject.add(query.gold_movie_ids, final_ids)
        subject_metrics[query.subject_id] = {
            key: float(value)
            for key, value in per_subject.result().items()
            if key.startswith(("Recall@", "NDCG@")) and value is not None
        }
        if config.capture_query_evidence:
            evidence[query.subject_id] = {
                "gold_movie_ids": list(query.gold_movie_ids),
                "source_pool_movie_ids": {
                    name: [item.movie_id for item in pool] for name, pool in pools.items()
                },
                "oracle_union_movie_ids": list(union_ids),
                "final_movie_ids": list(final_ids),
                "final_ranks": [item.rank for item in final],
            }
    if cached_pools is not None and next(cached_pools, None) is not None:
        raise ValueError("cached Known-User pool stream has extra Queries")
    final_result = final_metrics.result()
    return (
        {
            "source_pools": {name: accumulator.result() for name, accumulator in sources.items()},
            "oracle_union": union_metrics.result(),
            "final_lhf": final_result,
            "gold_loss": loss.finish(),
            "inference_latency": summarize_query_latency(scoring_samples_ms, warmup_count=5),
        },
        final_rankings,
        subject_metrics,
        evidence,
    )


def run_known_user_hybrid_benchmark(
    events: Iterable[RatingEvent], *, config: KnownUserHybridConfig | None = None,
    fit_cache: FitCache | None = None,
    pool_cache: PoolCache | None = None,
) -> KnownUserHybridResult:
    """Fit four train-only retrievers and LHF; report validation only."""

    resolved = config or KnownUserHybridConfig()
    fit_before = fit_cache.stats if fit_cache is not None else None
    code_fingerprint = (
        _package_source_fingerprint() if fit_cache is not None or pool_cache is not None else ""
    )
    run_started = monotonic()
    dataset_checksum = getattr(events, "dataset_checksum", None)
    splits, fit_events, input_fingerprint = _split(events, resolved.seed)
    split_seconds = monotonic() - run_started
    if not fit_events:
        raise ValueError("Known-User 10-core split has no training interactions")
    outer_catalog = frozenset(event.movie_id for event in fit_events)
    inner_seed = resolved.seed if resolved.inner_seed is None else resolved.inner_seed
    rows: list[FusionTrainingRow] = []
    fold_sources: list[dict[str, object]] = []
    label_event_ids: list[str] = []
    fold_query_count = 0
    fold_fit_count = 0
    fold_holdout_count = 0
    fold_fingerprints: list[str] = []
    pool_cache_log: list[dict[str, object]] = []
    inner_started = monotonic()
    for fold_index in range(resolved.inner_fold_count):
        inner_fit, inner_holdout = _inner_holdout(fit_events, inner_seed, fold_index)
        inner_snapshot, inner_retrievers = _fit_bank(
            inner_fit, resolved, fit_cache=fit_cache, code_fingerprint=code_fingerprint
        )
        inner_catalog = frozenset(event.movie_id for event in inner_fit)
        inner_histories: dict[int, list[int]] = {}
        inner_gold: dict[int, list[int]] = {}
        for event in inner_fit:
            inner_histories.setdefault(event.subject_id, []).append(event.movie_id)
        for event in inner_holdout:
            if event.movie_id in inner_catalog:
                inner_gold.setdefault(event.subject_id, []).append(event.movie_id)
                label_event_ids.append(event.event_id)
        inner_queries = tuple(
            _GoldQuery(subject_id, tuple(sorted(inner_histories[subject_id])), tuple(sorted(gold)))
            for subject_id, gold in sorted(inner_gold.items())
        )
        inner_builder = FusionFeatureBuilder.from_snapshot(
            inner_snapshot,
            tuple(PositiveInteraction.from_event(event) for event in inner_fit),
            retriever_bank=_BANK,
        )
        inner_stream = _cached_partition(
            pool_cache,
            partition=f"inner:{fold_index}",
            cohort_fingerprint=input_fingerprint,
            snapshot=inner_snapshot,
            retrievers=inner_retrievers,
            queries=inner_queries,
            config=resolved,
            code_fingerprint=code_fingerprint,
        )
        scored = (
            ((PoolQuery(query.subject_id, query.subject_id, query.history),
              _pools_for(query, inner_retrievers, resolved.pool_limit)) for query in inner_queries)
            if inner_stream is None else inner_stream
        )
        for query, (pool_query, pools) in zip(inner_queries, scored, strict=True):
            if (pool_query.key, pool_query.history) != (query.subject_id, query.history):
                raise ValueError("cached Known-User inner Query differs")
            rows.extend(
                build_lhf_training_rows(
                    (query,),
                    {query.subject_id: pools},
                    inner_builder,
                    max_negative_rows_per_query=resolved.max_negative_rows_per_query,
                    seed=inner_seed + fold_index,
                )
            )
        if inner_stream is not None:
            pool_cache_log.append({
                "partition": f"inner:{fold_index}",
                "fit_fingerprint": inner_snapshot.fingerprint,
                "sources": {
                    name: {"key": info.key, "cache_hit": info.cache_hit,
                           "stored_depth": info.stored_depth,
                           "model_fingerprint": inner_stream.model_fingerprints[name]}
                    for name, info in inner_stream.cache_info.items()
                },
            })
        fold_query_count += len(inner_queries)
        fold_fit_count += len(inner_fit)
        fold_holdout_count += len(inner_holdout)
        fold_fingerprints.append(inner_snapshot.fingerprint)
        fold_sources.append(
            {
                "validation_kind": "paper_inner_interaction_holdout",
                "fold_index": fold_index,
                "snapshot_fingerprint": inner_snapshot.fingerprint,
                "fit_event_ids_sha256": _fingerprint_event_ids(inner_fit),
                "heldout_event_ids_sha256": _fingerprint_event_ids(inner_holdout),
                "fit_event_count": len(inner_fit),
                "heldout_event_count": len(inner_holdout),
                "eligible_query_count": len(inner_queries),
            }
        )
    inner_fit_and_labels_seconds = monotonic() - inner_started
    combined_inner_fingerprint = (
        fold_fingerprints[0]
        if len(fold_fingerprints) == 1
        else hashlib.sha256("\n".join(fold_fingerprints).encode("ascii")).hexdigest()
    )
    fusion_started = monotonic()
    fusion = train_lhf_classifier(
        query_mode="known_user",
        retriever_bank=_BANK,
        feature_names=expected_feature_names(_BANK),
        rows=rows,
        source_snapshot_fingerprint=combined_inner_fingerprint,
        seed=inner_seed,
        validation_event_ids=label_event_ids,
        validation_sources=fold_sources,
    )
    fusion = replace(
        fusion,
        training_metadata={
            **fusion.training_metadata,
            "training_boundary": "paper_inner_interaction_holdout_within_outer_training",
            "paper_validation_used_for_training": False,
            "paper_test_used_for_training": False,
            "outer_fit_event_ids_sha256": _fingerprint_event_ids(fit_events),
            "inner_fold_count": resolved.inner_fold_count,
            "inner_fit_event_count": fold_fit_count,
            "inner_holdout_event_count": fold_holdout_count,
        },
    )
    fusion_training_seconds = monotonic() - fusion_started
    retained_rows = tuple(rows) if resolved.capture_training_rows else ()
    # The outer fit can be large; keep only the fitted classifier and compact provenance.
    del rows
    del inner_retrievers, inner_snapshot, inner_builder, inner_fit, inner_holdout
    del inner_histories, inner_gold, inner_queries
    outer_started = monotonic()
    outer_snapshot, outer_retrievers = _fit_bank(
        fit_events, resolved, fit_cache=fit_cache, code_fingerprint=code_fingerprint
    )
    outer_builder = FusionFeatureBuilder.from_snapshot(
        outer_snapshot,
        tuple(PositiveInteraction.from_event(event) for event in fit_events),
        retriever_bank=_BANK,
    )
    outer_fit_seconds = monotonic() - outer_started
    validation_queries = _queries(splits, outer_catalog, "validation")
    validation_stream = _cached_partition(
        pool_cache,
        partition="validation",
        cohort_fingerprint=input_fingerprint,
        snapshot=outer_snapshot,
        retrievers=outer_retrievers,
        queries=validation_queries,
        config=resolved,
        code_fingerprint=code_fingerprint,
    )
    validation_started = monotonic()
    evaluated, rankings, subject_metrics, evidence = _evaluate(
        validation_queries, outer_retrievers, fusion, outer_builder, outer_catalog, resolved,
        pool_stream=validation_stream,
    )
    if validation_stream is not None:
        pool_cache_log.append({
            "partition": "validation",
            "fit_fingerprint": outer_snapshot.fingerprint,
            "sources": {
                name: {"key": info.key, "cache_hit": info.cache_hit,
                       "stored_depth": info.stored_depth,
                       "model_fingerprint": validation_stream.model_fingerprints[name]}
                for name, info in validation_stream.cache_info.items()
            },
        })
    validation_scoring_seconds = monotonic() - validation_started
    final_metrics = evaluated["final_lhf"]
    report: dict[str, object] = {
        "schema_version": 1,
        "protocol": "lightgcn-ngcf-known-user",
        "pipeline": "four_retrievers_lhf",
        "evaluated_partition": "validation",
        "test_status": "sealed",
        "source": {
            "name": "MovieLens 20M" if isinstance(dataset_checksum, str) else "Rating Events",
            "dataset_sha256": dataset_checksum if isinstance(dataset_checksum, str) else None,
            "input_event_fingerprint_sha256": input_fingerprint,
        },
        "configuration": resolved.to_dict(),
        "package_source_sha256": _package_source_fingerprint(),
        "screen_cache": {
            "code_fingerprint": (
                code_fingerprint if fit_cache is not None or pool_cache is not None else None
            ),
            "pool_partitions": pool_cache_log,
            "fit": (
                {key: fit_cache.stats[key] - fit_before[key] for key in ("hits", "misses")}
                if fit_cache is not None and fit_before is not None else None
            ),
        },
        "run_identity": run_identity(events),
        "retriever_bank": list(_BANK),
        "training_catalog_movie_ids": sorted(outer_catalog),
        "split_membership_by_subject": {
            str(subject_id): {
                "train_movie_ids": list(split.train_movie_ids),
                "outer_train_movie_ids": list(split.outer_train_movie_ids),
                "validation_movie_ids": list(split.validation_movie_ids),
            }
            for subject_id, split in splits.items()
        },
        "split_membership_sha256": {
            partition: _membership_fingerprint(splits, partition)
            for partition in ("train", "validation", "test")
        },
        "counts": {
            "eligible_subject_count": len(splits),
            "outer_fit_interaction_count": len(fit_events),
            "inner_fit_interaction_count": fold_fit_count,
            "inner_holdout_interaction_count": fold_holdout_count,
            "inner_training_query_count": fold_query_count,
            "validation_query_count": sum(
                bool(query.gold_movie_ids) for query in validation_queries
            ),
            "validation_gold_movie_count": sum(
                len(query.gold_movie_ids) for query in validation_queries
            ),
            "validation_gold_movie_count_missing_from_training_catalog": sum(
                movie_id not in outer_catalog
                for split in splits.values()
                for movie_id in split.validation_movie_ids
            ),
            "reserved_test_interaction_count": sum(
                len(split.test_movie_ids) for split in splits.values()
            ),
        },
        "validation_gold_sets_by_subject": {
            str(query.subject_id): list(query.gold_movie_ids)
            for query in validation_queries
            if query.gold_movie_ids
        },
        "validation_subject_metrics": {
            str(subject_id): dict(values) for subject_id, values in subject_metrics.items()
        },
        "training": {
            "outer_snapshot_fingerprint": outer_snapshot.fingerprint,
            "inner_snapshot_fingerprint": combined_inner_fingerprint,
            "inner_folds": fold_sources,
            "outer_fit_event_ids_sha256": _fingerprint_event_ids(fit_events),
            "fusion": {
                "training_status": fusion.training_status,
                "training_metadata": dict(fusion.training_metadata),
            },
            "retriever_training_metadata": {
                name: dict(getattr(retriever, "training_metadata", {}))
                for name, retriever in outer_retrievers.items()
            },
            "outer_fit_partition": "train_movie_ids",
            "inner_labels_partition": "interaction holdouts within train_movie_ids",
            "paper_validation_or_test_in_fit": False,
        },
        "evaluation": {
            "candidate_universe": "complete outer training catalog after Subject history exclusion",
            "sampled_negatives": False,
            "synthetic_gold_insertion": False,
            "gold_exclusion": "Movies absent from outer training catalog",
            **evaluated,
        },
        "metrics": final_metrics,
        "runtime": {
            "split_seconds": split_seconds,
            "inner_retriever_fit_and_fusion_rows_seconds": inner_fit_and_labels_seconds,
            "fusion_classifier_training_seconds": fusion_training_seconds,
            "outer_retriever_fit_seconds": outer_fit_seconds,
            "validation_scoring_seconds": validation_scoring_seconds,
            "total_wall_seconds": monotonic() - run_started,
            "peak_process_rss_bytes": peak_rss_bytes(),
        },
    }
    return KnownUserHybridResult(
        report=report,
        split_by_subject={
            subject_id: replace(split, test_movie_ids=()) for subject_id, split in splits.items()
        },
        training_catalog=outer_catalog,
        inner_training_rows=retained_rows,
        fusion=fusion,
        validation_rankings=rankings,
        validation_subject_metrics=subject_metrics,
        validation_query_evidence=evidence,
        _test_movie_ids_by_subject={
            subject_id: split.test_movie_ids for subject_id, split in splits.items()
        },
        _retrievers=outer_retrievers,
        _feature_builder=outer_builder,
        _config=resolved,
    )


def evaluate_known_user_frozen_test(
    frozen: KnownUserHybridResult,
    *,
    test_access: FrozenTestAccess | None = None,
) -> dict[str, object]:
    """Open test only for an already selected, immutable hybrid fit."""

    from .paper_screening import _digest, _validation_seal

    if test_access is None:
        raise ValueError("frozen test access receipt is required")
    seal = _validation_seal(frozen.report, "known_user")
    source = frozen.report["source"]
    assert isinstance(source, Mapping)
    test_access.require(
        mode="known_user",
        source_sha256=source["dataset_sha256"],
        configuration_sha256=_digest(frozen._config.to_dict()),
        validation_cohort_sha256=str(seal["validation_cohort_sha256"]),
        test_membership_sha256=str(seal["test_membership_sha256"]),
    )
    queries = _queries(
        frozen.split_by_subject,
        frozen.training_catalog,
        "test",
        test_movie_ids_by_subject=frozen._test_movie_ids_by_subject,
    )
    evaluated, rankings, subject_metrics, evidence = _evaluate(
        queries,
        frozen._retrievers,
        frozen.fusion,
        frozen._feature_builder,
        frozen.training_catalog,
        frozen._config,
    )
    return {
        "protocol": "lightgcn-ngcf-known-user",
        "evaluated_partition": "test",
        "configuration": frozen._config.to_dict(),
        "training_snapshot_fingerprint": frozen._feature_builder.snapshot_fingerprint,
        "evaluation": evaluated,
        "metrics": evaluated["final_lhf"],
        "rankings": rankings,
        "subject_metrics": subject_metrics,
        "query_evidence": evidence,
    }


def write_known_user_hybrid_report(result: KnownUserHybridResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
