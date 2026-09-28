from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

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
from .models import Candidate, PositiveInteraction, RatingEvent
from .multivae import MultVAEConfig, fit_multivae
from .paper_diagnostics import PaperGoldDiagnostics
from .popularity import fit_popularity
from .retriever import CandidateRetriever, candidate_pool_for_query, validate_candidate_pool_limit

POSITIVE_RATING_THRESHOLD = 4.0
MIN_POSITIVE_INTERACTIONS = 5
HELD_OUT_FRACTION = 0.20
METRIC_CUTOFFS = (10, 20, 50, 100)
HISTORY_ONLY_BANK = FUSION_BANKS["history_only"]


@dataclass(frozen=True, slots=True)
class HistoryOnlyBenchmarkConfig:
    seed: int = 42
    validation_subject_count: int = 10_000
    test_subject_count: int = 10_000
    inner_held_out_fraction: float = 0.10
    candidate_pool_limit: int = 200
    fusion_negative_rows_per_query: int | None = 20
    multivae_config: MultVAEConfig = field(
        default_factory=lambda: MultVAEConfig(epochs=1, batch_size=256)
    )
    multivae_device_preference: str = "cpu"
    capture_evidence: bool = False

    def __post_init__(self) -> None:
        if self.validation_subject_count < 1 or self.test_subject_count < 1:
            raise ValueError("validation and test cohorts must each contain at least one Subject")
        if not 0 < self.inner_held_out_fraction < 1:
            raise ValueError("inner_held_out_fraction must be between zero and one")
        validate_candidate_pool_limit(self.candidate_pool_limit)
        if self.fusion_negative_rows_per_query is not None and (
            self.fusion_negative_rows_per_query < 1
        ):
            raise ValueError("fusion_negative_rows_per_query must be positive")
        if self.multivae_device_preference not in {"cpu", "auto", "mps"}:
            raise ValueError("unsupported Mult-VAE device preference")


@dataclass(frozen=True, slots=True)
class HistoryOnlyQuery:
    """The scoring input contains fold-in history, never a Subject identity."""

    history_movie_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HistoryOnlySubjectSplit:
    fold_in_movie_ids: tuple[int, ...]
    held_out_movie_ids: tuple[int, ...]
    query_history_movie_ids: tuple[int, ...]
    gold_movie_ids: tuple[int, ...]
    excluded_cold_fold_in_movie_ids: tuple[int, ...]
    excluded_cold_gold_movie_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HistoryOnlyBenchmarkResult:
    report: Mapping[str, object]
    cohort_subject_ids: Mapping[str, tuple[int, ...]]
    split_by_subject: Mapping[int, HistoryOnlySubjectSplit]
    queries_by_subject: Mapping[int, HistoryOnlyQuery]
    gold_sets: Mapping[int, tuple[int, ...]]
    training_catalog: frozenset[int]
    fusion: LearnedHybridFusion
    fusion_training_rows: tuple[FusionTrainingRow, ...]
    inner_fit_subject_ids: tuple[int, ...]
    inner_pseudo_held_out_subject_ids: tuple[int, ...]
    inner_feature_builder: FusionFeatureBuilder | None = field(repr=False)
    outer_feature_builder: FusionFeatureBuilder = field(repr=False)

    def to_dict(self) -> dict[str, object]:
        return dict(self.report)


@dataclass(frozen=True, slots=True)
class HistoryOnlyFusionQuery:
    """An opaque inner-fold key, fold-in history, and complete eligible Gold Set."""

    pool_key: str
    history: tuple[int, ...]
    gold_movie_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RetrieverBank:
    retrievers: Mapping[str, CandidateRetriever]
    feature_builder: FusionFeatureBuilder


def run_history_only_benchmark(
    events: Iterable[RatingEvent],
    *,
    config: HistoryOnlyBenchmarkConfig | None = None,
    evaluate_test: bool = False,
) -> HistoryOnlyBenchmarkResult:
    """Run the disjoint-Subject split with a three-source History-Only LHF baseline."""

    config = config or HistoryOnlyBenchmarkConfig()
    dataset_checksum = getattr(events, "dataset_checksum", None)
    subject_events: defaultdict[int, dict[int, RatingEvent]] = defaultdict(dict)
    rating_event_count = 0
    positive_rating_event_count = 0
    fingerprint_xor = 0
    fingerprint_sum = 0
    fingerprint_modulus = 1 << 256
    for event in events:
        rating_event_count += 1
        event_digest = int.from_bytes(
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
        fingerprint_xor ^= event_digest
        fingerprint_sum = (fingerprint_sum + event_digest) % fingerprint_modulus
        if event.rating >= POSITIVE_RATING_THRESHOLD:
            positive_rating_event_count += 1
            previous = subject_events[event.subject_id].get(event.movie_id)
            if previous is None or (event.event_time, event.event_id) < (
                previous.event_time,
                previous.event_id,
            ):
                subject_events[event.subject_id][event.movie_id] = event

    input_fingerprint = hashlib.sha256(
        f"{rating_event_count}:{fingerprint_xor:064x}:{fingerprint_sum:064x}".encode("ascii")
    ).hexdigest()
    unique_positive_count = sum(map(len, subject_events.values()))
    eligible_subject_movies = {
        subject_id: tuple(sorted(movie_events))
        for subject_id, movie_events in subject_events.items()
        if len(movie_events) >= MIN_POSITIVE_INTERACTIONS
    }
    required_subject_count = config.validation_subject_count + config.test_subject_count + 1
    if len(eligible_subject_movies) < required_subject_count:
        raise ValueError(
            "History-Only benchmark requires at least "
            f"{required_subject_count} Subjects with {MIN_POSITIVE_INTERACTIONS} positives "
            "to leave a nonempty training cohort"
        )

    ranked_subject_ids = sorted(
        eligible_subject_movies,
        key=lambda subject_id: (_split_rank(config.seed, "cohort", subject_id), subject_id),
    )
    validation_ids = frozenset(ranked_subject_ids[: config.validation_subject_count])
    test_ids = frozenset(
        ranked_subject_ids[
            config.validation_subject_count : (
                config.validation_subject_count + config.test_subject_count
            )
        ]
    )
    training_ids = set(eligible_subject_movies) - validation_ids - test_ids
    cohorts = {
        "train": tuple(sorted(training_ids)),
        "validation": tuple(sorted(validation_ids)),
        "test": tuple(sorted(test_ids)),
    }

    training_catalog = frozenset(
        movie_id
        for subject_id in cohorts["train"]
        for movie_id in eligible_subject_movies[subject_id]
    )
    short_subject_count = len(subject_events) - len(eligible_subject_movies)
    inner_subject_ids = _inner_subject_partition(cohorts["train"], config)
    inner_fit_ids, inner_pseudo_ids = inner_subject_ids
    inner_eligible_query_count = 0
    training_rows: list[FusionTrainingRow] = []
    inner_gold_event_ids: list[str] = []
    inner_snapshot_fingerprint: str | None = None
    inner_fit_event_count = 0
    inner_fit_event_ids_sha256: str | None = None
    inner_feature_builder: FusionFeatureBuilder | None = None
    if inner_fit_ids:
        inner_events = _events_for_subjects(
            inner_fit_ids, eligible_subject_movies, subject_events
        )
        inner_snapshot = DataSnapshot.from_events(inner_events)
        inner_bank = _fit_retriever_bank(
            inner_snapshot, config=config, seed=config.seed
        )
        inner_catalog = {event.movie_id for event in inner_snapshot.events}
        for index, subject_id in enumerate(inner_pseudo_ids):
            fold_in, held_out = _fold_in_split(
                eligible_subject_movies[subject_id], config.seed, subject_id, "inner-fold-in"
            )
            history = tuple(movie_id for movie_id in fold_in if movie_id in inner_catalog)
            gold = tuple(movie_id for movie_id in held_out if movie_id in inner_catalog)
            if not gold:
                continue
            key = f"inner-{index:06d}"
            query = HistoryOnlyFusionQuery(pool_key=key, history=history, gold_movie_ids=gold)
            pools = _query_pools(inner_bank.retrievers, history, config.candidate_pool_limit)
            training_rows.extend(
                build_history_only_lhf_training_rows(
                    (query,),
                    {key: pools},
                    inner_bank.feature_builder,
                    max_negative_rows_per_query=config.fusion_negative_rows_per_query,
                    seed=config.seed,
                )
            )
            inner_eligible_query_count += 1
            inner_gold_event_ids.extend(
                subject_events[subject_id][movie_id].event_id for movie_id in gold
            )
        inner_snapshot_fingerprint = inner_snapshot.fingerprint
        inner_fit_event_count = inner_snapshot.event_count
        inner_fit_event_ids_sha256 = _event_ids_fingerprint(inner_snapshot.events)
        inner_feature_builder = inner_bank.feature_builder
        del inner_bank, inner_snapshot, inner_events
    rows = tuple(training_rows)
    del training_rows
    inner_eligible_gold_event_count = len(inner_gold_event_ids)
    inner_eligible_gold_event_ids_sha256 = _string_ids_fingerprint(inner_gold_event_ids)
    del inner_gold_event_ids
    training_events = _events_for_subjects(
        cohorts["train"], eligible_subject_movies, subject_events
    )
    del subject_events
    outer_snapshot = DataSnapshot.from_events(training_events)
    if frozenset(event.movie_id for event in outer_snapshot.events) != training_catalog:
        raise RuntimeError("History-Only fit catalog differs from the training-Subject catalog")
    outer_snapshot_fingerprint = outer_snapshot.fingerprint
    outer_fit_event_ids_sha256 = _event_ids_fingerprint(training_events)
    training_interaction_count = len(training_events)
    fusion = train_lhf_classifier(
        query_mode="history_only",
        retriever_bank=HISTORY_ONLY_BANK,
        feature_names=expected_feature_names(HISTORY_ONLY_BANK),
        rows=rows,
        source_snapshot_fingerprint=outer_snapshot_fingerprint,
        seed=config.seed,
        validation_event_ids=(),
        validation_sources=(
            {
                "kind": "paper_inner_pseudo_held_out_training_subjects",
                "inner_fit_snapshot_fingerprint": inner_snapshot_fingerprint,
                "inner_fit_subject_count": len(inner_fit_ids),
                "inner_fit_subject_ids_sha256": _subject_ids_fingerprint(inner_fit_ids),
                "pseudo_held_out_subject_count": len(inner_pseudo_ids),
                "pseudo_held_out_subject_ids_sha256": _subject_ids_fingerprint(inner_pseudo_ids),
            },
        ),
    )
    fusion = replace(
        fusion,
        training_metadata={
            **fusion.training_metadata,
            "training_boundary": "paper_inner_pseudo_held_out_training_subjects",
            "training_snapshot_fingerprint": inner_snapshot_fingerprint,
            "outer_fit_partition": "training_subjects_only",
            "inner_fold_seed": config.seed,
            "inner_fit_subject_count": len(inner_fit_ids),
            "inner_pseudo_held_out_subject_count": len(inner_pseudo_ids),
            "inner_eligible_query_count": inner_eligible_query_count,
            "inner_eligible_gold_event_count": inner_eligible_gold_event_count,
            "inner_eligible_gold_event_ids_sha256": inner_eligible_gold_event_ids_sha256,
            "inner_fit_event_count": inner_fit_event_count,
            "outer_fit_event_count": outer_snapshot.event_count,
        },
    )
    training_row_count = len(rows)
    positive_training_row_count = sum(row.label for row in rows)
    negative_training_row_count = training_row_count - positive_training_row_count
    retained_training_rows = rows if config.capture_evidence else ()
    del rows
    outer_bank = _fit_retriever_bank(outer_snapshot, config=config, seed=config.seed)
    del outer_snapshot, training_events

    split_by_subject: dict[int, HistoryOnlySubjectSplit] = {}
    queries_by_subject: dict[int, HistoryOnlyQuery] = {}
    gold_sets: dict[int, tuple[int, ...]] = {}
    for subject_id in (*cohorts["validation"], *cohorts["test"]):
        fold_in, held_out = _fold_in_split(
            eligible_subject_movies[subject_id], config.seed, subject_id, "fold-in"
        )
        query_history = tuple(movie_id for movie_id in fold_in if movie_id in training_catalog)
        gold_set = tuple(movie_id for movie_id in held_out if movie_id in training_catalog)
        split_by_subject[subject_id] = HistoryOnlySubjectSplit(
            fold_in_movie_ids=fold_in,
            held_out_movie_ids=held_out,
            query_history_movie_ids=query_history,
            gold_movie_ids=gold_set,
            excluded_cold_fold_in_movie_ids=tuple(
                movie_id for movie_id in fold_in if movie_id not in training_catalog
            ),
            excluded_cold_gold_movie_ids=tuple(
                movie_id for movie_id in held_out if movie_id not in training_catalog
            ),
        )
        queries_by_subject[subject_id] = HistoryOnlyQuery(history_movie_ids=query_history)
        gold_sets[subject_id] = gold_set
    eligible_subject_count = len(eligible_subject_movies)
    retained_positive_count = sum(map(len, eligible_subject_movies.values()))
    del eligible_subject_movies

    cohort_counts = {
        "validation": _cohort_counts(cohorts["validation"], split_by_subject, gold_sets)
    }
    if evaluate_test:
        cohort_counts["test"] = _cohort_counts(cohorts["test"], split_by_subject, gold_sets)
    metrics: dict[str, dict[str, int | float | None]] = {}
    diagnostics: dict[str, object] = {}
    candidate_pools: dict[str, object] = {}
    for cohort in ("validation", "test") if evaluate_test else ("validation",):
        cohort_metrics, cohort_diagnostics, cohort_pools = _score_cohort(
            cohorts[cohort],
            split_by_subject,
            queries_by_subject,
            gold_sets,
            training_catalog,
            outer_bank,
            fusion,
            config.candidate_pool_limit,
            config.capture_evidence,
        )
        metrics[cohort] = cohort_metrics
        diagnostics[cohort] = cohort_diagnostics
        candidate_pools[cohort] = cohort_pools
    report: dict[str, object] = {
        "schema_version": 2,
        "protocol": "mult-vae-history-only",
        "baseline": "history-only-lhf",
        "source": {
            "name": "MovieLens 20M" if isinstance(dataset_checksum, str) else "Rating Events",
            "dataset_sha256": dataset_checksum if isinstance(dataset_checksum, str) else None,
            "input_event_fingerprint_sha256": input_fingerprint,
        },
        "evaluation": {
            "query_mode": "history_only",
            "query_features": ["fold_in_history_movie_ids"],
            "eligible_retrievers": ["popularity", "itemknn", "multivae"],
            "final_order": "History-Only Learned Hybrid Fusion Top-100",
            "oracle_union": "deduplicated source-pool union before final Top-100 truncation",
            "candidate_universe": (
                "complete training-visible catalog after fold-in history exclusion"
            ),
            "history_exclusion": "fold-in Movies present in the training catalog",
            "cold_item_policy": (
                "The training catalog contains only training-Subject positive Movies. "
                "Cold fold-in Movies are omitted from the scoring Query; cold held-out "
                "Movies are recorded and excluded from the eligible Gold Set denominator."
            ),
            "sampled_negatives": False,
            "synthetic_gold_insertion": False,
            "recall_denominator": "min(K, eligible Gold Set size)",
            "ndcg_relevance": "binary",
            "macro_average": "Subject with at least one eligible Gold Movie",
        },
        "reproducibility": {
            "split_seed": config.seed,
            "split_algorithm": "sha256-ranked-subject-and-fold-in-v1",
            "positive_rating_threshold": POSITIVE_RATING_THRESHOLD,
            "minimum_positive_interactions_per_subject": MIN_POSITIVE_INTERACTIONS,
            "validation_subject_count": config.validation_subject_count,
            "test_subject_count": config.test_subject_count,
            "fold_in_fraction": 1.0 - HELD_OUT_FRACTION,
            "held_out_fraction": HELD_OUT_FRACTION,
            "held_out_rounding": "ceil",
            "metric_cutoffs": list(METRIC_CUTOFFS),
            "candidate_pool_limit_per_source": config.candidate_pool_limit,
            "inner_held_out_fraction": config.inner_held_out_fraction,
            "inner_fold_seed": config.seed,
            "fusion_negative_rows_per_query": config.fusion_negative_rows_per_query,
            "multivae_config": config.multivae_config.to_dict(),
            "multivae_device_preference": config.multivae_device_preference,
            "capture_evidence": config.capture_evidence,
            "popularity_tie_break": "movie_id_ascending",
            "implementation": "history-only-three-source-lhf-v1",
            "implementation_sha256": _implementation_fingerprint(),
            "test_evaluated": evaluate_test,
        },
        "counts": {
            "rating_event_count": rating_event_count,
            "positive_rating_event_count": positive_rating_event_count,
            "duplicate_positive_interaction_count": (
                positive_rating_event_count - unique_positive_count
            ),
            "unique_positive_interaction_count": unique_positive_count,
            "eligible_subject_count": eligible_subject_count,
            "training_subject_count": len(cohorts["train"]),
            "validation_subject_count": len(cohorts["validation"]),
            "test_subject_count": len(cohorts["test"]),
            "training_interaction_count": training_interaction_count,
            "training_catalog_movie_count": len(training_catalog),
            "inner_fit_subject_count": len(inner_fit_ids),
            "inner_pseudo_held_out_subject_count": len(inner_pseudo_ids),
            "inner_eligible_fusion_query_count": inner_eligible_query_count,
            "validation": cohort_counts["validation"],
            "test": (
                cohort_counts["test"]
                if evaluate_test
                else {"subject_count": len(cohorts["test"]), "status": "sealed"}
            ),
        },
        "training_catalog_movie_ids": sorted(training_catalog),
        "cohort_subject_ids": {cohort: list(ids) for cohort, ids in cohorts.items()},
        "held_out_subject_splits": {
            "validation": {
                str(subject_id): _split_to_dict(split_by_subject[subject_id])
                for subject_id in cohorts["validation"]
            },
            "test": (
                {
                    str(subject_id): _split_to_dict(split_by_subject[subject_id])
                    for subject_id in cohorts["test"]
                }
                if evaluate_test
                else {
                    "status": "sealed",
                    "membership_sha256": _test_split_fingerprint(cohorts["test"], split_by_subject),
                }
            ),
        },
        "exclusions": {
            "positive_interaction_count_removed_with_short_subjects": (
                unique_positive_count - retained_positive_count
            ),
            "subject_count_removed_with_fewer_than_five_positives": (
                short_subject_count
            ),
            "validation": _cohort_exclusions(cohorts["validation"], split_by_subject),
            "test": (
                _cohort_exclusions(cohorts["test"], split_by_subject)
                if evaluate_test
                else {"status": "sealed"}
            ),
        },
        "fusion_training": {
            "query_mode": "history_only",
            "retriever_bank": list(HISTORY_ONLY_BANK),
            "feature_names": list(outer_bank.feature_builder.feature_names),
            "training_status": fusion.training_status,
            "training_metadata": dict(fusion.training_metadata),
            "outer_fit_snapshot_fingerprint": outer_snapshot_fingerprint,
            "inner_fit_snapshot_fingerprint": inner_snapshot_fingerprint,
            "outer_fit_subject_count": len(cohorts["train"]),
            "outer_fit_subject_ids_sha256": _subject_ids_fingerprint(cohorts["train"]),
            "inner_fit_subject_count": len(inner_fit_ids),
            "inner_fit_subject_ids_sha256": _subject_ids_fingerprint(inner_fit_ids),
            "inner_pseudo_held_out_subject_count": len(inner_pseudo_ids),
            "inner_pseudo_held_out_subject_ids_sha256": _subject_ids_fingerprint(
                inner_pseudo_ids
            ),
            "outer_fit_event_ids_sha256": outer_fit_event_ids_sha256,
            "inner_fit_event_ids_sha256": inner_fit_event_ids_sha256,
            "training_row_count": training_row_count,
            "positive_training_row_count": positive_training_row_count,
            "negative_training_row_count": negative_training_row_count,
        },
        "diagnostics": diagnostics,
        "candidate_pools": (
            candidate_pools if config.capture_evidence else {"status": "omitted"}
        ),
        "metrics": metrics,
    }
    visible_subject_ids = (
        (*cohorts["validation"], *cohorts["test"]) if evaluate_test else cohorts["validation"]
    )
    return HistoryOnlyBenchmarkResult(
        report=report,
        cohort_subject_ids=cohorts,
        split_by_subject={
            subject_id: split_by_subject[subject_id] for subject_id in visible_subject_ids
        },
        queries_by_subject={
            subject_id: queries_by_subject[subject_id] for subject_id in visible_subject_ids
        },
        gold_sets={subject_id: gold_sets[subject_id] for subject_id in visible_subject_ids},
        training_catalog=training_catalog,
        fusion=fusion,
        fusion_training_rows=retained_training_rows,
        inner_fit_subject_ids=inner_fit_ids,
        inner_pseudo_held_out_subject_ids=inner_pseudo_ids,
        inner_feature_builder=inner_feature_builder,
        outer_feature_builder=outer_bank.feature_builder,
    )


def _events_for_subjects(
    subject_ids: Sequence[int],
    subject_movies: Mapping[int, tuple[int, ...]],
    subject_events: Mapping[int, Mapping[int, RatingEvent]],
) -> tuple[RatingEvent, ...]:
    return tuple(
        subject_events[subject_id][movie_id]
        for subject_id in sorted(subject_ids)
        for movie_id in subject_movies[subject_id]
    )


def _fit_retriever_bank(
    snapshot: DataSnapshot,
    *,
    config: HistoryOnlyBenchmarkConfig,
    seed: int,
) -> _RetrieverBank:
    interactions = tuple(PositiveInteraction.from_event(event) for event in snapshot.events)
    retrievers: dict[str, CandidateRetriever] = {
        "popularity": fit_popularity(snapshot, interactions),
        "itemknn": fit_itemknn(snapshot, interactions).to_serving(),
        "multivae": fit_multivae(
            snapshot,
            interactions,
            config=config.multivae_config,
            seed=seed,
            device_preference=config.multivae_device_preference,
        ),
    }
    feature_builder = FusionFeatureBuilder.from_snapshot(
        snapshot, interactions, retriever_bank=HISTORY_ONLY_BANK
    )
    return _RetrieverBank(retrievers, feature_builder)


def _inner_subject_partition(
    training_subject_ids: tuple[int, ...], config: HistoryOnlyBenchmarkConfig
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(training_subject_ids) < 2:
        return (), ()
    ranked = sorted(
        training_subject_ids,
        key=lambda subject_id: (
            _split_rank(config.seed, "inner-cohort", subject_id),
            subject_id,
        ),
    )
    pseudo_count = min(
        len(training_subject_ids) - 1,
        max(1, math.ceil(len(training_subject_ids) * config.inner_held_out_fraction)),
    )
    pseudo_ids = frozenset(ranked[:pseudo_count])
    return (
        tuple(sorted(set(training_subject_ids) - pseudo_ids)),
        tuple(sorted(pseudo_ids)),
    )


def _fold_in_split(
    movie_ids: tuple[int, ...], seed: int, subject_id: int, partition: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ranked = sorted(
        movie_ids,
        key=lambda movie_id: (_split_rank(seed, partition, subject_id, movie_id), movie_id),
    )
    held_out_count = math.ceil(len(ranked) * HELD_OUT_FRACTION)
    return tuple(sorted(ranked[held_out_count:])), tuple(sorted(ranked[:held_out_count]))


def _query_pools(
    retrievers: Mapping[str, CandidateRetriever], history: tuple[int, ...], limit: int
) -> dict[str, tuple[Candidate, ...]]:
    return {
        name: candidate_pool_for_query(retrievers[name], None, history, limit=limit)
        for name in HISTORY_ONLY_BANK
    }


def build_history_only_lhf_training_rows(
    queries: Sequence[HistoryOnlyFusionQuery],
    pools: Mapping[int | str, Mapping[str, Sequence[Candidate]]],
    feature_builder: FusionFeatureBuilder,
    *,
    max_negative_rows_per_query: int | None = None,
    seed: int = 42,
) -> tuple[FusionTrainingRow, ...]:
    """Label every naturally retrieved inner Gold Movie; sample only negative rows."""

    if feature_builder.retriever_bank != HISTORY_ONLY_BANK:
        raise ValueError("History-Only fusion features require the three-source bank")
    return build_lhf_training_rows(
        queries,
        pools,
        feature_builder,
        max_negative_rows_per_query=max_negative_rows_per_query,
        seed=seed,
    )


def _score_cohort(
    subject_ids: tuple[int, ...],
    split_by_subject: Mapping[int, HistoryOnlySubjectSplit],
    queries_by_subject: Mapping[int, HistoryOnlyQuery],
    gold_sets: Mapping[int, tuple[int, ...]],
    training_catalog: frozenset[int],
    bank: _RetrieverBank,
    fusion: LearnedHybridFusion,
    pool_limit: int,
    capture_evidence: bool,
) -> tuple[dict[str, int | float | None], dict[str, object], dict[str, object]]:
    source_accumulators = {name: _MetricAccumulator() for name in HISTORY_ONLY_BANK}
    final_accumulator = _MetricAccumulator()
    source_full_pool_gold_hits = {name: 0 for name in HISTORY_ONLY_BANK}
    source_full_pool_covered_queries = {name: 0 for name in HISTORY_ONLY_BANK}
    source_full_pool_candidate_occurrences = {name: 0 for name in HISTORY_ONLY_BANK}
    oracle_query_count = 0
    oracle_covered_query_count = 0
    oracle_gold_count = 0
    oracle_retrieved_gold_count = 0
    oracle_candidate_occurrence_count = 0
    reported_pools: dict[str, object] = {}
    loss = PaperGoldDiagnostics(
        mode="history_only",
        source_names=HISTORY_ONLY_BANK,
        training_catalog=training_catalog,
        training_popularity=bank.feature_builder.item_popularity,
    )
    for subject_id in subject_ids:
        gold_set = gold_sets[subject_id]
        if not gold_set:
            loss.observe(
                raw_gold_movie_ids=split_by_subject[subject_id].held_out_movie_ids,
                eligible_gold_movie_ids=(),
                history=queries_by_subject[subject_id].history_movie_ids,
                pools={name: () for name in HISTORY_ONLY_BANK},
                fusion_input_movie_ids=(),
                final_ranking=(),
            )
            continue
        history = queries_by_subject[subject_id].history_movie_ids
        pools = _query_pools(bank.retrievers, history, pool_limit)
        union = build_candidate_union(pools, history=history)
        final = rank_lhf_union(fusion, bank.feature_builder, pools, history, top_n=100)
        loss.observe(
            raw_gold_movie_ids=split_by_subject[subject_id].held_out_movie_ids,
            eligible_gold_movie_ids=gold_set,
            history=history,
            pools=pools,
            fusion_input_movie_ids={candidate.movie_id for candidate in union},
            final_ranking=final,
        )
        gold_ids = set(gold_set)
        union_ids = {candidate.movie_id for candidate in union}
        union_hits = len(union_ids & gold_ids)
        oracle_query_count += 1
        oracle_covered_query_count += bool(union_hits)
        oracle_gold_count += len(gold_set)
        oracle_retrieved_gold_count += union_hits
        oracle_candidate_occurrence_count += len(union_ids)
        for name, pool in pools.items():
            source_accumulators[name].add(pool, gold_set)
            full_pool_hits = len(gold_ids & {candidate.movie_id for candidate in pool})
            source_full_pool_gold_hits[name] += full_pool_hits
            source_full_pool_covered_queries[name] += bool(full_pool_hits)
            source_full_pool_candidate_occurrences[name] += len(pool)
        final_accumulator.add(final, gold_set)
        if capture_evidence:
            reported_pools[str(subject_id)] = {
                "sources": {
                    name: [candidate.movie_id for candidate in pools[name]]
                    for name in HISTORY_ONLY_BANK
                },
                "oracle_union_movie_ids": [candidate.movie_id for candidate in union],
                "final_lhf_top100_movie_ids": [candidate.movie_id for candidate in final],
            }
    final_metrics = final_accumulator.finish(training_catalog)
    source_metrics = {
        name: source_accumulators[name].finish(training_catalog)
        for name in HISTORY_ONLY_BANK
    }
    diagnostic = {
        "sources": source_metrics,
        "source_full_pools": {
            name: {
                "eligible_gold_movie_count": oracle_gold_count,
                "retrieved_gold_movie_count": source_full_pool_gold_hits[name],
                "gold_retrieval_fraction": (
                    source_full_pool_gold_hits[name] / oracle_gold_count
                    if oracle_gold_count
                    else 0.0
                ),
                "query_retrieval_coverage": (
                    source_full_pool_covered_queries[name] / oracle_query_count
                    if oracle_query_count
                    else 0.0
                ),
                "candidate_occurrence_count": source_full_pool_candidate_occurrences[name],
            }
            for name in HISTORY_ONLY_BANK
        },
        "oracle_union": {
            "query_count": oracle_query_count,
            "eligible_gold_movie_count": oracle_gold_count,
            "retrieved_gold_movie_count": oracle_retrieved_gold_count,
            "gold_retrieval_fraction": (
                oracle_retrieved_gold_count / oracle_gold_count if oracle_gold_count else 0.0
            ),
            "query_retrieval_coverage": (
                oracle_covered_query_count / oracle_query_count if oracle_query_count else 0.0
            ),
            "candidate_occurrence_count": oracle_candidate_occurrence_count,
        },
        "final_lhf": final_metrics,
        "gold_loss": loss.finish(),
    }
    return final_metrics, diagnostic, reported_pools


def _event_ids_fingerprint(events: Sequence[RatingEvent]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(event.event_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _string_ids_fingerprint(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _subject_ids_fingerprint(subject_ids: Sequence[int]) -> str:
    return _string_ids_fingerprint(tuple(str(subject_id) for subject_id in sorted(subject_ids)))


def _split_rank(seed: int, partition: str, subject_id: int, movie_id: int | None = None) -> bytes:
    value = f"history-only-v1:{seed}:{partition}:{subject_id}"
    if movie_id is not None:
        value += f":{movie_id}"
    return hashlib.sha256(value.encode("ascii")).digest()


def _split_to_dict(split: HistoryOnlySubjectSplit) -> dict[str, list[int]]:
    return {
        "fold_in_movie_ids": list(split.fold_in_movie_ids),
        "held_out_movie_ids": list(split.held_out_movie_ids),
        "query_history_movie_ids": list(split.query_history_movie_ids),
        "eligible_gold_movie_ids": list(split.gold_movie_ids),
        "excluded_cold_fold_in_movie_ids": list(split.excluded_cold_fold_in_movie_ids),
        "excluded_cold_gold_movie_ids": list(split.excluded_cold_gold_movie_ids),
    }


def _test_split_fingerprint(
    subject_ids: tuple[int, ...], splits: Mapping[int, HistoryOnlySubjectSplit]
) -> str:
    digest = hashlib.sha256()
    for subject_id in subject_ids:
        split = splits[subject_id]
        for partition, movie_ids in (
            ("fold_in", split.fold_in_movie_ids),
            ("held_out", split.held_out_movie_ids),
        ):
            for movie_id in movie_ids:
                digest.update(f"{subject_id}:{partition}:{movie_id}\n".encode("ascii"))
    return digest.hexdigest()


def _cohort_counts(
    subject_ids: tuple[int, ...],
    splits: Mapping[int, HistoryOnlySubjectSplit],
    gold_sets: Mapping[int, tuple[int, ...]],
) -> dict[str, int]:
    return {
        "subject_count": len(subject_ids),
        "fold_in_interaction_count": sum(
            len(splits[subject_id].fold_in_movie_ids) for subject_id in subject_ids
        ),
        "held_out_interaction_count": sum(
            len(splits[subject_id].held_out_movie_ids) for subject_id in subject_ids
        ),
        "eligible_gold_movie_count": sum(len(gold_sets[subject_id]) for subject_id in subject_ids),
        "evaluation_query_count": sum(bool(gold_sets[subject_id]) for subject_id in subject_ids),
    }


def _cohort_exclusions(
    subject_ids: tuple[int, ...], splits: Mapping[int, HistoryOnlySubjectSplit]
) -> dict[str, int]:
    return {
        "fold_in_movie_count_missing_from_training_catalog": sum(
            len(splits[subject_id].excluded_cold_fold_in_movie_ids) for subject_id in subject_ids
        ),
        "held_out_gold_movie_count_missing_from_training_catalog": sum(
            len(splits[subject_id].excluded_cold_gold_movie_ids) for subject_id in subject_ids
        ),
        "subjects_without_eligible_gold": sum(
            not splits[subject_id].gold_movie_ids for subject_id in subject_ids
        ),
        "subjects_without_eligible_fold_in_history": sum(
            not splits[subject_id].query_history_movie_ids for subject_id in subject_ids
        ),
    }


@dataclass(slots=True)
class _MetricAccumulator:
    recall_totals: dict[int, float] = field(
        default_factory=lambda: {cutoff: 0.0 for cutoff in METRIC_CUTOFFS}
    )
    ndcg_totals: dict[int, float] = field(
        default_factory=lambda: {cutoff: 0.0 for cutoff in METRIC_CUTOFFS}
    )
    query_count: int = 0
    covered_queries: int = 0
    recommended_movies: set[int] = field(default_factory=set)

    def add(self, ranked_candidates: Sequence[Candidate], gold_set: tuple[int, ...]) -> None:
        self.query_count += 1
        ranked_movies = tuple(candidate.movie_id for candidate in ranked_candidates[:100])
        rank_by_movie = {movie_id: rank for rank, movie_id in enumerate(ranked_movies, 1)}
        self.covered_queries += any(movie_id in rank_by_movie for movie_id in gold_set)
        self.recommended_movies.update(ranked_movies)
        for cutoff in METRIC_CUTOFFS:
            hit_ranks = [
                rank_by_movie[movie_id]
                for movie_id in gold_set
                if movie_id in rank_by_movie and rank_by_movie[movie_id] <= cutoff
            ]
            self.recall_totals[cutoff] += len(hit_ranks) / min(cutoff, len(gold_set))
            discounted_gain = sum(1.0 / math.log2(rank + 1) for rank in hit_ranks)
            ideal_gain = sum(
                1.0 / math.log2(rank + 1) for rank in range(1, min(cutoff, len(gold_set)) + 1)
            )
            self.ndcg_totals[cutoff] += discounted_gain / ideal_gain

    def finish(self, training_catalog: frozenset[int]) -> dict[str, int | float | None]:
        divisor = self.query_count or 1
        return {
            "query_count": self.query_count,
            **{
                f"Recall@{cutoff}": total / divisor
                for cutoff, total in self.recall_totals.items()
            },
            **{
                f"NDCG@{cutoff}": total / divisor
                for cutoff, total in self.ndcg_totals.items()
            },
            "QueryRetrievalCoverage@100": self.covered_queries / divisor,
            "CatalogCoverage@100": (
                len(self.recommended_movies) / len(training_catalog)
                if training_catalog and self.query_count
                else None
            ),
        }


def _implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in (
        "history_only_benchmark.py",
        "serving.py",
        "itemknn.py",
        "multivae.py",
        "fusion.py",
    ):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def write_history_only_benchmark_report(result: HistoryOnlyBenchmarkResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
