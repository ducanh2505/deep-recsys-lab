from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .models import RatingEvent
from .serving import PopularityRetriever

POSITIVE_RATING_THRESHOLD = 4.0
K_CORE_MIN_DEGREE = 10
TEST_FRACTION = 0.20
VALIDATION_FRACTION_OF_TRAIN = 0.10
METRIC_CUTOFFS = (10, 20, 50, 100)


@dataclass(frozen=True, slots=True)
class KnownUserSubjectSplit:
    train_movie_ids: tuple[int, ...]
    outer_train_movie_ids: tuple[int, ...]
    validation_movie_ids: tuple[int, ...]
    test_movie_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class KnownUserBenchmarkResult:
    report: Mapping[str, object]
    split_by_subject: Mapping[int, KnownUserSubjectSplit]
    gold_sets: Mapping[int, tuple[int, ...]]
    training_catalog: frozenset[int]

    def to_dict(self) -> dict[str, object]:
        return dict(self.report)


def run_known_user_benchmark(
    events: Iterable[RatingEvent], *, seed: int = 42
) -> KnownUserBenchmarkResult:
    """Build a deterministic Known-User split and report a Popularity baseline."""

    dataset_checksum = getattr(events, "dataset_checksum", None)
    positive_by_pair: dict[tuple[int, int], RatingEvent] = {}
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
        if not event.rating >= POSITIVE_RATING_THRESHOLD:
            continue
        positive_rating_event_count += 1
        pair = (event.subject_id, event.movie_id)
        current = positive_by_pair.get(pair)
        if current is None or (event.event_time, event.event_id) < (
            current.event_time,
            current.event_id,
        ):
            positive_by_pair[pair] = event

    input_fingerprint = hashlib.sha256(
        f"{rating_event_count}:{fingerprint_xor:064x}:{fingerprint_sum:064x}".encode("ascii")
    ).hexdigest()
    duplicate_positive_interaction_count = positive_rating_event_count - len(positive_by_pair)
    positive_before_k_core_count = len(positive_by_pair)
    retained = _filter_to_k_core(positive_by_pair, K_CORE_MIN_DEGREE)
    subject_events: dict[int, list[RatingEvent]] = {}
    for event in retained.values():
        subject_events.setdefault(event.subject_id, []).append(event)

    split_by_subject: dict[int, KnownUserSubjectSplit] = {}
    train_events: list[RatingEvent] = []
    validation_event_count = 0
    test_event_count = 0
    outer_train_event_count = 0
    for subject_id in sorted(subject_events):
        subject_interactions = subject_events[subject_id]
        ordered_for_test = sorted(
            subject_interactions,
            key=lambda event: (
                _split_rank(seed, "test", subject_id, event.movie_id),
                event.movie_id,
            ),
        )
        test_count = max(1, math.ceil(len(subject_interactions) * TEST_FRACTION))
        test_events = ordered_for_test[:test_count]
        test_ids = {event.movie_id for event in test_events}
        outer_train_events = [
            event for event in subject_interactions if event.movie_id not in test_ids
        ]
        ordered_for_validation = sorted(
            outer_train_events,
            key=lambda event: (
                _split_rank(seed, "validation", subject_id, event.movie_id),
                event.movie_id,
            ),
        )
        validation_count = max(1, math.ceil(len(outer_train_events) * VALIDATION_FRACTION_OF_TRAIN))
        validation_events = ordered_for_validation[:validation_count]
        validation_ids = {event.movie_id for event in validation_events}
        fitted_train_events = [
            event for event in outer_train_events if event.movie_id not in validation_ids
        ]
        train_events.extend(fitted_train_events)
        outer_train_event_count += len(outer_train_events)
        validation_event_count += len(validation_events)
        test_event_count += len(test_events)
        split_by_subject[subject_id] = KnownUserSubjectSplit(
            train_movie_ids=tuple(sorted(event.movie_id for event in fitted_train_events)),
            outer_train_movie_ids=tuple(sorted(event.movie_id for event in outer_train_events)),
            validation_movie_ids=tuple(sorted(event.movie_id for event in validation_events)),
            test_movie_ids=tuple(sorted(event.movie_id for event in test_events)),
        )

    catalog = frozenset(event.movie_id for event in train_events)
    gold_sets: dict[int, tuple[int, ...]] = {}
    cold_test_gold_by_subject: dict[int, tuple[int, ...]] = {}
    missing_from_catalog_count = 0
    history_overlap_count = 0
    subjects_without_test_gold = 0
    for subject_id, split in split_by_subject.items():
        cold_test_gold = tuple(
            movie_id for movie_id in split.test_movie_ids if movie_id not in catalog
        )
        if cold_test_gold:
            cold_test_gold_by_subject[subject_id] = cold_test_gold
        missing_from_catalog_count += len(cold_test_gold)
        history_overlap_count += sum(
            movie_id in split.train_movie_ids for movie_id in split.test_movie_ids
        )
        eligible_gold = tuple(
            movie_id
            for movie_id in split.test_movie_ids
            if movie_id in catalog and movie_id not in split.train_movie_ids
        )
        gold_sets[subject_id] = eligible_gold
        if not eligible_gold:
            subjects_without_test_gold += 1

    popularity_counts = Counter(event.movie_id for event in train_events)
    subject_histories: dict[int, tuple[int, ...]] = {}
    for subject_id, split in split_by_subject.items():
        subject_histories[subject_id] = split.train_movie_ids
    popularity = PopularityRetriever(
        catalog=tuple(sorted(catalog)),
        counts=popularity_counts,
        subject_histories=subject_histories,
    )
    metrics = _known_user_metrics(popularity, split_by_subject, gold_sets, catalog)

    report: dict[str, object] = {
        "schema_version": 1,
        "protocol": "lightgcn-ngcf-known-user",
        "baseline": "popularity",
        "source": {
            "name": "MovieLens 20M" if isinstance(dataset_checksum, str) else "Rating Events",
            "dataset_sha256": dataset_checksum if isinstance(dataset_checksum, str) else None,
            "input_event_fingerprint_sha256": input_fingerprint,
        },
        "evaluation": {
            "candidate_universe": (
                "complete final training catalog after Subject history exclusion"
            ),
            "history_exclusion": "final training interactions for the Subject",
            "sampled_negatives": False,
            "synthetic_gold_insertion": False,
            "cold_item_policy": (
                "exclude test Gold Movies absent from the final training catalog "
                "before calculating the denominator"
            ),
            "recall_denominator": "complete eligible Gold Set size",
            "ndcg_relevance": "binary",
            "macro_average": "Subject",
        },
        "reproducibility": {
            "split_seed": seed,
            "split_algorithm": "sha256-ranked-per-subject-v1",
            "positive_rating_threshold": POSITIVE_RATING_THRESHOLD,
            "k_core_min_degree": K_CORE_MIN_DEGREE,
            "test_fraction": TEST_FRACTION,
            "validation_fraction_of_train": VALIDATION_FRACTION_OF_TRAIN,
            "metric_cutoffs": list(METRIC_CUTOFFS),
            "subject_cohort_cap": None,
            "training_catalog_source": "final_training_interactions",
            "history_source": "final_training_interactions",
            "test_fit_partition": "train_movie_ids",
            "validation_partition": "validation_movie_ids reserved from outer_train_movie_ids",
            "validation_refit_before_test": False,
            "popularity_tie_break": "movie_id_ascending",
            "package_source_sha256": _package_source_fingerprint(),
            "split_membership_sha256": {
                partition: _membership_fingerprint(split_by_subject, partition)
                for partition in ("train", "validation", "test")
            },
        },
        "counts": {
            "rating_event_count": rating_event_count,
            "positive_rating_event_count": positive_rating_event_count,
            "duplicate_positive_interaction_count": duplicate_positive_interaction_count,
            "positive_interaction_count_before_10_core": positive_before_k_core_count,
            "positive_interaction_count_after_10_core": len(retained),
            "eligible_subject_count": len(split_by_subject),
            "outer_train_interaction_count": outer_train_event_count,
            "validation_interaction_count": validation_event_count,
            "training_interaction_count": len(train_events),
            "test_interaction_count": test_event_count,
            "evaluation_query_count": len(gold_sets) - subjects_without_test_gold,
            "eligible_test_gold_movie_count": sum(map(len, gold_sets.values())),
            "training_catalog_movie_count": len(catalog),
        },
        "training_catalog_movie_ids": sorted(catalog),
        "split_membership_by_subject": {
            str(subject_id): {
                "train_movie_ids": list(split.train_movie_ids),
                "validation_movie_ids": list(split.validation_movie_ids),
                "outer_train_movie_ids": list(split.outer_train_movie_ids),
                "test_movie_ids": list(split.test_movie_ids),
            }
            for subject_id, split in split_by_subject.items()
        },
        "gold_sets_by_subject": {
            str(subject_id): list(gold_set) for subject_id, gold_set in gold_sets.items()
        },
        "exclusions": {
            "positive_interaction_count_removed_by_10_core": (
                positive_before_k_core_count - len(retained)
            ),
            "test_gold_movie_count_missing_from_training_catalog": missing_from_catalog_count,
            "test_gold_movie_ids_missing_from_training_catalog_by_subject": {
                str(subject_id): list(movie_ids)
                for subject_id, movie_ids in cold_test_gold_by_subject.items()
            },
            "test_gold_movie_count_overlapping_training_history": history_overlap_count,
            "training_catalog_history_movie_occurrence_count": sum(
                movie_id in catalog
                for split in split_by_subject.values()
                for movie_id in split.train_movie_ids
            ),
            "subjects_without_eligible_test_gold": subjects_without_test_gold,
        },
        "metrics": metrics,
    }
    return KnownUserBenchmarkResult(
        report=report,
        split_by_subject=split_by_subject,
        gold_sets=gold_sets,
        training_catalog=catalog,
    )


def _filter_to_k_core(
    interactions: Mapping[tuple[int, int], RatingEvent], min_degree: int
) -> dict[tuple[int, int], RatingEvent]:
    retained = dict(interactions)
    while retained:
        subject_degrees = Counter(subject_id for subject_id, _movie_id in retained)
        movie_degrees = Counter(movie_id for _subject_id, movie_id in retained)
        next_retained = {
            pair: event
            for pair, event in retained.items()
            if subject_degrees[pair[0]] >= min_degree and movie_degrees[pair[1]] >= min_degree
        }
        if len(next_retained) == len(retained):
            return next_retained
        retained = next_retained
    return retained


def _split_rank(seed: int, partition: str, subject_id: int, movie_id: int) -> bytes:
    value = f"known-user-v1:{seed}:{partition}:{subject_id}:{movie_id}"
    return hashlib.sha256(value.encode("ascii")).digest()


def _known_user_metrics(
    popularity: PopularityRetriever,
    splits: Mapping[int, KnownUserSubjectSplit],
    gold_sets: Mapping[int, tuple[int, ...]],
    training_catalog: frozenset[int],
) -> dict[str, int | float | None]:
    recall_totals = {cutoff: 0.0 for cutoff in METRIC_CUTOFFS}
    ndcg_totals = {cutoff: 0.0 for cutoff in METRIC_CUTOFFS}
    query_count = 0
    covered_queries = 0
    recommended_movies: set[int] = set()
    global_ranking = popularity.ranked_movie_ids()

    for subject_id, gold_set in gold_sets.items():
        if not gold_set:
            continue
        query_count += 1
        history = splits[subject_id].train_movie_ids
        history_ids = set(history)
        ranked_movie_ids = tuple(
            movie_id for movie_id in global_ranking if movie_id not in history_ids
        )[:100]
        rank_by_movie = {movie_id: rank for rank, movie_id in enumerate(ranked_movie_ids, start=1)}
        if any(
            rank_by_movie.get(movie_id, len(ranked_movie_ids) + 1) <= 100 for movie_id in gold_set
        ):
            covered_queries += 1
        recommended_movies.update(ranked_movie_ids[:100])

        for cutoff in METRIC_CUTOFFS:
            hits = [
                rank_by_movie[movie_id]
                for movie_id in gold_set
                if movie_id in rank_by_movie and rank_by_movie[movie_id] <= cutoff
            ]
            recall_totals[cutoff] += len(hits) / len(gold_set)
            discounted_gain = sum(1.0 / math.log2(rank + 1) for rank in hits)
            ideal_gain = sum(
                1.0 / math.log2(rank + 1) for rank in range(1, min(cutoff, len(gold_set)) + 1)
            )
            ndcg_totals[cutoff] += discounted_gain / ideal_gain if ideal_gain else 0.0

    divisor = query_count or 1
    return {
        "query_count": query_count,
        **{f"Recall@{cutoff}": value / divisor for cutoff, value in recall_totals.items()},
        **{f"NDCG@{cutoff}": value / divisor for cutoff, value in ndcg_totals.items()},
        "QueryRetrievalCoverage@100": covered_queries / divisor if query_count else 0.0,
        "CatalogCoverage@100": (
            len(recommended_movies) / len(training_catalog)
            if training_catalog and query_count
            else None
        ),
    }


def _membership_fingerprint(splits: Mapping[int, KnownUserSubjectSplit], partition: str) -> str:
    digest = hashlib.sha256()
    for subject_id in sorted(splits):
        split = splits[subject_id]
        if partition == "train":
            movie_ids = split.train_movie_ids
        elif partition == "validation":
            movie_ids = split.validation_movie_ids
        elif partition == "test":
            movie_ids = split.test_movie_ids
        else:
            raise ValueError(f"unsupported split partition: {partition}")
        for movie_id in movie_ids:
            digest.update(f"{subject_id}:{movie_id}\n".encode("ascii"))
    return digest.hexdigest()


def _package_source_fingerprint() -> str:
    digest = hashlib.sha256()
    for source_file in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source_file.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(source_file.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def write_known_user_benchmark_report(result: KnownUserBenchmarkResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
