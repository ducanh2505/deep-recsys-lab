from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from .models import RatingEvent
from .serving import PopularityRetriever

POSITIVE_RATING_THRESHOLD = 4.0
MIN_POSITIVE_INTERACTIONS = 5
HELD_OUT_FRACTION = 0.20
METRIC_CUTOFFS = (10, 20, 50, 100)


@dataclass(frozen=True, slots=True)
class HistoryOnlyBenchmarkConfig:
    seed: int = 42
    validation_subject_count: int = 10_000
    test_subject_count: int = 10_000

    def __post_init__(self) -> None:
        if self.validation_subject_count < 1 or self.test_subject_count < 1:
            raise ValueError("validation and test cohorts must each contain at least one Subject")


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

    def to_dict(self) -> dict[str, object]:
        return dict(self.report)


def run_history_only_benchmark(
    events: Iterable[RatingEvent],
    *,
    config: HistoryOnlyBenchmarkConfig | None = None,
    evaluate_test: bool = False,
) -> HistoryOnlyBenchmarkResult:
    """Run the disjoint-Subject Mult-VAE split with a Popularity baseline."""

    config = config or HistoryOnlyBenchmarkConfig()
    dataset_checksum = getattr(events, "dataset_checksum", None)
    subject_movies: defaultdict[int, set[int]] = defaultdict(set)
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
            subject_movies[event.subject_id].add(event.movie_id)

    input_fingerprint = hashlib.sha256(
        f"{rating_event_count}:{fingerprint_xor:064x}:{fingerprint_sum:064x}".encode("ascii")
    ).hexdigest()
    unique_positive_count = sum(map(len, subject_movies.values()))
    eligible_subject_movies = {
        subject_id: tuple(sorted(movie_ids))
        for subject_id, movie_ids in subject_movies.items()
        if len(movie_ids) >= MIN_POSITIVE_INTERACTIONS
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
    popularity_counts = Counter(
        movie_id
        for subject_id in cohorts["train"]
        for movie_id in eligible_subject_movies[subject_id]
    )
    popularity = PopularityRetriever(
        catalog=tuple(sorted(training_catalog)),
        counts=popularity_counts,
        subject_histories={},
    )

    split_by_subject: dict[int, HistoryOnlySubjectSplit] = {}
    queries_by_subject: dict[int, HistoryOnlyQuery] = {}
    gold_sets: dict[int, tuple[int, ...]] = {}
    for subject_id in (*cohorts["validation"], *cohorts["test"]):
        ranked_movies = sorted(
            eligible_subject_movies[subject_id],
            key=lambda movie_id: (
                _split_rank(config.seed, "fold-in", subject_id, movie_id),
                movie_id,
            ),
        )
        held_out_count = math.ceil(len(ranked_movies) * HELD_OUT_FRACTION)
        held_out = tuple(sorted(ranked_movies[:held_out_count]))
        fold_in = tuple(sorted(ranked_movies[held_out_count:]))
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

    cohort_counts = {
        "validation": _cohort_counts(cohorts["validation"], split_by_subject, gold_sets)
    }
    if evaluate_test:
        cohort_counts["test"] = _cohort_counts(cohorts["test"], split_by_subject, gold_sets)
    metrics = {
        "validation": _history_only_metrics(
            popularity,
            cohorts["validation"],
            queries_by_subject,
            gold_sets,
            training_catalog,
        )
    }
    if evaluate_test:
        metrics["test"] = _history_only_metrics(
            popularity,
            cohorts["test"],
            queries_by_subject,
            gold_sets,
            training_catalog,
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "protocol": "mult-vae-history-only",
        "baseline": "popularity",
        "source": {
            "name": "MovieLens 20M" if isinstance(dataset_checksum, str) else "Rating Events",
            "dataset_sha256": dataset_checksum if isinstance(dataset_checksum, str) else None,
            "input_event_fingerprint_sha256": input_fingerprint,
        },
        "evaluation": {
            "query_mode": "history_only",
            "query_features": ["fold_in_history_movie_ids"],
            "eligible_retrievers": ["popularity", "itemknn", "multivae"],
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
            "popularity_tie_break": "movie_id_ascending",
            "implementation": "history-only-popularity-v1",
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
            "eligible_subject_count": len(eligible_subject_movies),
            "training_subject_count": len(cohorts["train"]),
            "validation_subject_count": len(cohorts["validation"]),
            "test_subject_count": len(cohorts["test"]),
            "training_interaction_count": sum(popularity_counts.values()),
            "training_catalog_movie_count": len(training_catalog),
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
                unique_positive_count - sum(map(len, eligible_subject_movies.values()))
            ),
            "subject_count_removed_with_fewer_than_five_positives": (
                len(subject_movies) - len(eligible_subject_movies)
            ),
            "validation": _cohort_exclusions(cohorts["validation"], split_by_subject),
            "test": (
                _cohort_exclusions(cohorts["test"], split_by_subject)
                if evaluate_test
                else {"status": "sealed"}
            ),
        },
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
    )


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


def _history_only_metrics(
    popularity: PopularityRetriever,
    subject_ids: tuple[int, ...],
    queries_by_subject: Mapping[int, HistoryOnlyQuery],
    gold_sets: Mapping[int, tuple[int, ...]],
    training_catalog: frozenset[int],
) -> dict[str, int | float | None]:
    recall_totals = {cutoff: 0.0 for cutoff in METRIC_CUTOFFS}
    ndcg_totals = {cutoff: 0.0 for cutoff in METRIC_CUTOFFS}
    query_count = 0
    covered_queries = 0
    recommended_movies: set[int] = set()
    global_ranking = popularity.ranked_movie_ids()

    for subject_id in subject_ids:
        gold_set = gold_sets[subject_id]
        if not gold_set:
            continue
        query_count += 1
        history = set(queries_by_subject[subject_id].history_movie_ids)
        ranked_movies = tuple(
            islice((movie_id for movie_id in global_ranking if movie_id not in history), 100)
        )
        rank_by_movie = {movie_id: rank for rank, movie_id in enumerate(ranked_movies, 1)}
        covered_queries += any(movie_id in rank_by_movie for movie_id in gold_set)
        recommended_movies.update(ranked_movies)
        for cutoff in METRIC_CUTOFFS:
            hit_ranks = [
                rank_by_movie[movie_id]
                for movie_id in gold_set
                if movie_id in rank_by_movie and rank_by_movie[movie_id] <= cutoff
            ]
            recall_totals[cutoff] += len(hit_ranks) / min(cutoff, len(gold_set))
            discounted_gain = sum(1.0 / math.log2(rank + 1) for rank in hit_ranks)
            ideal_gain = sum(
                1.0 / math.log2(rank + 1) for rank in range(1, min(cutoff, len(gold_set)) + 1)
            )
            ndcg_totals[cutoff] += discounted_gain / ideal_gain

    divisor = query_count or 1
    return {
        "query_count": query_count,
        **{f"Recall@{cutoff}": total / divisor for cutoff, total in recall_totals.items()},
        **{f"NDCG@{cutoff}": total / divisor for cutoff, total in ndcg_totals.items()},
        "QueryRetrievalCoverage@100": covered_queries / divisor,
        "CatalogCoverage@100": (
            len(recommended_movies) / len(training_catalog)
            if training_catalog and query_count
            else None
        ),
    }


def _implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in ("history_only_benchmark.py", "serving.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def write_history_only_benchmark_report(result: HistoryOnlyBenchmarkResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
