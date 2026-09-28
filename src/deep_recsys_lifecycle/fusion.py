from __future__ import annotations

import math
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Literal

from .event_store import DataSnapshot
from .models import Candidate, PositiveInteraction
from .retriever import MAX_CANDIDATE_POOL, validate_candidate_pool_limit

# Importing LightGBM eagerly can initialize a native OpenMP runtime before PyTorch's small
# CPU workloads have finished.  Keep the native dependency lazy at the training/loading seam.
lgb: Any | None = None


FusionQueryMode = Literal["known_user", "history_only"]
FEATURE_SCHEMA_VERSION = 1
FUSION_BANKS: dict[FusionQueryMode, tuple[str, ...]] = {
    "known_user": ("popularity", "itemknn", "multivae", "lightgcn"),
    "history_only": ("popularity", "itemknn", "multivae"),
}
MISSING_EVIDENCE = {"presence": 0.0, "rank": 0.0, "score": 0.0}


def expected_feature_names(retriever_bank: Sequence[str]) -> tuple[str, ...]:
    names: list[str] = []
    for retriever in retriever_bank:
        names.extend(
            (
                f"{retriever}_present",
                f"{retriever}_rank",
                f"{retriever}_score",
            )
        )
    names.extend(
        (
            "hit_count",
            "min_rank",
            "mean_rank",
            "history_length",
            "user_cold",
            "item_popularity",
            "interaction_new",
        )
    )
    return tuple(names)


def _validate_bank(query_mode: FusionQueryMode, retriever_bank: Sequence[str]) -> tuple[str, ...]:
    expected = FUSION_BANKS[query_mode]
    resolved = tuple(retriever_bank)
    if resolved != expected:
        raise ValueError(f"{query_mode} fusion retriever bank must be {expected}, got {resolved}")
    return resolved


@dataclass(frozen=True, slots=True)
class FusionFeatureBuilder:
    """Build one versioned, serving-reconstructible LHF feature vector.

    Missing retriever evidence is represented by ``presence=0``, ``rank=0`` and
    ``score=0.0``.  Presence makes the zero rank/score representation unambiguous.
    ``user_cold`` is one when the deduplicated request history is shorter than five Movies.
    """

    snapshot_fingerprint: str
    retriever_bank: tuple[str, ...]
    item_popularity: Mapping[int, int]
    interaction_new_items: frozenset[int]
    user_cold_history_threshold: int = 5

    @classmethod
    def from_snapshot(
        cls,
        snapshot: DataSnapshot,
        interactions: Sequence[PositiveInteraction],
        *,
        retriever_bank: Sequence[str],
        user_cold_history_threshold: int = 5,
    ) -> FusionFeatureBuilder:
        bank = tuple(retriever_bank)
        if bank not in FUSION_BANKS.values():
            raise ValueError(f"unsupported fusion retriever bank: {bank}")
        if user_cold_history_threshold < 0:
            raise ValueError("user_cold_history_threshold cannot be negative")
        snapshot_event_ids = {event.event_id for event in snapshot.events}
        catalog = {event.movie_id for event in snapshot.events}
        counts: dict[int, int] = {}
        for interaction in interactions:
            if interaction.event_id not in snapshot_event_ids:
                continue
            counts[interaction.movie_id] = counts.get(interaction.movie_id, 0) + 1
        return cls.from_serving(
            snapshot_fingerprint=snapshot.fingerprint,
            retriever_bank=bank,
            catalog=catalog,
            item_popularity=counts,
            user_cold_history_threshold=user_cold_history_threshold,
        )

    @classmethod
    def from_serving(
        cls,
        *,
        snapshot_fingerprint: str,
        retriever_bank: Sequence[str],
        catalog: Collection[int],
        item_popularity: Mapping[int, int],
        user_cold_history_threshold: int = 5,
    ) -> FusionFeatureBuilder:
        """Reconstruct the same feature state from immutable serving payloads."""

        bank = tuple(retriever_bank)
        if bank not in FUSION_BANKS.values():
            raise ValueError(f"unsupported fusion retriever bank: {bank}")
        if user_cold_history_threshold < 0:
            raise ValueError("user_cold_history_threshold cannot be negative")
        catalog_ids = set(catalog)
        counts = {
            movie_id: count
            for movie_id, count in item_popularity.items()
            if movie_id in catalog_ids
        }
        return cls(
            snapshot_fingerprint=snapshot_fingerprint,
            retriever_bank=bank,
            item_popularity=counts,
            interaction_new_items=frozenset(catalog_ids - set(counts)),
            user_cold_history_threshold=user_cold_history_threshold,
        )

    @property
    def schema_version(self) -> int:
        return FEATURE_SCHEMA_VERSION

    @property
    def feature_names(self) -> tuple[str, ...]:
        return expected_feature_names(self.retriever_bank)

    @property
    def feature_schema(self) -> dict[str, object]:
        return {
            "version": self.schema_version,
            "feature_names": list(self.feature_names),
            "missing_evidence": dict(MISSING_EVIDENCE),
            "user_cold_history_threshold": self.user_cold_history_threshold,
        }

    def features_for(
        self,
        movie_id: int,
        pools: Mapping[str, Sequence[Candidate]],
        history: Sequence[int],
    ) -> tuple[float, ...]:
        values: list[float] = []
        ranks: list[float] = []
        observed = set(history)
        for retriever in self.retriever_bank:
            candidate = _candidate_for_movie(pools.get(retriever, ()), movie_id)
            if candidate is None:
                values.extend((0.0, 0.0, 0.0))
                continue
            rank = float(candidate.rank)
            score = float(candidate.score)
            if not math.isfinite(rank) or not math.isfinite(score) or rank < 1:
                raise ValueError("retriever evidence must contain finite positive ranks")
            values.extend((1.0, rank, score))
            ranks.append(rank)

        unique_history_length = len(set(observed))
        hit_count = float(len(ranks))
        min_rank = min(ranks) if ranks else 0.0
        mean_rank = sum(ranks) / len(ranks) if ranks else 0.0
        values.extend(
            (
                hit_count,
                min_rank,
                mean_rank,
                float(unique_history_length),
                float(unique_history_length < self.user_cold_history_threshold),
                float(self.item_popularity.get(movie_id, 0)),
                float(movie_id in self.interaction_new_items),
            )
        )
        result = tuple(values)
        if len(result) != len(self.feature_names) or not all(
            math.isfinite(value) for value in result
        ):
            raise ValueError("fusion feature vector is invalid")
        return result


@dataclass(frozen=True, slots=True)
class FusionTrainingRow:
    query_key: int | str
    movie_id: int
    features: tuple[float, ...]
    label: int


def build_candidate_union(
    pools: Mapping[str, Sequence[Candidate]],
    *,
    history: Sequence[int] = (),
    limit: int = MAX_CANDIDATE_POOL,
) -> tuple[Candidate, ...]:
    """Build a deterministic deduplicated union without ever injecting a Gold Candidate."""

    validate_candidate_pool_limit(limit)
    observed = set(history)
    movie_ids: set[int] = set()
    for candidates in pools.values():
        for candidate in tuple(candidates)[:limit]:
            if candidate.movie_id not in observed:
                movie_ids.add(candidate.movie_id)
    return tuple(
        Candidate(movie_id=movie_id, score=0.0, rank=rank)
        for rank, movie_id in enumerate(sorted(movie_ids), start=1)
    )


def _candidate_for_movie(candidates: Sequence[Candidate], movie_id: int) -> Candidate | None:
    found = [candidate for candidate in candidates if candidate.movie_id == movie_id]
    if not found:
        return None
    return min(found, key=lambda candidate: (candidate.rank, candidate.movie_id))


def _query_key(query: Any) -> int | str:
    key = getattr(query, "pool_key", None)
    if isinstance(key, (int, str)) and not isinstance(key, bool):
        return key
    subject_id = getattr(query, "subject_id", None)
    if not isinstance(subject_id, int) or isinstance(subject_id, bool):
        raise ValueError("evaluation query without identity needs a pool_key")
    return subject_id


def build_lhf_training_rows(
    cohort: Any,
    pools: Mapping[int | str, Mapping[str, Sequence[Candidate]]],
    feature_builder: FusionFeatureBuilder,
    *,
    max_negative_rows_per_query: int | None = None,
    seed: int = 42,
) -> tuple[FusionTrainingRow, ...]:
    """Make labels for every naturally retrieved Gold Set Movie.

    The chronological cohort still supplies ``gold_movie_id``. Paper-style queries
    supply ``gold_movie_ids``; neither form inserts held-out Movies into a pool.
    """

    if max_negative_rows_per_query is not None and (
        isinstance(max_negative_rows_per_query, bool) or max_negative_rows_per_query < 1
    ):
        raise ValueError("max_negative_rows_per_query must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("fusion row sampling seed must be an integer")

    rows: list[FusionTrainingRow] = []
    expected_names = set(feature_builder.retriever_bank)
    for query in cohort:
        key = _query_key(query)
        multi_gold = getattr(query, "gold_movie_ids", None)
        if multi_gold is None:
            gold_ids = frozenset((query.gold_movie_id,))
        else:
            gold_ids = frozenset(multi_gold)
        query_pools = pools.get(key, {})
        unknown_retrievers = set(query_pools) - expected_names
        if unknown_retrievers:
            raise ValueError(f"fusion pools contain ineligible retrievers: {unknown_retrievers}")
        union = build_candidate_union(query_pools, history=query.history)
        selected = union
        if max_negative_rows_per_query is not None:
            positive = tuple(
                candidate for candidate in union if candidate.movie_id in gold_ids
            )
            negatives = sorted(
                (
                    candidate
                    for candidate in union
                    if candidate.movie_id not in gold_ids
                ),
                key=lambda candidate: (
                    sha256(f"{seed}:{key}:{candidate.movie_id}".encode()).digest(),
                    candidate.movie_id,
                ),
            )[:max_negative_rows_per_query]
            selected = tuple(sorted((*positive, *negatives), key=lambda item: item.movie_id))
        for candidate in selected:
            rows.append(
                FusionTrainingRow(
                    query_key=key,
                    movie_id=candidate.movie_id,
                    features=feature_builder.features_for(
                        candidate.movie_id, query_pools, query.history
                    ),
                    label=int(candidate.movie_id in gold_ids),
                )
            )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class LearnedHybridFusion:
    """A CPU-loadable LightGBM classifier plus an explicit fallback state."""

    query_mode: FusionQueryMode
    retriever_bank: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_schema: Mapping[str, object]
    training_status: str
    training_metadata: Mapping[str, object]
    model_string: str | None = None
    fallback_strategy: str = "rrf"
    _booster: Any = field(default=None, repr=False, compare=False)

    @property
    def algorithm(self) -> str:
        return "lightgbm_binary_classifier"

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "query_mode": self.query_mode,
            "retriever_bank": list(self.retriever_bank),
            "feature_schema": dict(self.feature_schema),
            "training_status": self.training_status,
            "fallback_strategy": self.fallback_strategy,
        }

    @property
    def is_trained(self) -> bool:
        return self.training_status == "trained" and self._booster is not None

    def score(self, features: Sequence[float]) -> float:
        if not self.is_trained:
            raise ValueError("untrained fallback has no learned score")
        if len(features) != len(self.feature_names):
            raise ValueError("fusion features do not match the classifier schema")
        score_value = float(self._booster.predict([list(features)], num_threads=1)[0])
        if not math.isfinite(score_value):
            raise ValueError("LightGBM produced a non-finite fusion score")
        return score_value

    def to_dict(self) -> dict[str, object]:
        return {
            "payload_schema_version": 1,
            "classifier": self.algorithm,
            "query_mode": self.query_mode,
            "retriever_bank": list(self.retriever_bank),
            "feature_schema": dict(self.feature_schema),
            "feature_names": list(self.feature_names),
            "training_status": self.training_status,
            "training_metadata": dict(self.training_metadata),
            "fallback_strategy": self.fallback_strategy,
            "model_string": self.model_string,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> LearnedHybridFusion:
        if (
            value.get("payload_schema_version") != 1
            or value.get("classifier") != "lightgbm_binary_classifier"
        ):
            raise ValueError("LHF payload has an unsupported schema")
        raw_mode = value.get("query_mode")
        if raw_mode not in FUSION_BANKS:
            raise ValueError("LHF payload has an unsupported query mode")
        query_mode = raw_mode
        raw_bank = value.get("retriever_bank")
        raw_schema = value.get("feature_schema")
        raw_names = value.get("feature_names")
        raw_metadata = value.get("training_metadata")
        status = value.get("training_status")
        fallback = value.get("fallback_strategy")
        if (
            not isinstance(raw_bank, list)
            or not isinstance(raw_schema, Mapping)
            or not isinstance(raw_names, list)
            or not isinstance(raw_metadata, Mapping)
            or status not in {"trained", "untrained fallback"}
            or fallback != "rrf"
        ):
            raise ValueError("LHF payload has invalid schema or training metadata")
        bank = _validate_bank(query_mode, tuple(raw_bank))
        names = tuple(_strict_string(item, "LHF feature name") for item in raw_names)
        expected_names = expected_feature_names(bank)
        if names != expected_names:
            raise ValueError("LHF payload feature order is incompatible with its retriever bank")
        if (
            raw_schema.get("version") != FEATURE_SCHEMA_VERSION
            or tuple(raw_schema.get("feature_names", ())) != expected_names
            or raw_schema.get("missing_evidence") != MISSING_EVIDENCE
            or not isinstance(raw_schema.get("user_cold_history_threshold"), int)
            or isinstance(raw_schema.get("user_cold_history_threshold"), bool)
        ):
            raise ValueError("LHF payload feature schema is incompatible")
        model_string = value.get("model_string")
        booster = None
        if status == "trained":
            if not isinstance(model_string, str):
                raise ValueError("trained LHF payload is missing its LightGBM model")
            resolved_model_string: str | None = model_string
            lightgbm = _load_lightgbm()
            if lightgbm is None:
                raise ValueError("LightGBM is unavailable for a trained LHF payload")
            try:
                booster = lightgbm.Booster(model_str=model_string)
            except Exception as error:
                raise ValueError("trained LHF model is unreadable") from error
        else:
            if model_string is not None:
                raise ValueError("untrained fallback must not contain a model")
            resolved_model_string = None
        return cls(
            query_mode=query_mode,
            retriever_bank=bank,
            feature_names=names,
            feature_schema=dict(raw_schema),
            training_status=status,
            training_metadata=dict(raw_metadata),
            model_string=resolved_model_string,
            fallback_strategy=fallback,
            _booster=booster,
        )


LHFClassifier = LearnedHybridFusion


def train_lhf_classifier(
    *,
    query_mode: FusionQueryMode,
    retriever_bank: Sequence[str],
    feature_names: Sequence[str],
    rows: Sequence[FusionTrainingRow],
    source_snapshot_fingerprint: str,
    seed: int = 42,
    validation_event_ids: Sequence[str] = (),
    validation_sources: Sequence[Mapping[str, object]] = (),
    min_positive_rows: int = 2,
    min_negative_rows: int = 2,
) -> LearnedHybridFusion:
    bank = _validate_bank(query_mode, retriever_bank)
    names = tuple(feature_names)
    if names != expected_feature_names(bank):
        raise ValueError("feature order does not match the query-mode retriever bank")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("LHF seed must be an integer")
    positive_count = sum(row.label == 1 for row in rows)
    negative_count = sum(row.label == 0 for row in rows)
    metadata: dict[str, object] = {
        "algorithm": "lightgbm.Booster(binary_classifier)",
        "seed": seed,
        "device": "cpu",
        "source_snapshot_fingerprint": source_snapshot_fingerprint,
        "training_snapshot_fingerprint": source_snapshot_fingerprint,
        "training_boundary": "inner_validation_before_future_window",
        "future_window_used_for_training": False,
        "validation_event_ids": list(validation_event_ids),
        "validation_sources": [dict(source) for source in validation_sources],
        "row_count": len(rows),
        "positive_row_count": positive_count,
        "negative_row_count": negative_count,
        "hyperparameters": _lightgbm_hyperparameters(seed),
    }
    schema = {
        "version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(names),
        "missing_evidence": dict(MISSING_EVIDENCE),
        "user_cold_history_threshold": 5,
    }
    reason: str | None = None
    if positive_count < min_positive_rows:
        reason = "insufficient positive validation rows"
    elif negative_count < min_negative_rows:
        reason = "insufficient negative validation rows"
    lightgbm_module = _load_lightgbm()
    if reason is None and lightgbm_module is None:
        reason = "LightGBM is unavailable"
    if reason is not None:
        metadata["fallback_reason"] = reason
        return LearnedHybridFusion(
            query_mode=query_mode,
            retriever_bank=bank,
            feature_names=names,
            feature_schema=schema,
            training_status="untrained fallback",
            training_metadata=metadata,
        )

    if lightgbm_module is None:  # pragma: no cover - reason above handles this branch
        raise RuntimeError("LightGBM is unavailable")
    try:
        import numpy as np

        dataset = lightgbm_module.Dataset(
            np.asarray([row.features for row in rows], dtype=float),
            label=np.asarray([row.label for row in rows], dtype=int),
            feature_name=list(names),
            free_raw_data=True,
        )
        booster = lightgbm_module.train(
            _lightgbm_hyperparameters(seed),
            dataset,
            num_boost_round=32,
        )
        model_string = booster.model_to_string()
    except Exception as error:
        metadata["fallback_reason"] = f"LightGBM training failed: {type(error).__name__}: {error}"
        return LearnedHybridFusion(
            query_mode=query_mode,
            retriever_bank=bank,
            feature_names=names,
            feature_schema=schema,
            training_status="untrained fallback",
            training_metadata=metadata,
        )
    metadata["fallback_reason"] = None
    return LearnedHybridFusion(
        query_mode=query_mode,
        retriever_bank=bank,
        feature_names=names,
        feature_schema=schema,
        training_status="trained",
        training_metadata=metadata,
        model_string=model_string,
        _booster=booster,
    )


def untrained_fusion_classifier(
    query_mode: FusionQueryMode,
    *,
    source_snapshot_fingerprint: str,
    reason: str = "no valid validation labels",
) -> LearnedHybridFusion:
    """Create an explicitly labelled fallback without fabricating training examples."""

    bank = FUSION_BANKS[query_mode]
    names = expected_feature_names(bank)
    classifier = train_lhf_classifier(
        query_mode=query_mode,
        retriever_bank=bank,
        feature_names=names,
        rows=(),
        source_snapshot_fingerprint=source_snapshot_fingerprint,
        validation_event_ids=(),
        min_positive_rows=1,
        min_negative_rows=1,
    )
    return replace(
        classifier,
        training_metadata={
            **dict(classifier.training_metadata),
            "fallback_reason": reason,
        },
    )


def _lightgbm_hyperparameters(seed: int) -> dict[str, object]:
    return {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 1,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "seed": seed,
        "num_threads": 1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def _load_lightgbm() -> Any | None:
    global lgb
    if lgb is not None:
        return lgb
    # The Apple wheels for PyTorch and LightGBM can ship separate libomp copies.  The
    # lifecycle is single-process and single-threaded; allow that bounded coexistence so the
    # native classifier can run after CPU neural fitting instead of aborting the process.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import lightgbm
    except ImportError:  # pragma: no cover - exercised only by an incomplete environment
        return None
    lgb = lightgbm
    return lgb


def rank_lhf_union(
    classifier: LearnedHybridFusion,
    feature_builder: FusionFeatureBuilder,
    pools: Mapping[str, Sequence[Candidate]],
    history: Sequence[int],
    *,
    top_n: int = MAX_CANDIDATE_POOL,
) -> tuple[Candidate, ...]:
    """Score one retriever union and apply the required score/Movie-ID ordering."""

    validate_candidate_pool_limit(top_n)
    if tuple(feature_builder.retriever_bank) != classifier.retriever_bank:
        raise ValueError("fusion feature builder and classifier banks do not match")
    union = build_candidate_union(pools, history=history)
    if not classifier.is_trained:
        fallback_scores: dict[int, float] = {}
        for retriever in classifier.retriever_bank:
            for rank, candidate in enumerate(
                tuple(pools.get(retriever, ()))[:MAX_CANDIDATE_POOL], start=1
            ):
                if candidate.movie_id not in set(history):
                    fallback_scores[candidate.movie_id] = fallback_scores.get(
                        candidate.movie_id, 0.0
                    ) + 1.0 / (60 + rank)
        fallback_ordered = sorted(
            union,
            key=lambda item: (-fallback_scores.get(item.movie_id, 0.0), item.movie_id),
        )
        return tuple(
            Candidate(
                movie_id=item.movie_id,
                score=fallback_scores.get(item.movie_id, 0.0),
                rank=rank,
            )
            for rank, item in enumerate(fallback_ordered[:top_n], start=1)
        )

    scored = [
        (
            candidate.movie_id,
            classifier.score(feature_builder.features_for(candidate.movie_id, pools, history)),
        )
        for candidate in union
    ]
    scored_ordered: list[tuple[int, float]] = sorted(scored, key=lambda item: (-item[1], item[0]))[
        :top_n
    ]
    return tuple(
        Candidate(movie_id=movie_id, score=score, rank=rank)
        for rank, (movie_id, score) in enumerate(scored_ordered, start=1)
    )


def _strict_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value
