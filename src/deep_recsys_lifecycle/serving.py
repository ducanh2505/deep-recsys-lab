from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from .models import Candidate
from .retriever import MAX_CANDIDATE_POOL, validate_candidate_pool_limit


@dataclass(frozen=True, slots=True)
class PopularityRetriever:
    """CPU-only Popularity state used by the serving runtime.

    This module intentionally has no Event Store, dataset, or fitting imports.  The
    training module can produce this state, while the API only needs this runtime
    representation to answer queries from an immutable artifact.
    """

    catalog: tuple[int, ...]
    counts: Mapping[int, int]
    subject_histories: Mapping[int, tuple[int, ...]]

    @property
    def name(self) -> str:
        return "popularity"

    def ranked_movie_ids(self, excluded_movie_ids: Collection[int] = ()) -> tuple[int, ...]:
        excluded = set(excluded_movie_ids)
        return tuple(
            movie_id
            for movie_id in sorted(self.catalog, key=lambda item: (-self.counts.get(item, 0), item))
            if movie_id not in excluded
        )

    def recommend(self, excluded_movie_ids: Collection[int], top_n: int) -> tuple[Candidate, ...]:
        if not 1 <= top_n <= 100:
            raise ValueError("top_n must be between 1 and 100")
        return self.candidate_pool(excluded_movie_ids, limit=top_n)

    def candidate_pool(
        self,
        excluded_movie_ids: Collection[int],
        limit: int = MAX_CANDIDATE_POOL,
    ) -> tuple[Candidate, ...]:
        validate_candidate_pool_limit(limit)
        movie_ids = self.ranked_movie_ids(excluded_movie_ids)[:limit]
        return tuple(
            Candidate(movie_id=movie_id, score=self.counts.get(movie_id, 0), rank=rank)
            for rank, movie_id in enumerate(movie_ids, start=1)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog": list(self.catalog),
            "counts": {str(movie_id): count for movie_id, count in self.counts.items()},
            "subject_histories": {
                str(subject_id): list(movie_ids)
                for subject_id, movie_ids in self.subject_histories.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PopularityRetriever:
        raw_catalog = value.get("catalog")
        raw_counts = value.get("counts")
        raw_histories = value.get("subject_histories")
        if (
            not isinstance(raw_catalog, list)
            or not isinstance(raw_counts, Mapping)
            or not isinstance(raw_histories, Mapping)
        ):
            raise ValueError("Popularity payload has invalid catalog, count, or history mappings")

        catalog = tuple(_strict_int(item, "catalog movie id") for item in raw_catalog)
        counts: dict[int, int] = {}
        for movie_id, count in raw_counts.items():
            if not isinstance(movie_id, str):
                raise ValueError("Popularity payload count keys must be strings")
            counts[_parse_int(movie_id, "count movie id")] = _strict_int(count, "movie count")

        subject_histories: dict[int, tuple[int, ...]] = {}
        for subject_id, movie_ids in raw_histories.items():
            if not isinstance(subject_id, str) or not isinstance(movie_ids, list):
                raise ValueError("Popularity payload has an invalid Subject history")
            subject_histories[_parse_int(subject_id, "history Subject id")] = tuple(
                _strict_int(movie_id, "history movie id") for movie_id in movie_ids
            )

        if len(set(catalog)) != len(catalog):
            raise ValueError("Popularity payload catalog contains duplicate Movies")
        if any(count < 0 for count in counts.values()):
            raise ValueError("Popularity payload counts cannot be negative")

        return cls(
            catalog=catalog,
            counts=counts,
            subject_histories=subject_histories,
        )


@dataclass(frozen=True, slots=True)
class MultVAERetriever:
    """CPU inference state for a snapshot-fitted Mult-VAE Candidate Retriever.

    The lifecycle trains the neural network, but the serving side deliberately keeps only
    JSON-safe weights and performs inference with the small dense network below.  This keeps
    the runtime loader independent from fitting, Kafka, and the Event Store.
    """

    catalog: tuple[int, ...]
    subject_histories: Mapping[int, tuple[int, ...]]
    encoder_weight: tuple[tuple[float, ...], ...]
    encoder_bias: tuple[float, ...]
    mean_weight: tuple[tuple[float, ...], ...]
    mean_bias: tuple[float, ...]
    decoder_weight: tuple[tuple[float, ...], ...]
    decoder_bias: tuple[float, ...]
    output_weight: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]
    configuration: Mapping[str, Any]
    training_metadata: Mapping[str, Any]

    @property
    def name(self) -> str:
        return "multivae"

    @property
    def index_by_movie_id(self) -> dict[int, int]:
        return {movie_id: index for index, movie_id in enumerate(self.catalog)}

    def profile_for_history(self, history: Collection[int]) -> tuple[int, ...]:
        """Build the one shared binary profile semantics for every query mode.

        IDs not in the snapshot catalog are ignored by the profile but remain observed for
        candidate exclusion.  Repeated IDs therefore have exactly the same representation as
        one occurrence.
        """

        observed = set(history)
        return tuple(int(movie_id in observed) for movie_id in self.catalog)

    def profile_for_subject(self, subject_id: int) -> tuple[int, ...]:
        history = self.subject_histories.get(subject_id)
        if history is None:
            raise KeyError(subject_id)
        return self.profile_for_history(history)

    def ranked_movie_ids(self, history: Collection[int] = ()) -> tuple[int, ...]:
        return tuple(candidate.movie_id for candidate in self.candidate_pool(history))

    def candidate_pool(
        self,
        history: Collection[int],
        limit: int = MAX_CANDIDATE_POOL,
    ) -> tuple[Candidate, ...]:
        validate_candidate_pool_limit(limit)
        observed = set(history)
        profile = self.profile_for_history(history)
        if not any(profile):
            # A history with no catalog IDs has no learned personalization signal.  Keep the
            # behavior explicit and deterministic; public Empty-History requests never route
            # here and continue to use Popularity.
            ordered = ((movie_id, 0.0) for movie_id in self.catalog if movie_id not in observed)
        else:
            scores = self._scores(profile)
            ordered = (
                (movie_id, score)
                for movie_id, score in zip(self.catalog, scores, strict=True)
                if movie_id not in observed
            )
        ranked = sorted(ordered, key=lambda item: (-item[1], item[0]))[:limit]
        return tuple(
            Candidate(movie_id=movie_id, score=score, rank=rank)
            for rank, (movie_id, score) in enumerate(ranked, start=1)
        )

    def recommend(self, history: Collection[int], top_n: int) -> tuple[Candidate, ...]:
        if not 1 <= top_n <= 100:
            raise ValueError("top_n must be between 1 and 100")
        return self.candidate_pool(history, limit=top_n)

    def _scores(self, profile: tuple[int, ...]) -> tuple[float, ...]:
        norm = math.sqrt(sum(float(value * value) for value in profile))
        normalized = tuple(value / norm for value in profile)
        hidden = tuple(
            math.tanh(
                sum(weight * value for weight, value in zip(row, normalized, strict=True)) + bias
            )
            for row, bias in zip(self.encoder_weight, self.encoder_bias, strict=True)
        )
        latent = tuple(
            sum(weight * value for weight, value in zip(row, hidden, strict=True)) + bias
            for row, bias in zip(self.mean_weight, self.mean_bias, strict=True)
        )
        decoded = tuple(
            math.tanh(sum(weight * value for weight, value in zip(row, latent, strict=True)) + bias)
            for row, bias in zip(self.decoder_weight, self.decoder_bias, strict=True)
        )
        scores = tuple(
            sum(weight * value for weight, value in zip(row, decoded, strict=True)) + bias
            for row, bias in zip(self.output_weight, self.output_bias, strict=True)
        )
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Mult-VAE produced a non-finite score")
        return scores

    def to_dict(self) -> dict[str, object]:
        return {
            "payload_schema_version": 1,
            "retriever": "multivae",
            "catalog": list(self.catalog),
            "index_by_movie_id": {
                str(movie_id): index for index, movie_id in enumerate(self.catalog)
            },
            "subject_histories": {
                str(subject_id): list(movie_ids)
                for subject_id, movie_ids in self.subject_histories.items()
            },
            "configuration": dict(self.configuration),
            "training_metadata": dict(self.training_metadata),
            "weights": {
                "encoder_weight": [list(row) for row in self.encoder_weight],
                "encoder_bias": list(self.encoder_bias),
                "mean_weight": [list(row) for row in self.mean_weight],
                "mean_bias": list(self.mean_bias),
                "decoder_weight": [list(row) for row in self.decoder_weight],
                "decoder_bias": list(self.decoder_bias),
                "output_weight": [list(row) for row in self.output_weight],
                "output_bias": list(self.output_bias),
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MultVAERetriever:
        if value.get("payload_schema_version") != 1 or value.get("retriever") != "multivae":
            raise ValueError("Mult-VAE payload has an unsupported schema or retriever")
        raw_catalog = value.get("catalog")
        raw_index = value.get("index_by_movie_id")
        raw_histories = value.get("subject_histories")
        raw_configuration = value.get("configuration")
        raw_training = value.get("training_metadata")
        raw_weights = value.get("weights")
        if (
            not isinstance(raw_catalog, list)
            or not isinstance(raw_index, Mapping)
            or not isinstance(raw_histories, Mapping)
            or not isinstance(raw_configuration, Mapping)
            or not isinstance(raw_training, Mapping)
            or not isinstance(raw_weights, Mapping)
        ):
            raise ValueError("Mult-VAE payload is missing catalog, mapping, history, or weights")

        catalog = tuple(_strict_int(item, "Mult-VAE catalog movie id") for item in raw_catalog)
        if len(set(catalog)) != len(catalog):
            raise ValueError("Mult-VAE payload catalog contains duplicate Movies")
        expected_index = {str(movie_id): index for index, movie_id in enumerate(catalog)}
        if dict(raw_index) != expected_index:
            raise ValueError("Mult-VAE payload catalog/index mapping is inconsistent")

        subject_histories: dict[int, tuple[int, ...]] = {}
        catalog_ids = set(catalog)
        for subject_id, movie_ids in raw_histories.items():
            if not isinstance(subject_id, str) or not isinstance(movie_ids, list):
                raise ValueError("Mult-VAE payload has an invalid Subject history")
            parsed_history = tuple(
                _strict_int(item, "Mult-VAE history movie id") for item in movie_ids
            )
            if (
                len(set(parsed_history)) != len(parsed_history)
                or not set(parsed_history) <= catalog_ids
            ):
                raise ValueError("Mult-VAE Subject history is not a deduplicated catalog profile")
            subject_histories[_parse_int(subject_id, "Mult-VAE history Subject id")] = (
                parsed_history
            )

        encoder_weight = _matrix(raw_weights, "encoder_weight")
        encoder_bias = _vector(raw_weights, "encoder_bias")
        mean_weight = _matrix(raw_weights, "mean_weight")
        mean_bias = _vector(raw_weights, "mean_bias")
        decoder_weight = _matrix(raw_weights, "decoder_weight")
        decoder_bias = _vector(raw_weights, "decoder_bias")
        output_weight = _matrix(raw_weights, "output_weight")
        output_bias = _vector(raw_weights, "output_bias")
        hidden_dim = len(encoder_weight)
        latent_dim = len(mean_weight)
        if hidden_dim < 1 or latent_dim < 1 or len(catalog) < 1:
            raise ValueError("Mult-VAE payload dimensions must be positive")
        _expect_matrix_shape(encoder_weight, hidden_dim, len(catalog), "encoder_weight")
        _expect_vector_shape(encoder_bias, hidden_dim, "encoder_bias")
        _expect_matrix_shape(mean_weight, latent_dim, hidden_dim, "mean_weight")
        _expect_vector_shape(mean_bias, latent_dim, "mean_bias")
        _expect_matrix_shape(decoder_weight, hidden_dim, latent_dim, "decoder_weight")
        _expect_vector_shape(decoder_bias, hidden_dim, "decoder_bias")
        _expect_matrix_shape(output_weight, len(catalog), hidden_dim, "output_weight")
        _expect_vector_shape(output_bias, len(catalog), "output_bias")

        return cls(
            catalog=catalog,
            subject_histories=subject_histories,
            encoder_weight=encoder_weight,
            encoder_bias=encoder_bias,
            mean_weight=mean_weight,
            mean_bias=mean_bias,
            decoder_weight=decoder_weight,
            decoder_bias=decoder_bias,
            output_weight=output_weight,
            output_bias=output_bias,
            configuration=dict(raw_configuration),
            training_metadata=dict(raw_training),
        )


def _strict_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Popularity payload {label} must be an integer")
    return value


def _parse_int(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Popularity payload {label} must be an integer") from error


def _vector(value: Mapping[str, object], key: str) -> tuple[float, ...]:
    raw = value.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"Mult-VAE payload weight {key!r} must be a list")
    return tuple(_finite_float(item, f"Mult-VAE weight {key}") for item in raw)


def _matrix(value: Mapping[str, object], key: str) -> tuple[tuple[float, ...], ...]:
    raw = value.get(key)
    if not isinstance(raw, list) or any(not isinstance(row, list) for row in raw):
        raise ValueError(f"Mult-VAE payload weight {key!r} must be a matrix")
    return tuple(
        tuple(_finite_float(item, f"Mult-VAE weight {key}") for item in row) for row in raw
    )


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must contain numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must contain finite numbers")
    return result


def _expect_vector_shape(value: tuple[float, ...], length: int, label: str) -> None:
    if len(value) != length:
        raise ValueError(f"Mult-VAE payload {label} has the wrong shape")


def _expect_matrix_shape(
    value: tuple[tuple[float, ...], ...], rows: int, columns: int, label: str
) -> None:
    if len(value) != rows or any(len(row) != columns for row in value):
        raise ValueError(f"Mult-VAE payload {label} has the wrong shape")
