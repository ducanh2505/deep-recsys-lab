from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .artifact import ServingArtifact
from .fusion import FusionFeatureBuilder, rank_lhf_union
from .models import Candidate

QueryMode = Literal["known_user", "history_only", "empty_history"]


@dataclass(frozen=True, slots=True)
class KnownUserQuery:
    subject_id: int

    @property
    def mode(self) -> QueryMode:
        return "known_user"


@dataclass(frozen=True, slots=True)
class HistoryOnlyQuery:
    movie_ids: tuple[int, ...]

    @property
    def mode(self) -> QueryMode:
        return "history_only"


@dataclass(frozen=True, slots=True)
class EmptyHistoryQuery:
    @property
    def mode(self) -> QueryMode:
        return "empty_history"


Query = KnownUserQuery | HistoryOnlyQuery | EmptyHistoryQuery


class UnknownSubjectError(LookupError):
    def __init__(self, subject_id: int) -> None:
        super().__init__(f"unknown Subject: {subject_id}")
        self.subject_id = subject_id


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    query_mode: QueryMode
    candidates: tuple[Candidate, ...]
    retriever: str
    artifact_fingerprint: str
    artifact_id: str
    bundle_fingerprint: str
    fusion_training_status: str | None = None

    def to_dict(self) -> dict[str, object]:
        provenance: dict[str, object] = {
            "artifact_id": self.artifact_id,
            "artifact_fingerprint": self.bundle_fingerprint,
            "data_snapshot_fingerprint": self.artifact_fingerprint,
        }
        if self.fusion_training_status is not None:
            provenance["fusion_training_status"] = self.fusion_training_status
        return {
            "query_mode": self.query_mode,
            "retriever": self.retriever,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "provenance": provenance,
        }


class RecommendationService:
    """Route public API queries and expose loaded retriever smoke seams."""

    def __init__(self, artifact: ServingArtifact) -> None:
        self.artifact = artifact

    def recommend(self, query: Query, top_n: int = 10) -> RecommendationResult:
        _validate_top_n(top_n)
        if isinstance(query, KnownUserQuery):
            history = self.artifact.model.subject_histories.get(query.subject_id)
            if history is None:
                raise UnknownSubjectError(query.subject_id)
            return self._recommend_fused(query, history, top_n)
        elif isinstance(query, HistoryOnlyQuery):
            return self._recommend_fused(query, query.movie_ids, top_n)
        else:
            candidates = self.artifact.model.recommend((), top_n)
            return self._result(query, candidates, "popularity")

    def _recommend_fused(
        self, query: KnownUserQuery | HistoryOnlyQuery, history: tuple[int, ...], top_n: int
    ) -> RecommendationResult:
        fusion = (
            self.artifact.known_user_fusion
            if isinstance(query, KnownUserQuery)
            else self.artifact.history_only_fusion
        )
        if (
            fusion is None
            or self.artifact.itemknn is None
            or self.artifact.multivae is None
            or (isinstance(query, KnownUserQuery) and self.artifact.lightgcn is None)
        ):
            return self._result(query, self.artifact.model.recommend(history, top_n), "popularity")
        pools = {
            "popularity": self.artifact.model.candidate_pool(history),
            "itemknn": self.artifact.itemknn.candidate_pool(history),
            "multivae": self.artifact.multivae.candidate_pool(history),
        }
        if isinstance(query, KnownUserQuery):
            if self.artifact.lightgcn is None:  # pragma: no cover - guarded above
                raise ValueError("Known-User fusion requires a LightGCN payload")
            pools["lightgcn"] = self.artifact.lightgcn.candidate_pool_for_subject(
                query.subject_id, history
            )
        builder = FusionFeatureBuilder.from_serving(
            snapshot_fingerprint=str(self.artifact.manifest["data_snapshot_fingerprint"]),
            retriever_bank=fusion.retriever_bank,
            catalog=self.artifact.model.catalog,
            item_popularity=self.artifact.model.counts,
            user_cold_history_threshold=_feature_threshold(fusion.feature_schema),
        )
        candidates = rank_lhf_union(fusion, builder, pools, history, top_n=200)[:top_n]
        return self._result(
            query,
            candidates,
            "lhf",
            fusion_training_status=fusion.training_status,
        )

    def _result(
        self,
        query: Query,
        candidates: tuple[Candidate, ...],
        retriever: str,
        *,
        fusion_training_status: str | None = None,
    ) -> RecommendationResult:
        return RecommendationResult(
            query_mode=query.mode,
            candidates=candidates,
            retriever=retriever,
            artifact_fingerprint=str(self.artifact.manifest["data_snapshot_fingerprint"]),
            artifact_id=self.artifact.artifact_id,
            bundle_fingerprint=self.artifact.artifact_fingerprint,
            fusion_training_status=fusion_training_status,
        )

    def recommend_with_retriever(
        self,
        query: Query,
        retriever: str,
        top_n: int = 10,
    ) -> RecommendationResult:
        """Run a loaded Candidate Retriever without changing the public API fallback.

        Empty-History intentionally has no Mult-VAE route.  The public ``recommend`` method
        uses the query-mode LHF classifier for Known-User and History-Only requests.
        """

        _validate_top_n(top_n)
        if retriever == "popularity":
            if isinstance(query, KnownUserQuery):
                popularity_history = self._known_history(query.subject_id)
            elif isinstance(query, HistoryOnlyQuery):
                popularity_history = query.movie_ids
            else:
                popularity_history = ()
            return self._result(
                query, self.artifact.model.recommend(popularity_history, top_n), "popularity"
            )
        if retriever == "itemknn":
            if self.artifact.itemknn is None:
                raise ValueError("Serving Artifact has no ItemKNN payload")
            if isinstance(query, KnownUserQuery):
                itemknn_history = self._known_history(query.subject_id)
            elif isinstance(query, HistoryOnlyQuery):
                itemknn_history = query.movie_ids
            else:
                raise ValueError("ItemKNN does not serve Empty-History queries")
            return self._result(
                query, self.artifact.itemknn.recommend(itemknn_history, top_n), "itemknn"
            )
        if retriever == "lightgcn":
            if not isinstance(query, KnownUserQuery):
                raise ValueError("LightGCN supports Known-User queries only")
            if self.artifact.lightgcn is None:
                raise ValueError("Serving Artifact has no LightGCN payload")
            try:
                lightgcn_history = self.artifact.lightgcn.subject_histories[query.subject_id]
            except KeyError as error:
                raise UnknownSubjectError(query.subject_id) from error
            return self._result(
                query,
                self.artifact.lightgcn.recommend_for_subject(
                    query.subject_id, lightgcn_history, top_n
                ),
                "lightgcn",
            )
        if retriever != "multivae":
            raise ValueError(f"unsupported serving retriever: {retriever}")
        if isinstance(query, EmptyHistoryQuery):
            raise ValueError("Mult-VAE does not serve Empty-History queries")
        multivae = self.artifact.multivae
        if multivae is None:
            raise ValueError("Serving Artifact has no Mult-VAE payload")
        if isinstance(query, KnownUserQuery):
            multivae_history = self._known_history(query.subject_id)
        else:
            multivae_history = query.movie_ids
        return self._result(query, multivae.recommend(multivae_history, top_n), "multivae")

    def _known_history(self, subject_id: int) -> tuple[int, ...]:
        history = self.artifact.model.subject_histories.get(subject_id)
        if history is None:
            raise UnknownSubjectError(subject_id)
        return history


def _validate_top_n(top_n: int) -> None:
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 100:
        raise ValueError("top_n must be between 1 and 100")


def _feature_threshold(feature_schema: Mapping[str, object]) -> int:
    raw_threshold = feature_schema.get("user_cold_history_threshold", 5)
    if isinstance(raw_threshold, int) and not isinstance(raw_threshold, bool):
        return raw_threshold
    return 5
