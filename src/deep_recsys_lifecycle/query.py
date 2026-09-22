from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .artifact import ServingArtifact
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

    def to_dict(self) -> dict[str, object]:
        return {
            "query_mode": self.query_mode,
            "retriever": self.retriever,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "provenance": {
                "artifact_id": self.artifact_id,
                "artifact_fingerprint": self.bundle_fingerprint,
                "data_snapshot_fingerprint": self.artifact_fingerprint,
            },
        }


class RecommendationService:
    """Route public API queries and expose loaded retriever smoke seams."""

    def __init__(self, artifact: ServingArtifact) -> None:
        self.artifact = artifact

    def recommend(self, query: Query, top_n: int = 10) -> RecommendationResult:
        if isinstance(query, KnownUserQuery):
            history = self.artifact.model.subject_histories.get(query.subject_id)
            if history is None:
                raise UnknownSubjectError(query.subject_id)
            excluded = history
        elif isinstance(query, HistoryOnlyQuery):
            excluded = query.movie_ids
        else:
            excluded = ()

        return RecommendationResult(
            query_mode=query.mode,
            candidates=self.artifact.model.recommend(excluded, top_n),
            retriever="popularity",
            artifact_fingerprint=str(self.artifact.manifest["data_snapshot_fingerprint"]),
            artifact_id=self.artifact.artifact_id,
            bundle_fingerprint=self.artifact.artifact_fingerprint,
        )

    def recommend_with_retriever(
        self,
        query: Query,
        retriever: str,
        top_n: int = 10,
    ) -> RecommendationResult:
        """Run a loaded Candidate Retriever without changing the public API fallback.

        Empty-History intentionally has no Mult-VAE route.  The public ``recommend`` method
        remains Popularity for every query until learned fusion is implemented.
        """

        if retriever == "popularity":
            return self.recommend(query, top_n=top_n)
        if retriever == "lightgcn":
            if not isinstance(query, KnownUserQuery):
                raise ValueError("LightGCN supports Known-User queries only")
            if self.artifact.lightgcn is None:
                raise ValueError("Serving Artifact has no LightGCN payload")
            try:
                lightgcn_history = self.artifact.lightgcn.subject_histories[query.subject_id]
            except KeyError as error:
                raise UnknownSubjectError(query.subject_id) from error
            return RecommendationResult(
                query_mode=query.mode,
                candidates=self.artifact.lightgcn.recommend_for_subject(
                    query.subject_id, lightgcn_history, top_n
                ),
                retriever="lightgcn",
                artifact_fingerprint=str(self.artifact.manifest["data_snapshot_fingerprint"]),
                artifact_id=self.artifact.artifact_id,
                bundle_fingerprint=self.artifact.artifact_fingerprint,
            )
        if retriever != "multivae":
            raise ValueError(f"unsupported serving retriever: {retriever}")
        if isinstance(query, EmptyHistoryQuery):
            raise ValueError("Mult-VAE does not serve Empty-History queries")
        multivae = self.artifact.multivae
        if multivae is None:
            raise ValueError("Serving Artifact has no Mult-VAE payload")
        history: tuple[int, ...]
        if isinstance(query, KnownUserQuery):
            known_history = multivae.subject_histories.get(query.subject_id)
            if known_history is None:
                raise UnknownSubjectError(query.subject_id)
            history = known_history
        else:
            history = query.movie_ids
        return RecommendationResult(
            query_mode=query.mode,
            candidates=multivae.recommend(history, top_n),
            retriever="multivae",
            artifact_fingerprint=str(self.artifact.manifest["data_snapshot_fingerprint"]),
            artifact_id=self.artifact.artifact_id,
            bundle_fingerprint=self.artifact.artifact_fingerprint,
        )
