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

    def to_dict(self) -> dict[str, object]:
        return {
            "query_mode": self.query_mode,
            "retriever": self.retriever,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "provenance": {"data_snapshot_fingerprint": self.artifact_fingerprint},
        }


class RecommendationService:
    """Route the three public Query modes to the Popularity artifact."""

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
        )
