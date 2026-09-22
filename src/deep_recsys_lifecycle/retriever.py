from __future__ import annotations

from collections.abc import Collection
from typing import Protocol, cast

from .models import Candidate

MAX_CANDIDATE_POOL = 200


class CandidateRetriever(Protocol):
    """The small contract shared by retrievers in the evaluation pipeline."""

    @property
    def name(self) -> str: ...

    def candidate_pool(
        self, history: Collection[int], limit: int = MAX_CANDIDATE_POOL
    ) -> tuple[Candidate, ...]: ...


class SubjectAwareCandidateRetriever(Protocol):
    """Optional seam for retrievers that require a persisted Subject identity."""

    def candidate_pool_for_subject(
        self,
        subject_id: int,
        history: Collection[int],
        limit: int = MAX_CANDIDATE_POOL,
    ) -> tuple[Candidate, ...]: ...


def candidate_pool_for_query(
    retriever: CandidateRetriever,
    subject_id: int | None,
    history: Collection[int],
    limit: int = MAX_CANDIDATE_POOL,
) -> tuple[Candidate, ...]:
    """Use Subject identity only when a retriever explicitly declares that seam."""

    subject_method = getattr(retriever, "candidate_pool_for_subject", None)
    if subject_method is not None:
        if subject_id is None:
            raise ValueError("Subject-aware retriever cannot serve a query without identity")
        subject_aware = cast(SubjectAwareCandidateRetriever, retriever)
        return subject_aware.candidate_pool_for_subject(subject_id, history, limit=limit)
    return retriever.candidate_pool(history, limit=limit)


def validate_candidate_pool_limit(limit: int) -> None:
    if not 1 <= limit <= MAX_CANDIDATE_POOL:
        raise ValueError(f"candidate pool limit must be between 1 and {MAX_CANDIDATE_POOL}")
