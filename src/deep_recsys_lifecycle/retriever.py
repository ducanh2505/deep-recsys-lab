from __future__ import annotations

from collections.abc import Collection
from typing import Protocol

from .models import Candidate

MAX_CANDIDATE_POOL = 200


class CandidateRetriever(Protocol):
    """The small contract shared by retrievers in the evaluation pipeline."""

    @property
    def name(self) -> str: ...

    def candidate_pool(
        self, history: Collection[int], limit: int = MAX_CANDIDATE_POOL
    ) -> tuple[Candidate, ...]: ...


def validate_candidate_pool_limit(limit: int) -> None:
    if not 1 <= limit <= MAX_CANDIDATE_POOL:
        raise ValueError(f"candidate pool limit must be between 1 and {MAX_CANDIDATE_POOL}")
