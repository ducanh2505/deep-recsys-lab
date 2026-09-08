"""Provider-neutral reranker contract."""

from __future__ import annotations

from typing import Protocol

from recsys.core.types import Candidate


class Reranker(Protocol):
    name: str

    def rerank(
        self,
        history: list[dict[str, object]],
        candidates: list[Candidate],
        *,
        top_k: int,
    ) -> list[Candidate]: ...
