"""Optional Ollama reranker configured without provider-specific core assumptions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from recsys.core.types import Candidate


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    endpoint: str = "http://127.0.0.1:11434"
    model: str = ""
    prompt_template: str = (
        "Given this interaction history:\n{history}\n"
        "Score the relevance of this candidate from 0 to 1:\n{candidate}\n"
        'Return JSON only: {"score": number}.'
    )
    timeout_seconds: float = 30.0


class OllamaReranker:
    name = "ollama"

    def __init__(self, settings: OllamaSettings) -> None:
        if not settings.model:
            raise ValueError("Ollama model must be configured")
        self.settings = settings

    def _score(self, history: list[dict[str, object]], candidate: Candidate) -> float:
        import httpx

        prompt = self.settings.prompt_template.format(
            history=json.dumps(history, ensure_ascii=False),
            candidate=json.dumps(candidate.metadata, ensure_ascii=False),
        )
        response = httpx.post(
            f"{self.settings.endpoint.rstrip('/')}/api/generate",
            json={
                "model": self.settings.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        parsed = json.loads(str(payload["response"]))
        return float(parsed["score"])

    def rerank(
        self,
        history: list[dict[str, object]],
        candidates: list[Candidate],
        *,
        top_k: int,
    ) -> list[Candidate]:
        rescored = [
            Candidate(candidate.item_index, self._score(history, candidate), candidate.metadata)
            for candidate in candidates
        ]
        return sorted(rescored, key=lambda value: (-value.score, value.item_index))[:top_k]
