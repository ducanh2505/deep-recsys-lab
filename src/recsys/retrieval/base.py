"""Candidate-generator lifecycle contract."""

from __future__ import annotations

from typing import Protocol

from recsys.artifacts import RuntimeSpec
from recsys.models.base import TrainingContext


class RetrieverPlugin(Protocol):
    name: str

    def fit(self, context: TrainingContext) -> RuntimeSpec: ...
