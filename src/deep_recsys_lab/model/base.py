"""Model contracts shared by training, evaluation, and serving."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ModelOutput:
    """A recommender output with optional variational posterior statistics."""

    logits: torch.Tensor
    mu: torch.Tensor | None = None
    logvar: torch.Tensor | None = None

    def __iter__(self) -> Iterator[torch.Tensor | None]:
        """Preserve the convenient ``logits, mu, logvar = model(x)`` API."""

        yield self.logits
        yield self.mu
        yield self.logvar


class BaseRecommender(nn.Module, ABC):
    """Minimal interface a future recommender model must implement."""

    n_items: int

    @abstractmethod
    def forward(self, interactions: torch.Tensor, sample: bool | None = None) -> ModelOutput:
        """Return item logits and optional latent statistics."""

    def architecture_config(self) -> dict[str, Any]:
        """Return constructor arguments required to recreate this model."""

        return {"name": self.__class__.__name__.lower(), "n_items": self.n_items}

    def score(self, interactions: torch.Tensor) -> torch.Tensor:
        """Return deterministic ranking scores in evaluation mode."""

        was_training = self.training
        self.eval()
        with torch.inference_mode():
            output = self.forward(interactions, sample=False)
            scores = output.logits
        if was_training:
            self.train()
        return scores
