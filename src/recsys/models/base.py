"""Model lifecycle contract without a framework dependency."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recsys.artifacts import LoadedArtifact, RuntimeSpec
from recsys.artifacts.runtimes import ArtifactRuntime, runtime_from_artifact
from recsys.conf.schema import TrainingConfig
from recsys.datasets import PreparedDataset


@dataclass(frozen=True, slots=True)
class TrainingContext:
    data: PreparedDataset
    training: TrainingConfig
    parameters: dict[str, Any]
    scratch: Path


class ModelPlugin(ABC):
    """Framework-neutral train, evaluate, export, and runtime lifecycle."""

    name: str = ""

    @abstractmethod
    def train(self, context: TrainingContext) -> RuntimeSpec: ...

    def export(self, trained: RuntimeSpec) -> RuntimeSpec:
        if trained.plugin != self.name:
            raise ValueError("trained runtime plugin identity does not match its model plugin")
        return trained

    def evaluate(
        self, artifact: LoadedArtifact, data: PreparedDataset, *, top_k: int
    ) -> dict[str, float | int]:
        from recsys.experiments.runner import evaluate_artifact

        return evaluate_artifact(artifact, data, top_k=top_k)

    def load_runtime(self, artifact: LoadedArtifact) -> ArtifactRuntime:
        return runtime_from_artifact(artifact)
