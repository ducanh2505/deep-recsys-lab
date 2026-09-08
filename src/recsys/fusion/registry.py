"""Built-in fusion factories and extension registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from recsys.core.registry import Registry

from .base import FusionStrategy
from .strategies import LearnedLightGBMFusion, ReciprocalRankFusion, WeightedFusion

FusionFactory = Callable[[dict[str, Any], Path], FusionStrategy]


def _rrf(settings: dict[str, Any], _: Path) -> FusionStrategy:
    parameters = cast(dict[str, Any], settings.get("parameters", {}))
    return ReciprocalRankFusion(float(parameters.get("rank_constant", 60)))


def _weighted(settings: dict[str, Any], _: Path) -> FusionStrategy:
    return WeightedFusion(cast(dict[str, float], settings.get("weights", {})))


def _learned(_: dict[str, Any], root: Path) -> FusionStrategy:
    return LearnedLightGBMFusion.load(root / "fusion_model.txt")


FUSION_REGISTRY: Registry[FusionFactory] = Registry("fusion")
FUSION_REGISTRY.register("rrf", _rrf)
FUSION_REGISTRY.register("weighted", _weighted)
FUSION_REGISTRY.register("learned_lightgbm", _learned)


def load_fusion(name: str, settings: dict[str, Any], root: Path) -> FusionStrategy:
    return FUSION_REGISTRY.get(name)(settings, root)
