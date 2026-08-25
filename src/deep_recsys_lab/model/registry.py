"""A small model registry decoupling configs from construction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import BaseRecommender

ModelFactory = Callable[..., BaseRecommender]
_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    def decorator(factory: ModelFactory) -> ModelFactory:
        key = name.strip().lower()
        if key in _REGISTRY and _REGISTRY[key] is not factory:
            raise ValueError(f"model is already registered: {name}")
        _REGISTRY[key] = factory
        return factory

    return decorator


def build_model(name: str, **kwargs: Any) -> BaseRecommender:
    """Build a registered model by name."""

    key = name.strip().lower()
    if key not in _REGISTRY and key == "multvae":
        # Importing the package registers built-ins without making registry.py
        # import itself circular.
        from .multvae import MultiVAE

        _REGISTRY[key] = MultiVAE
    try:
        return _REGISTRY[key](**kwargs)
    except KeyError as exc:
        raise KeyError(f"unknown recommender model {name!r}; choices={sorted(_REGISTRY)}") from exc


def registered_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
