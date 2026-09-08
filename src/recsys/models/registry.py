"""Lazy model plugin registry."""

from __future__ import annotations

from importlib import import_module
from typing import cast

from recsys.artifacts import RuntimeSpec
from recsys.core.registry import Registry

from .base import ModelPlugin, TrainingContext

MODEL_REGISTRY: Registry[str] = Registry("model")
for _name, _target in {
    "bpr": "recsys.models.factorization:BPRPlugin",
    "lightgcn": "recsys.models.lightgcn:LightGCNPlugin",
    "multivae": "recsys.models.multivae:MultiVAEPlugin",
    "sasrec": "recsys.models.sasrec:SASRecPlugin",
    "two_tower": "recsys.models.factorization:TwoTowerPlugin",
}.items():
    MODEL_REGISTRY.register(_name, _target)


def _load(target: str) -> ModelPlugin:
    module_name, separator, attribute = target.partition(":")
    if not separator:
        raise ValueError(f"invalid plugin target: {target}")
    plugin_type = getattr(import_module(module_name), attribute)
    return cast(ModelPlugin, plugin_type())


def fit_model(name: str, context: TrainingContext) -> RuntimeSpec:
    plugin = _load(MODEL_REGISTRY.get(name))
    return plugin.export(plugin.train(context))
