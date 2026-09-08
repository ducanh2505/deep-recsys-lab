"""Built-in model plugins."""

from .base import ModelPlugin, TrainingContext
from .registry import MODEL_REGISTRY, fit_model

__all__ = ["MODEL_REGISTRY", "ModelPlugin", "TrainingContext", "fit_model"]
