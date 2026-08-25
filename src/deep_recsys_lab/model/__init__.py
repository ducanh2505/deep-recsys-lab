"""Model interfaces and registry."""

from .base import BaseRecommender, ModelOutput
from .multvae import MultiVAE
from .registry import build_model, register_model

__all__ = ["BaseRecommender", "ModelOutput", "MultiVAE", "build_model", "register_model"]
