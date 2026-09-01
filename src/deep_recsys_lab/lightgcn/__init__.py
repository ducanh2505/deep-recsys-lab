"""Independent LightGCN reproduction workflow.

The package intentionally does not register LightGCN with the MultiVAE serving
stack.  It exists solely for the MovieLens research experiment and report.
"""

from .config import LightGCNConfig, compose_lightgcn_config
from .data import LightGCNData, load_lightgcn_data, prepare_lightgcn_data
from .model import LightGCN

__all__ = [
    "LightGCN",
    "LightGCNConfig",
    "LightGCNData",
    "compose_lightgcn_config",
    "load_lightgcn_data",
    "prepare_lightgcn_data",
]
