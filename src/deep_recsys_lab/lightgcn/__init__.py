"""Independent LightGCN reproduction workflow.

The package intentionally does not register LightGCN with the MultiVAE serving
stack.  It exists solely for the MovieLens research experiment and report.
"""

from .config import LightGCNConfig, compose_lightgcn_config
from .data import (
    LightGCNData,
    load_lightgcn_data,
    prepare_lightgcn_data,
    prepare_lightgcn_from_splits,
)
from .model import LightGCN
from .yelp2018 import download_yelp2018, prepare_yelp2018_data, read_official_yelp2018_split

__all__ = [
    "LightGCN",
    "LightGCNConfig",
    "LightGCNData",
    "compose_lightgcn_config",
    "load_lightgcn_data",
    "prepare_lightgcn_from_splits",
    "prepare_lightgcn_data",
    "download_yelp2018",
    "prepare_yelp2018_data",
    "read_official_yelp2018_split",
]
