"""MovieLens download, preprocessing, and sparse interaction datasets."""

from .dataset import InteractionDataset, load_prepared_data
from .download import download_movielens20m, extract_movielens_archive
from .preprocess import prepare_from_rows, prepare_movielens20m
from .types import PreparedData

__all__ = [
    "InteractionDataset",
    "PreparedData",
    "download_movielens20m",
    "extract_movielens_archive",
    "load_prepared_data",
    "prepare_from_rows",
    "prepare_movielens20m",
]
