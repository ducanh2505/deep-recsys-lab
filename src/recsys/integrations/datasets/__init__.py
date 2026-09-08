"""Local-file dataset integrations."""

from .movielens import read_movielens
from .yelp import read_yelp

__all__ = ["read_movielens", "read_yelp"]
