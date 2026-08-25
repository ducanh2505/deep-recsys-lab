"""Reproducible Multi-VAE collaborative filtering."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deep-recsys-lab")
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.1.0"

__all__ = ["__version__"]
