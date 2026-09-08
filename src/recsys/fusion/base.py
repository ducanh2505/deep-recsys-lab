"""Fusion contract."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FusionStrategy(Protocol):
    name: str

    def fuse(self, rankings: dict[str, np.ndarray]) -> np.ndarray: ...
