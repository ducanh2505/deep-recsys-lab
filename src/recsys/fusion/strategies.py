"""Heuristic and learned fusion implementations."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np


class _Predictor(Protocol):
    def predict(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class _Tree:
    split_features: tuple[int, ...]
    thresholds: tuple[float, ...]
    left_children: tuple[int, ...]
    right_children: tuple[int, ...]
    leaf_values: tuple[float, ...]

    def predict_row(self, row: np.ndarray) -> float:
        node = 0
        while node >= 0 and self.split_features:
            feature = self.split_features[node]
            node = (
                self.left_children[node]
                if row[feature] <= self.thresholds[node]
                else self.right_children[node]
            )
        leaf = -node - 1 if node < 0 else 0
        return self.leaf_values[leaf]


class _NativeLightGBMPredictor:
    """Evaluate numeric LightGBM trees directly from their native text format."""

    def __init__(self, model_text: str) -> None:
        self.model_text = model_text
        self.trees = self._parse(model_text)

    @staticmethod
    def _parse(model_text: str) -> tuple[_Tree, ...]:
        trees: list[_Tree] = []
        for raw_block in model_text.split("\nTree=")[1:]:
            values = {
                line.partition("=")[0]: line.partition("=")[2]
                for line in raw_block.splitlines()
                if "=" in line
            }
            leaves = tuple(float(value) for value in values["leaf_value"].split())
            trees.append(
                _Tree(
                    tuple(int(value) for value in values.get("split_feature", "").split()),
                    tuple(float(value) for value in values.get("threshold", "").split()),
                    tuple(int(value) for value in values.get("left_child", "").split()),
                    tuple(int(value) for value in values.get("right_child", "").split()),
                    leaves,
                )
            )
        if not trees:
            raise ValueError("native LightGBM model contains no trees")
        return tuple(trees)

    def predict(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        return np.asarray(
            [sum(tree.predict_row(row) for tree in self.trees) for row in matrix],
            dtype=np.float64,
        )


def reciprocal_features(rankings: dict[str, np.ndarray], constant: float = 60.0) -> np.ndarray:
    features: list[np.ndarray] = []
    for name in sorted(rankings):
        scores = np.asarray(rankings[name], dtype=np.float64)
        item_indices = np.arange(len(scores))
        finite = np.isfinite(scores)
        rank = np.full(len(scores), len(scores) + 1, dtype=np.float64)
        ordered = item_indices[finite][np.lexsort((item_indices[finite], -scores[finite]))]
        rank[ordered] = np.arange(1, len(ordered) + 1)
        features.extend(
            (np.nan_to_num(scores, nan=0.0, neginf=0.0, posinf=0.0), 1.0 / (constant + rank))
        )
    return np.column_stack(features).astype(np.float32)


class ReciprocalRankFusion:
    name = "rrf"

    def __init__(self, rank_constant: float = 60.0) -> None:
        if rank_constant <= 0:
            raise ValueError("rank_constant must be positive")
        self.rank_constant = rank_constant

    def fuse(self, rankings: dict[str, np.ndarray]) -> np.ndarray:
        if not rankings:
            raise ValueError("RRF requires at least one component")
        features = reciprocal_features(rankings, self.rank_constant)
        return features[:, 1::2].sum(axis=1)


class WeightedFusion:
    name = "weighted"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {}

    def fuse(self, rankings: dict[str, np.ndarray]) -> np.ndarray:
        if not rankings:
            raise ValueError("weighted fusion requires at least one component")
        result = np.zeros_like(next(iter(rankings.values())), dtype=np.float64)
        for name, values in rankings.items():
            scores = np.asarray(values, dtype=np.float64)
            finite = np.isfinite(scores)
            normalized = np.zeros_like(scores)
            if finite.any():
                selected = scores[finite]
                scale = selected.std()
                normalized[finite] = (selected - selected.mean()) / (scale if scale > 0 else 1.0)
            result += float(self.weights.get(name, 1.0)) * normalized
        return result.astype(np.float32)


class LearnedLightGBMFusion:
    """Native LightGBM ranking model; no Python object serialization is used."""

    name = "learned_lightgbm"

    def __init__(self, booster: _Predictor | None = None, *, model_text: str | None = None) -> None:
        self.booster = booster
        self.model_text = model_text

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        groups: list[int],
        *,
        seed: int,
        parameters: dict[str, Any] | None = None,
    ) -> LearnedLightGBMFusion:
        options: dict[str, Any] = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "verbosity": -1,
            "seed": seed,
            "num_threads": 1,
        }
        options.update(parameters or {})
        with tempfile.TemporaryDirectory(prefix="recsys-lightgbm-") as temporary:
            root = Path(temporary)
            input_path = root / "training.npz"
            output_path = root / "fusion_model.txt"
            np.savez_compressed(
                input_path,
                features=np.asarray(features, dtype=np.float32),
                labels=np.asarray(labels, dtype=np.float32),
                groups=np.asarray(groups, dtype=np.int64),
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "recsys.fusion.lightgbm_worker",
                    str(input_path),
                    str(output_path),
                    json.dumps(options, separators=(",", ":"), sort_keys=True),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "unknown worker error"
                raise RuntimeError(f"LightGBM training worker failed: {detail}")
            return cls.load(output_path)

    @classmethod
    def load(cls, path: Path) -> LearnedLightGBMFusion:
        model_text = path.read_text(encoding="utf-8")
        return cls(_NativeLightGBMPredictor(model_text), model_text=model_text)

    def save(self, path: Path) -> None:
        if self.booster is None:
            raise ValueError("learned fusion is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model_text is not None:
            path.write_text(self.model_text, encoding="utf-8")
            return
        save_model = getattr(self.booster, "save_model", None)
        if save_model is None:
            raise ValueError("learned fusion predictor cannot be serialized")
        cast(Any, save_model)(str(path))

    def fuse(self, rankings: dict[str, np.ndarray]) -> np.ndarray:
        if self.booster is None:
            raise ValueError("learned fusion is not fitted")
        return np.asarray(self.booster.predict(reciprocal_features(rankings)), dtype=np.float32)
