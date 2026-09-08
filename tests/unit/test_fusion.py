from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recsys.fusion import (
    FUSION_REGISTRY,
    LearnedLightGBMFusion,
    ReciprocalRankFusion,
    WeightedFusion,
)
from recsys.fusion.strategies import reciprocal_features


def rankings() -> dict[str, np.ndarray]:
    return {
        "a": np.array([3.0, 2.0, 1.0]),
        "b": np.array([1.0, 3.0, 2.0]),
    }


def test_heuristic_fusion_strategies() -> None:
    assert FUSION_REGISTRY.names() == ("learned_lightgbm", "rrf", "weighted")
    features = reciprocal_features(rankings(), constant=10)
    assert features.shape == (3, 4)
    rrf = ReciprocalRankFusion(10).fuse(rankings())
    weighted = WeightedFusion({"a": 2, "b": 1}).fuse(rankings())
    assert rrf.shape == weighted.shape == (3,)
    with pytest.raises(ValueError, match="positive"):
        ReciprocalRankFusion(0)
    with pytest.raises(ValueError, match="component"):
        ReciprocalRankFusion().fuse({})
    with pytest.raises(ValueError, match="component"):
        WeightedFusion().fuse({})


def test_lightgbm_fusion_uses_native_model(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    x = np.array([[1, 0], [0, 1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    y = np.array([1, 0, 1, 0], dtype=np.int8)
    fusion = LearnedLightGBMFusion.fit(
        x,
        y,
        [2, 2],
        seed=2,
        parameters={"num_boost_round": 2, "min_data_in_leaf": 1},
    )
    path = tmp_path / "fusion_model.txt"
    fusion.save(path)
    assert path.read_text().startswith("tree")
    loaded = LearnedLightGBMFusion.load(path)
    assert loaded.booster is not None


def test_learned_fusion_predicts_standard_features() -> None:
    class FakeBooster:
        def predict(self, values: np.ndarray) -> np.ndarray:
            return values.sum(axis=1)

    fusion = LearnedLightGBMFusion(FakeBooster())
    assert fusion.fuse({"only": np.array([1.0, 0.0])}).shape == (2,)
    with pytest.raises(ValueError, match="not fitted"):
        LearnedLightGBMFusion().fuse(rankings())
