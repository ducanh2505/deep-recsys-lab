from __future__ import annotations

import numpy as np
import pytest

from deep_recsys_lab.serving.ranking import topk_unseen


def test_topk_unseen_masks_history_and_breaks_ties_by_catalog_index() -> None:
    scores = np.asarray([[0.5, 10.0, 0.5, 0.2, 0.5]], dtype=np.float32)
    seen = np.asarray([[0, 1, 0, 0, 0]], dtype=np.float32)

    indices, values = topk_unseen(scores, seen, 3)

    np.testing.assert_array_equal(indices, [[0, 2, 4]])
    np.testing.assert_allclose(values, [[0.5, 0.5, 0.5]])


def test_topk_unseen_supports_batches_and_validates_inputs() -> None:
    scores = np.asarray([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], dtype=np.float32)
    seen = np.zeros_like(scores)

    indices, _ = topk_unseen(scores, seen, 2)

    np.testing.assert_array_equal(indices, [[2, 1], [0, 2]])
    with pytest.raises(ValueError, match="equal shapes"):
        topk_unseen(scores, seen[:, :2], 1)
    with pytest.raises(ValueError, match="positive"):
        topk_unseen(scores, seen, 0)
    with pytest.raises(ValueError, match="fewer than k"):
        topk_unseen(scores, np.asarray([[1, 1, 0], [0, 0, 0]]), 2)
