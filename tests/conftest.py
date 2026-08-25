from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("BENTOML_HOME", tempfile.mkdtemp(prefix="deep-recsys-pytest-"))

import pandas as pd
import pytest

from deep_recsys_lab.config import DatasetConfig
from deep_recsys_lab.data.preprocess import prepare_from_rows
from deep_recsys_lab.data.types import PreparedData


@pytest.fixture()
def prepared_data(tmp_path: Path) -> PreparedData:
    rows = [
        {"userId": user, "movieId": movie, "rating": 5.0}
        for user in range(1, 9)
        for movie in range(1, 11)
        if (user * movie) % 4 != 0
    ]
    movies = pd.DataFrame(
        {
            "movieId": list(range(1, 11)),
            "title": [f"Movie {movie}" for movie in range(1, 11)],
            "genres": ["Drama" if movie % 2 else "Comedy" for movie in range(1, 11)],
        }
    )
    return prepare_from_rows(
        pd.DataFrame(rows),
        tmp_path / "processed",
        movies=movies,
        config=DatasetConfig(
            name="synthetic",
            min_positive_ratings=3,
            n_validation_users=2,
            n_test_users=2,
            strict_user_counts=True,
        ),
    )
