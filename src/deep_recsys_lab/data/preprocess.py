"""Deterministic MovieLens filtering, user splits, and CSR artifact writing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from ..config import DatasetConfig
from .types import MATRIX_FILES, PreparedData


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config(config: DatasetConfig | dict[str, Any] | None) -> DatasetConfig:
    if config is None:
        return DatasetConfig()
    if isinstance(config, DatasetConfig):
        return config
    return DatasetConfig(
        **{key: value for key, value in config.items() if key in DatasetConfig.__dataclass_fields__}
    )


def _csr_from_pairs(
    user_ids: Iterable[int],
    movie_ids: Iterable[int],
    item_to_index: dict[int, int],
    user_order: list[int],
) -> csr_matrix:
    user_to_index = {user_id: index for index, user_id in enumerate(user_order)}
    rows: list[int] = []
    cols: list[int] = []
    for user_id, movie_id in zip(user_ids, movie_ids, strict=True):
        if int(user_id) in user_to_index and int(movie_id) in item_to_index:
            rows.append(user_to_index[int(user_id)])
            cols.append(item_to_index[int(movie_id)])
    values = np.ones(len(rows), dtype=np.float32)
    matrix = csr_matrix(
        (values, (rows, cols)), shape=(len(user_order), len(item_to_index)), dtype=np.float32
    )
    matrix.data[:] = 1.0
    return matrix


def _split_user_history(
    user_movie_map: dict[int, np.ndarray],
    users: list[int],
    item_to_index: dict[int, int],
    fold_in_ratio: float,
    seed: int,
) -> tuple[csr_matrix, csr_matrix]:
    n_items = len(item_to_index)
    fold_in = csr_matrix((len(users), n_items), dtype=np.float32).tolil()
    fold_out = csr_matrix((len(users), n_items), dtype=np.float32).tolil()
    for row, user_id in enumerate(users):
        items = np.array(
            [
                item_to_index[int(item)]
                for item in user_movie_map[user_id]
                if int(item) in item_to_index
            ],
            dtype=np.int64,
        )
        items = np.unique(items)
        if len(items) == 0:
            continue
        if len(items) == 1:
            n_out = 0
        else:
            n_out = min(len(items) - 1, max(1, int(np.floor(len(items) * (1 - fold_in_ratio)))))
        rng = np.random.default_rng(seed + row * 1_000_003)
        out_positions = (
            np.sort(rng.choice(len(items), size=n_out, replace=False))
            if n_out
            else np.array([], dtype=int)
        )
        out_items = set(items[out_positions].tolist())
        in_items = [item for item in items.tolist() if item not in out_items]
        if in_items:
            fold_in[row, in_items] = 1.0
        if len(out_items):
            fold_out[row, sorted(out_items)] = 1.0
    return fold_in.tocsr(), fold_out.tocsr()


def _write_artifacts(
    output_dir: Path,
    matrices: dict[str, csr_matrix],
    item_ids: np.ndarray,
    movies: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> PreparedData:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        save_npz(
            output_dir / MATRIX_FILES[name], matrix.tocsr().astype(np.float32), compressed=True
        )
    np.save(output_dir / "item_ids.npy", item_ids.astype(np.int64), allow_pickle=False)
    (output_dir / "movies.json").write_text(
        json.dumps(movies, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    file_hashes = {
        filename: {
            "sha256": _sha256(output_dir / filename),
            "size_bytes": (output_dir / filename).stat().st_size,
        }
        for filename in [*MATRIX_FILES.values(), "item_ids.npy", "movies.json"]
    }
    manifest = {**manifest, "files": file_hashes}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    return PreparedData(
        train=matrices["train"],
        validation_fold_in=matrices["validation_fold_in"],
        validation_fold_out=matrices["validation_fold_out"],
        test_fold_in=matrices["test_fold_in"],
        test_fold_out=matrices["test_fold_out"],
        item_ids=item_ids,
        movies=movies,
        manifest=manifest,
        root=output_dir,
    )


def prepare_from_rows(
    rows: pd.DataFrame | Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    movies: pd.DataFrame | Iterable[dict[str, Any]] | None = None,
    config: DatasetConfig | dict[str, Any] | None = None,
) -> PreparedData:
    """Prepare an in-memory ratings table; useful for integration smoke tests."""

    cfg = _config(config)
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    required = {"userId", "movieId", "rating"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ratings are missing columns: {sorted(missing)}")
    frame = frame.loc[
        frame["rating"].astype(float) >= cfg.positive_rating_threshold,
        ["userId", "movieId", "rating"],
    ]
    counts = frame.groupby("userId", sort=True).size()
    eligible = sorted(
        int(user) for user, count in counts.items() if count >= cfg.min_positive_ratings
    )
    if len(eligible) < cfg.n_validation_users + cfg.n_test_users and cfg.strict_user_counts:
        raise ValueError(
            f"need {cfg.n_validation_users + cfg.n_test_users} eligible users, "
            f"found {len(eligible)}"
        )
    # The official reference notebook uses NumPy's legacy seeded permutation
    # and assigns the first users to training, followed by validation/test.
    # Keeping that convention is important for the expected 20,108-item
    # MovieLens-20M catalog.
    rng = np.random.RandomState(cfg.seed)
    shuffled = np.array(eligible, dtype=np.int64)
    shuffled = rng.permutation(shuffled)
    n_val = min(cfg.n_validation_users, len(shuffled))
    n_test = min(cfg.n_test_users, max(0, len(shuffled) - n_val))
    n_train = max(0, len(shuffled) - n_val - n_test)
    train_users = sorted(int(value) for value in shuffled[:n_train])
    validation_users = sorted(int(value) for value in shuffled[n_train : n_train + n_val])
    test_users = sorted(
        int(value) for value in shuffled[n_train + n_val : n_train + n_val + n_test]
    )
    if not train_users:
        # A tiny fixture can have no remaining users; keep the first eligible
        # user as train so the item catalog and model dimension remain defined.
        train_users = [validation_users.pop() if validation_users else test_users.pop()]
    user_movie_map = {
        int(user): np.sort(group["movieId"].astype(np.int64).unique())
        for user, group in frame.groupby("userId", sort=True)
    }
    catalog = sorted({int(movie) for user in train_users for movie in user_movie_map[user]})
    if not catalog:
        raise ValueError("training users produced an empty item catalog")
    item_ids = np.asarray(catalog, dtype=np.int64)
    item_to_index = {movie_id: index for index, movie_id in enumerate(catalog)}
    train = _csr_from_pairs(
        (user for user in frame["userId"].astype(int)),
        (movie for movie in frame["movieId"].astype(int)),
        item_to_index,
        train_users,
    )
    validation_in, validation_out = _split_user_history(
        user_movie_map, validation_users, item_to_index, cfg.fold_in_ratio, cfg.seed + 1
    )
    test_in, test_out = _split_user_history(
        user_movie_map, test_users, item_to_index, cfg.fold_in_ratio, cfg.seed + 2
    )
    if movies is None:
        movie_frame = pd.DataFrame(
            {
                "movieId": item_ids,
                "title": [str(value) for value in item_ids],
                "genres": ["" for _ in item_ids],
            }
        )
    else:
        movie_frame = (
            movies.copy() if isinstance(movies, pd.DataFrame) else pd.DataFrame(list(movies))
        )
        movie_frame = movie_frame.drop_duplicates("movieId").set_index("movieId")
        movie_frame = movie_frame.reindex(item_ids)
        default_titles = pd.Series(
            movie_frame.index.astype(str).to_numpy(), index=movie_frame.index
        )
        movie_frame["title"] = movie_frame["title"].fillna(default_titles)
        movie_frame["genres"] = movie_frame["genres"].fillna("")
        movie_frame = movie_frame.reset_index()
    metadata = [
        {"movie_id": int(row.movieId), "title": str(row.title), "genres": str(row.genres)}
        for row in movie_frame.itertuples(index=False)
    ]
    manifest = {
        "schema_version": "1.0",
        "dataset": cfg.name,
        "preprocessing": asdict(cfg),
        "expected_statistics": {
            "raw_ratings": 20_000_263,
            "raw_users": 138_493,
            "raw_movies": 27_278,
            "paper_target_n_items": 20_108,
        },
        "mapping": {
            "count": len(item_ids),
            "sha256": hashlib.sha256(item_ids.tobytes()).hexdigest(),
        },
        "statistics": {
            "n_train_users": len(train_users),
            "n_validation_users": len(validation_users),
            "n_test_users": len(test_users),
            "n_items": len(item_ids),
            "n_train_interactions": int(train.nnz),
            "n_validation_fold_out": int(validation_out.nnz),
            "n_test_fold_out": int(test_out.nnz),
        },
    }
    return _write_artifacts(
        Path(output_dir),
        {
            "train": train,
            "validation_fold_in": validation_in,
            "validation_fold_out": validation_out,
            "test_fold_in": test_in,
            "test_fold_out": test_out,
        },
        item_ids,
        metadata,
        manifest,
    )


def prepare_movielens20m(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    config: DatasetConfig | dict[str, Any] | None = None,
) -> PreparedData:
    """Read the GroupLens CSVs and write a complete versioned data artifact."""

    cfg = _config(config)
    root = Path(raw_dir)
    if (root / "ml-20m").exists():
        root = root / "ml-20m"
    ratings = root / cfg.ratings_file
    movies = root / cfg.movies_file
    if not ratings.exists():
        raise FileNotFoundError(f"MovieLens ratings file not found: {ratings}")
    if not movies.exists():
        raise FileNotFoundError(f"MovieLens movies file not found: {movies}")
    ratings_frame = pd.read_csv(ratings, usecols=["userId", "movieId", "rating"])
    movies_frame = pd.read_csv(movies, usecols=["movieId", "title", "genres"])
    return prepare_from_rows(ratings_frame, output_dir, movies=movies_frame, config=cfg)
