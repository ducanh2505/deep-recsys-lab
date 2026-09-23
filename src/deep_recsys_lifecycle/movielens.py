from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, overload
from zipfile import ZipFile

import duckdb
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .models import RatingEvent

OFFICIAL_ARCHIVE_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
OFFICIAL_ARCHIVE_MD5 = "cd245b17a1ae2cc31bb14903e1204af3"
EXPECTED_RATING_COUNT = 20_000_263


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def prepare_movielens_20m(
    cache_dir: Path,
    *,
    archive_path: Path | None = None,
    expected_md5: str = OFFICIAL_ARCHIVE_MD5,
    expected_rating_count: int = EXPECTED_RATING_COUNT,
) -> MovieLens20MSource:
    """Verify the official archive and prepare a chronologically ordered local Parquet source.

    The dataset stays under the caller's ignored cache directory. The sort runs in DuckDB with
    a bounded memory limit and spill directory; no ratings are retained as Python objects.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_path or cache_dir / "ml-20m.zip"
    if not archive.exists():
        with urllib.request.urlopen(OFFICIAL_ARCHIVE_URL, timeout=60) as response:
            with archive.open("wb") as destination:
                shutil.copyfileobj(response, destination, length=8 * 1024 * 1024)
    actual_md5 = _file_digest(archive, "md5")
    if actual_md5 != expected_md5.lower():
        raise ValueError(f"MovieLens archive checksum mismatch: {actual_md5}")

    archive_sha256 = _file_digest(archive, "sha256")
    ordered_path = cache_dir / "ratings-chronological.parquet"
    manifest_path = cache_dir / "source.json"
    if ordered_path.exists() and manifest_path.exists():
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            cached_manifest.get("archive_sha256") == archive_sha256
            and cached_manifest.get("rating_count") == expected_rating_count
            and cached_manifest.get("ordered_sha256") == _file_digest(ordered_path, "sha256")
        ):
            return MovieLens20MSource(ordered_path, archive_sha256)

    ratings_path = cache_dir / "ratings.csv"
    with ZipFile(archive) as bundle:
        with bundle.open("ml-20m/ratings.csv") as archive_member, ratings_path.open("wb") as target:
            shutil.copyfileobj(archive_member, target, length=8 * 1024 * 1024)

    temp_dir = cache_dir / "duckdb-temp"
    temp_dir.mkdir(exist_ok=True)
    provisional = cache_dir / "ratings-chronological.partial.parquet"
    con = duckdb.connect()
    try:
        con.execute("SET memory_limit = '6GB'")
        con.execute(f"SET temp_directory = {_sql_path(temp_dir)}")
        con.execute("SET preserve_insertion_order = false")
        con.execute(
            f"""
            COPY (
                WITH identified AS (
                    SELECT
                        'movielens:' || sha256(
                            'movielens' || chr(31) || CAST(userId AS VARCHAR) || chr(31)
                            || CAST(movieId AS VARCHAR) || chr(31)
                            || printf('%.17g', rating) || chr(31)
                            || CAST(timestamp AS VARCHAR)
                        ) AS event_id,
                        userId AS subject_id,
                        movieId AS movie_id,
                        rating,
                        timestamp AS event_time
                    FROM read_csv({_sql_path(ratings_path)}, header = true,
                        columns = {{'userId': 'BIGINT', 'movieId': 'BIGINT',
                                    'rating': 'DOUBLE', 'timestamp': 'BIGINT'}})
                )
                SELECT DISTINCT event_id, subject_id, movie_id, rating, event_time
                FROM identified
                ORDER BY event_time, event_id
            ) TO {_sql_path(provisional)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
    finally:
        con.close()
        ratings_path.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

    prepared_source = MovieLens20MSource(provisional, archive_sha256)
    if len(prepared_source) != expected_rating_count:
        provisional.unlink(missing_ok=True)
        raise ValueError(
            f"MovieLens rating count mismatch: {len(prepared_source)} != {expected_rating_count}"
        )
    provisional.replace(ordered_path)
    manifest: dict[str, Any] = {
        "source": "MovieLens 20M",
        "archive_url": OFFICIAL_ARCHIVE_URL,
        "archive_md5": actual_md5,
        "archive_sha256": archive_sha256,
        "ordered_sha256": _file_digest(ordered_path, "sha256"),
        "rating_count": expected_rating_count,
        "ordering": ["event_time", "event_id"],
        "event_id_algorithm": "RatingEvent.from_movielens",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return MovieLens20MSource(ordered_path, archive_sha256)


class MovieLens20MSource(Sequence[RatingEvent]):
    """A lazy, ordered view over the prepared source with a bounded row-group cache."""

    def __init__(self, path: Path, dataset_checksum: str) -> None:
        self.path = path
        self.dataset_checksum = dataset_checksum
        self._file = pq.ParquetFile(path)
        self._offsets = [0]
        for index in range(self._file.num_row_groups):
            self._offsets.append(self._offsets[-1] + self._file.metadata.row_group(index).num_rows)
        self._cached_group = -1
        self._cached_rows: dict[str, list[Any]] = {}

    def __len__(self) -> int:
        return self._offsets[-1]

    @overload
    def __getitem__(self, index: int) -> RatingEvent: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[RatingEvent]: ...

    def __getitem__(self, index: int | slice) -> RatingEvent | Sequence[RatingEvent]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return _SourceWindow(self, start, stop, step)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        from bisect import bisect_right

        group = bisect_right(self._offsets, index) - 1
        if group != self._cached_group:
            self._cached_rows = self._file.read_row_group(group).to_pydict()
            self._cached_group = group
        position = index - self._offsets[group]
        rows = self._cached_rows
        return RatingEvent(
            event_id=rows["event_id"][position],
            subject_id=rows["subject_id"][position],
            movie_id=rows["movie_id"][position],
            rating=rows["rating"][position],
            event_time=rows["event_time"][position],
        )


class _SourceWindow(Sequence[RatingEvent]):
    def __init__(self, source: MovieLens20MSource, start: int, stop: int, step: int) -> None:
        self.source = source
        self.range = range(start, stop, step)

    def __len__(self) -> int:
        return len(self.range)

    @overload
    def __getitem__(self, index: int) -> RatingEvent: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[RatingEvent]: ...

    def __getitem__(self, index: int | slice) -> RatingEvent | Sequence[RatingEvent]:
        if isinstance(index, slice):
            values = self.range[index]
            return _SourceWindow(self.source, values.start, values.stop, values.step)
        return self.source[self.range[index]]

    def __iter__(self) -> Iterator[RatingEvent]:
        for index in self.range:
            yield self.source[index]
