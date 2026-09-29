"""Validation-only, content-addressed Candidate Pool shards for paper screening.

The caller supplies fingerprints for its prepared cohort, fit inputs, fitted retriever,
and implementation. No Gold Set or test scoring input is accepted by this interface.
Pools are streamed one Query at a time from bounded Parquet shards; an exact-depth hit
does not invoke the scorer. A deeper shard can back a shallower request only after every
Query's shallow pool has been compared with the scorer's exact shallow result.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .models import Candidate

_VERSION = 1
_SCHEMA = pa.schema(
    [
        ("query_ordinal", pa.int32()),
        ("movie_id", pa.int64()),
        ("rank", pa.int32()),
        ("score", pa.float64()),
    ]
)
_SOURCE_BANK = {
    "known_user": frozenset({"popularity", "itemknn", "multivae", "lightgcn"}),
    "history_only": frozenset({"popularity", "itemknn", "multivae"}),
}
_FORBIDDEN_CONFIG_KEYS = ("gold", "label", "target", "test")


class PoolCacheCorruptionError(ValueError):
    """A previously committed cache entry failed content or structure validation."""


class PoolCacheEvictedError(PoolCacheCorruptionError):
    """An entry was reclaimed between lookup and stream consumption."""


class PoolCacheCapacityError(OSError):
    """The cache cannot preserve its byte limit and free-space reserve."""


@dataclass(frozen=True, slots=True)
class PoolScope:
    """All score-affecting inputs except Query contents and requested depth.

    ``fit_fingerprint`` covers the ordered fit events/Subjects and partition;
    ``model_fingerprint`` covers actual fitted state, including a checkpoint;
    ``code_fingerprint`` covers relevant source/scoring code and dependencies.
    These are content hashes supplied by the runner, not Git dirty flags.
    """

    mode: Literal["known_user", "history_only"]
    partition: str
    cohort_fingerprint: str
    fit_fingerprint: str
    model_fingerprint: str
    code_fingerprint: str
    source_name: str
    source_config: Mapping[str, object]
    seed: int
    device: str
    checkpoint_policy: str
    catalog_movie_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PoolQuery:
    """Score-only Query identity; intentionally has no Gold Set field."""

    key: int | str
    subject_id: int | None
    history: tuple[int, ...]


@dataclass(slots=True)
class PoolCacheResult:
    key: str
    cache_hit: bool
    stored_depth: int
    requested_depth: int
    _cache: PoolCache
    _queries: tuple[PoolQuery, ...]
    _catalog: frozenset[int]
    _scope: PoolScope
    _score: Callable[[PoolQuery, int], tuple[Candidate, ...]]
    _verify_deeper_prefix: bool

    def iter_pools(self) -> Generator[tuple[PoolQuery, tuple[Candidate, ...]]]:
        """Read one bounded shard at a time, validating each reconstructed pool."""

        for attempt in range(3):
            try:
                yield from self._cache._iter_entry(self.key, self._queries, self._catalog)
                return
            except PoolCacheEvictedError:
                if attempt == 2:
                    raise
                # A result is a lookup, not a long-lived lease. Another worker may
                # reclaim it before iteration starts; rebuild from identical inputs.
                rebuilt = self._cache.iter_or_build(
                    self._scope,
                    self._queries,
                    self.requested_depth,
                    self._score,
                    verify_deeper_prefix=self._verify_deeper_prefix,
                )
                self.key = rebuilt.key
                self.cache_hit = rebuilt.cache_hit
                self.stored_depth = rebuilt.stored_depth


class PoolCache:
    """A bounded, verified cache for one source in an inner or validation partition."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = 12 * 1024**3,
        min_free_bytes: int = 8 * 1024**3,
        shard_queries: int = 512,
    ) -> None:
        if max_bytes < 1 or min_free_bytes < 0 or shard_queries < 1:
            raise ValueError("cache limits and shard_queries must be positive")
        if shard_queries > 2**31 - 1:
            raise ValueError("shard_queries exceeds the ordinal storage limit")
        self.root = Path(root)
        self.entries = self.root / "entries"
        self.locks = self.root / "locks"
        self.staging = self.root / "staging"
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.shard_queries = shard_queries
        for directory in (self.entries, self.locks, self.staging):
            directory.mkdir(parents=True, exist_ok=True)

    def iter_or_build(
        self,
        scope: PoolScope,
        queries: Iterable[PoolQuery],
        depth: int,
        score: Callable[[PoolQuery, int], tuple[Candidate, ...]],
        *,
        verify_deeper_prefix: bool = False,
    ) -> PoolCacheResult:
        """Return a streamable entry, building bounded shards on a miss.

        ``verify_deeper_prefix`` recomputes the requested shallow pool for *every*
        Query before making a shallow alias to a deeper entry. It never assumes a
        retriever's limit-invariance from its name or configuration.
        """

        scope_value, catalog = _scope_value(scope)
        if isinstance(depth, bool) or depth < 1:
            raise ValueError("depth must be a positive integer")
        ordered = _ordered_queries(queries, scope.mode)
        if len(ordered) > 2**31 - 1:
            raise ValueError("Query count exceeds the ordinal storage limit")
        query_hash = _query_hash(ordered)
        base_key = _digest({"scope": scope_value, "queries": query_hash, "version": _VERSION})
        key = _digest({"base_key": base_key, "depth": depth})
        entry = self._entry_path(base_key, key)

        with self._lock(key, exclusive=True):
            if entry.exists():
                manifest = self._manifest(entry, expected_key=key)
                self._verify_entry(entry, manifest)
                _touch(entry)
                return PoolCacheResult(
                    key,
                    True,
                    int(manifest["stored_depth"]),
                    depth,
                    self,
                    ordered,
                    catalog,
                    scope,
                    score,
                    verify_deeper_prefix,
                )

            if verify_deeper_prefix:
                deeper = self._find_deeper(base_key, depth)
                if deeper is not None:
                    deep_entry, deep_manifest = deeper
                    deep_key = str(deep_manifest["key"])
                    with self._lock(deep_key, exclusive=False):
                        self._verify_entry(deep_entry, deep_manifest)
                        if self._prefix_matches(
                            deep_manifest, deep_entry, ordered, catalog, depth, score
                        ):
                            alias = {
                                "version": _VERSION,
                                "kind": "alias",
                                "key": key,
                                "base_key": base_key,
                                "requested_depth": depth,
                                "stored_depth": int(deep_manifest["stored_depth"]),
                                "query_count": len(ordered),
                                "query_hash": query_hash,
                                "scope": scope_value,
                                "target_key": deep_key,
                                "target_manifest_hash": str(deep_manifest["checksum"]),
                                "shards": [],
                            }
                            self._commit_manifest_only(entry, alias)
                            _touch(deep_entry)
                            return PoolCacheResult(
                                key,
                                False,
                                int(deep_manifest["stored_depth"]),
                                depth,
                                self,
                                ordered,
                                catalog,
                                scope,
                                score,
                                verify_deeper_prefix,
                            )

            estimate = len(ordered) * min(depth, len(catalog)) * 32 + 1024 * (
                math.ceil(len(ordered) / self.shard_queries) + 1
            )
            self._ensure_space(estimate, protect=frozenset({key}))
            self._build(
                entry,
                key,
                base_key,
                query_hash,
                scope_value,
                ordered,
                catalog,
                depth,
                score,
            )
            try:
                self._ensure_space(0, protect=frozenset({key}))
            except PoolCacheCapacityError:
                shutil.rmtree(entry)
                raise
            return PoolCacheResult(
                key,
                False,
                depth,
                depth,
                self,
                ordered,
                catalog,
                scope,
                score,
                verify_deeper_prefix,
            )

    def get_or_build(
        self,
        scope: PoolScope,
        query: PoolQuery,
        depth: int,
        score: Callable[[PoolQuery, int], tuple[Candidate, ...]],
        *,
        verify_deeper_prefix: bool = False,
    ) -> tuple[Candidate, ...]:
        """Single-Query adapter for small fixtures; bulk screens should stream."""

        result = self.iter_or_build(
            scope, (query,), depth, score, verify_deeper_prefix=verify_deeper_prefix
        )
        with closing(result.iter_pools()) as stream:
            return next(stream)[1]

    def _build(
        self,
        entry: Path,
        key: str,
        base_key: str,
        query_hash: str,
        scope_value: Mapping[str, object],
        queries: tuple[PoolQuery, ...],
        catalog: frozenset[int],
        depth: int,
        score: Callable[[PoolQuery, int], tuple[Candidate, ...]],
    ) -> None:
        stage = Path(tempfile.mkdtemp(prefix=f"{key}-", dir=self.staging))
        lease = (stage / ".lease").open("a+b")
        fcntl.flock(lease, fcntl.LOCK_EX)
        try:
            shards: list[dict[str, object]] = []
            for start in range(0, len(queries), self.shard_queries):
                chunk = queries[start : start + self.shard_queries]
                ordinals: list[int] = []
                movie_ids: list[int] = []
                ranks: list[int] = []
                scores: list[float] = []
                for ordinal, query in enumerate(chunk, start):
                    pool = score(query, depth)
                    _validate_pool(pool, query, catalog, depth)
                    for candidate in pool:
                        ordinals.append(ordinal)
                        movie_ids.append(candidate.movie_id)
                        ranks.append(candidate.rank)
                        scores.append(float(candidate.score))
                name = f"shard-{len(shards):06d}.parquet"
                path = stage / name
                temp_path = stage / f".{name}.tmp"
                table = pa.Table.from_arrays(
                    [
                        pa.array(ordinals, type=pa.int32()),
                        pa.array(movie_ids, type=pa.int64()),
                        pa.array(ranks, type=pa.int32()),
                        pa.array(scores, type=pa.float64()),
                    ],
                    schema=_SCHEMA,
                )
                pq.write_table(table, temp_path, compression="zstd", use_dictionary=False)
                _sync_file(temp_path)
                os.replace(temp_path, path)
                _sync_directory(stage)
                shards.append(
                    {
                        "name": name,
                        "start": start,
                        "query_count": len(chunk),
                        "row_count": len(ordinals),
                        "bytes": path.stat().st_size,
                        "sha256": _file_hash(path),
                    }
                )
                self._ensure_space(0, protect=frozenset({key}))
            manifest: dict[str, object] = {
                "version": _VERSION,
                "kind": "data",
                "key": key,
                "base_key": base_key,
                "requested_depth": depth,
                "stored_depth": depth,
                "query_count": len(queries),
                "query_hash": query_hash,
                "scope": scope_value,
                "shards": shards,
            }
            _write_manifest(stage / "manifest.json", manifest)
            entry.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, entry)
            _sync_directory(entry.parent)
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)
            lease.close()
            if stage.exists():
                shutil.rmtree(stage)

    def _commit_manifest_only(self, entry: Path, manifest: dict[str, object]) -> None:
        stage = Path(tempfile.mkdtemp(prefix=f"{entry.name}-", dir=self.staging))
        try:
            _write_manifest(stage / "manifest.json", manifest)
            entry.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, entry)
            _sync_directory(entry.parent)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _find_deeper(self, base_key: str, depth: int) -> tuple[Path, Mapping[str, Any]] | None:
        parent = self.entries / base_key
        choices: list[tuple[int, Path, Mapping[str, Any]]] = []
        if not parent.exists():
            return None
        for entry in parent.iterdir():
            if not entry.is_dir():
                continue
            manifest = self._manifest(entry)
            stored_depth = int(manifest["stored_depth"])
            if manifest["kind"] == "data" and stored_depth > depth:
                choices.append((stored_depth, entry, manifest))
        if not choices:
            return None
        _, entry, manifest = min(choices, key=lambda item: item[0])
        return entry, manifest

    def _prefix_matches(
        self,
        manifest: Mapping[str, Any],
        entry: Path,
        queries: tuple[PoolQuery, ...],
        catalog: frozenset[int],
        depth: int,
        score: Callable[[PoolQuery, int], tuple[Candidate, ...]],
    ) -> bool:
        for query, deeper_pool in self._iter_data(entry, manifest, queries, catalog):
            shallow = score(query, depth)
            _validate_pool(shallow, query, catalog, depth)
            if not _pools_identical(shallow, deeper_pool[:depth]):
                return False
        return True

    def _iter_entry(
        self,
        key: str,
        queries: tuple[PoolQuery, ...],
        catalog: frozenset[int],
    ) -> Iterator[tuple[PoolQuery, tuple[Candidate, ...]]]:
        with self._lock(key, exclusive=False):
            entry = self._locate(key)
            if entry is None:
                raise PoolCacheEvictedError(f"Candidate Pool cache entry disappeared: {key}")
            manifest = self._manifest(entry, expected_key=key)
            self._verify_entry(entry, manifest)
            if int(manifest["query_count"]) != len(queries) or str(
                manifest["query_hash"]
            ) != _query_hash(queries):
                raise PoolCacheCorruptionError("Candidate Pool Query manifest mismatch")
            if manifest["kind"] == "alias":
                target_key = str(manifest["target_key"])
                with self._lock(target_key, exclusive=False):
                    target = self._locate(target_key)
                    if target is None:
                        raise PoolCacheCorruptionError("Candidate Pool alias target is missing")
                    target_manifest = self._manifest(target, expected_key=target_key)
                    if target_manifest["checksum"] != manifest["target_manifest_hash"]:
                        raise PoolCacheCorruptionError("Candidate Pool alias target changed")
                    self._verify_entry(target, target_manifest)
                    for query, pool in self._iter_data(target, target_manifest, queries, catalog):
                        yield query, pool[: int(manifest["requested_depth"])]
            else:
                yield from self._iter_data(entry, manifest, queries, catalog)
            _touch(entry)

    def _iter_data(
        self,
        entry: Path,
        manifest: Mapping[str, Any],
        queries: tuple[PoolQuery, ...],
        catalog: frozenset[int],
    ) -> Iterator[tuple[PoolQuery, tuple[Candidate, ...]]]:
        depth = int(manifest["stored_depth"])
        for shard in manifest["shards"]:
            start = int(shard["start"])
            count = int(shard["query_count"])
            table = pq.read_table(entry / str(shard["name"]))
            if table.schema != _SCHEMA or table.num_rows != int(shard["row_count"]):
                raise PoolCacheCorruptionError("Candidate Pool shard schema or row count changed")
            rows: list[list[Candidate]] = [[] for _ in range(count)]
            ordinals = table.column("query_ordinal").to_pylist()
            movie_ids = table.column("movie_id").to_pylist()
            ranks = table.column("rank").to_pylist()
            scores = table.column("score").to_pylist()
            for ordinal, movie_id, rank, value in zip(
                ordinals, movie_ids, ranks, scores, strict=True
            ):
                if ordinal is None or not start <= ordinal < start + count:
                    raise PoolCacheCorruptionError("Candidate Pool row has invalid Query ordinal")
                if movie_id is None or rank is None or value is None:
                    raise PoolCacheCorruptionError("Candidate Pool row contains a null")
                rows[ordinal - start].append(Candidate(movie_id, value, rank))
            for offset, pool in enumerate(rows):
                query = queries[start + offset]
                result = tuple(pool)
                try:
                    _validate_pool(result, query, catalog, depth)
                except ValueError as error:
                    raise PoolCacheCorruptionError(str(error)) from error
                yield query, result

    def _verify_entry(self, entry: Path, manifest: Mapping[str, Any]) -> None:
        if manifest["kind"] == "alias":
            if manifest["target_key"] == manifest["key"]:
                raise PoolCacheCorruptionError("Candidate Pool alias cannot target itself")
            target = self._locate(str(manifest["target_key"]))
            if target is None:
                raise PoolCacheCorruptionError("Candidate Pool alias target is missing")
            target_manifest = self._manifest(target, expected_key=str(manifest["target_key"]))
            if target_manifest["kind"] != "data":
                raise PoolCacheCorruptionError("Candidate Pool alias must target data")
            if target_manifest["checksum"] != manifest["target_manifest_hash"]:
                raise PoolCacheCorruptionError("Candidate Pool alias target changed")
            self._verify_entry(target, target_manifest)
            return
        expected_start = 0
        total_queries = 0
        for shard in manifest["shards"]:
            if not isinstance(shard, Mapping):
                raise PoolCacheCorruptionError("Candidate Pool shard metadata is invalid")
            name = shard.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"shard-\d{6}\.parquet", name):
                raise PoolCacheCorruptionError("Candidate Pool shard name is invalid")
            path = entry / name
            if not path.is_file() or path.stat().st_size != shard.get("bytes"):
                raise PoolCacheCorruptionError("Candidate Pool shard is missing or changed")
            if _file_hash(path) != shard.get("sha256"):
                raise PoolCacheCorruptionError("Candidate Pool shard checksum mismatch")
            if shard.get("start") != expected_start:
                raise PoolCacheCorruptionError("Candidate Pool shard order changed")
            count = shard.get("query_count")
            if not isinstance(count, int) or count < 1:
                raise PoolCacheCorruptionError("Candidate Pool shard Query count is invalid")
            expected_start += count
            total_queries += count
        if total_queries != manifest.get("query_count"):
            raise PoolCacheCorruptionError("Candidate Pool Query count mismatch")

    def _manifest(self, entry: Path, *, expected_key: str | None = None) -> Mapping[str, Any]:
        try:
            raw = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("not an object")
            checksum = raw.pop("checksum")
            if checksum != _digest(raw):
                raise ValueError("checksum mismatch")
            raw["checksum"] = checksum
            if raw.get("version") != _VERSION or raw.get("kind") not in {"data", "alias"}:
                raise ValueError("unsupported version or kind")
            if expected_key is not None and raw.get("key") != expected_key:
                raise ValueError("key mismatch")
            if not isinstance(raw.get("shards"), list):
                raise ValueError("missing shards")
            if not isinstance(raw.get("scope"), dict) or not isinstance(raw.get("query_hash"), str):
                raise ValueError("missing content-addressed inputs")
            base_key = _digest(
                {"scope": raw["scope"], "queries": raw["query_hash"], "version": _VERSION}
            )
            if raw.get("base_key") != base_key or raw.get("key") != _digest(
                {"base_key": base_key, "depth": raw["requested_depth"]}
            ):
                raise ValueError("content-addressed key mismatch")
            return raw
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise PoolCacheCorruptionError(f"Invalid Candidate Pool manifest: {entry}") from error

    def _locate(self, key: str) -> Path | None:
        for path in self.entries.glob(f"*/{key}"):
            if path.is_dir():
                return path
        return None

    def _entry_path(self, base_key: str, key: str) -> Path:
        return self.entries / base_key / key

    def _ensure_space(self, needed_bytes: int, *, protect: frozenset[str]) -> None:
        with self._lock("eviction", exclusive=True):
            self._reclaim_stale_staging()
            entries = [path for path in self.entries.glob("*/*") if path.is_dir()]
            sizes = {path: _directory_size(path) for path in entries}
            total = sum(sizes.values()) + sum(
                _directory_size(path) for path in self.staging.iterdir() if path.is_dir()
            )
            free = shutil.disk_usage(self.root).free
            if (
                total + needed_bytes <= self.max_bytes
                and free - needed_bytes >= self.min_free_bytes
            ):
                return
            aliases: dict[str, list[Path]] = {}
            for path in entries:
                try:
                    manifest = self._manifest(path)
                except PoolCacheCorruptionError:
                    continue
                if manifest["kind"] == "alias":
                    aliases.setdefault(str(manifest["target_key"]), []).append(path)
            for path in sorted(entries, key=lambda item: item.stat().st_mtime_ns):
                if path.name in protect or not path.exists():
                    continue
                with self._try_exclusive_lock(path.name) as acquired:
                    if not acquired:
                        continue
                    active_alias = False
                    for alias in aliases.get(path.name, []):
                        if alias.exists() and alias.name not in protect:
                            with self._try_exclusive_lock(alias.name) as alias_acquired:
                                if alias_acquired:
                                    size = sizes.get(alias, 0)
                                    shutil.rmtree(alias)
                                    total -= size
                                    free = shutil.disk_usage(self.root).free
                                else:
                                    active_alias = True
                        elif alias.name in protect:
                            active_alias = True
                    if active_alias:
                        continue
                    if path.exists():
                        size = sizes[path]
                        shutil.rmtree(path)
                        total -= size
                        free = shutil.disk_usage(self.root).free
                if (
                    total + needed_bytes <= self.max_bytes
                    and free - needed_bytes >= self.min_free_bytes
                ):
                    return
            if total + needed_bytes > self.max_bytes or free - needed_bytes < self.min_free_bytes:
                raise PoolCacheCapacityError("Candidate Pool cache lacks bounded scratch space")

    def _reclaim_stale_staging(self) -> None:
        for stage in self.staging.iterdir():
            if not stage.is_dir():
                continue
            lease_path = stage / ".lease"
            if not lease_path.exists():
                if time.time() - stage.stat().st_mtime > 60:
                    shutil.rmtree(stage)
                continue
            with lease_path.open("a+b") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    shutil.rmtree(stage)
                finally:
                    fcntl.flock(lease, fcntl.LOCK_UN)

    @contextmanager
    def _lock(self, key: str, *, exclusive: bool) -> Iterator[None]:
        path = self.locks / f"{key}.lock"
        with path.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @contextmanager
    def _try_exclusive_lock(self, key: str) -> Iterator[bool]:
        path = self.locks / f"{key}.lock"
        with path.open("a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
            else:
                try:
                    yield True
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)


def _scope_value(scope: PoolScope) -> tuple[dict[str, object], frozenset[int]]:
    if scope.mode not in _SOURCE_BANK or scope.source_name not in _SOURCE_BANK[scope.mode]:
        raise ValueError("source is not available in the requested Query mode")
    if scope.partition != "validation" and not re.fullmatch(
        r"inner:[A-Za-z0-9_-]+", scope.partition
    ):
        raise ValueError("Candidate Pool cache accepts only inner or validation partitions")
    for name in ("cohort_fingerprint", "fit_fingerprint", "model_fingerprint", "code_fingerprint"):
        if not getattr(scope, name):
            raise ValueError(f"{name} is required for Candidate Pool cache keys")
    if isinstance(scope.seed, bool) or not isinstance(scope.seed, int):
        raise ValueError("seed must be an integer")
    if not scope.device or not scope.checkpoint_policy:
        raise ValueError("device and checkpoint policy are required")
    catalog = tuple(sorted(scope.catalog_movie_ids))
    if len(catalog) != len(set(catalog)) or not catalog:
        raise ValueError("Candidate Catalog must contain unique Movie IDs")
    if any(isinstance(movie_id, bool) or not isinstance(movie_id, int) for movie_id in catalog):
        raise ValueError("Candidate Catalog Movie IDs must be integers")
    _reject_label_keys(scope.source_config)
    config_json = _canonical(scope.source_config)
    value: dict[str, object] = {
        "mode": scope.mode,
        "partition": scope.partition,
        "cohort_fingerprint": scope.cohort_fingerprint,
        "fit_fingerprint": scope.fit_fingerprint,
        "model_fingerprint": scope.model_fingerprint,
        "code_fingerprint": scope.code_fingerprint,
        "source_name": scope.source_name,
        "source_config": json.loads(config_json),
        "seed": scope.seed,
        "device": scope.device,
        "checkpoint_policy": scope.checkpoint_policy,
        "catalog_hash": _digest(catalog),
    }
    return value, frozenset(catalog)


def _reject_label_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, member in value.items():
            if not isinstance(key, str) or any(
                word in key.lower() for word in _FORBIDDEN_CONFIG_KEYS
            ):
                raise ValueError("source config may contain only score-affecting fields")
            _reject_label_keys(member)
    elif isinstance(value, (list, tuple)):
        for member in value:
            _reject_label_keys(member)


def _ordered_queries(queries: Iterable[PoolQuery], mode: str) -> tuple[PoolQuery, ...]:
    received = tuple(queries)
    if any(
        isinstance(query.key, bool) or not isinstance(query.key, int | str) for query in received
    ):
        raise ValueError("Query key must be an integer or string")
    ordered = tuple(sorted(received, key=lambda query: _query_sort_key(query.key)))
    keys: set[int | str] = set()
    for query in ordered:
        if query.key in keys:
            raise ValueError("Candidate Pool Queries require distinct keys")
        keys.add(query.key)
        if mode == "history_only" and query.subject_id is not None:
            raise ValueError("History-Only Candidate Pool Queries cannot carry Subject identity")
        if mode == "known_user" and (
            isinstance(query.subject_id, bool) or not isinstance(query.subject_id, int)
        ):
            raise ValueError("Known-User Candidate Pool Queries require Subject identity")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in query.history):
            raise ValueError("Query history must contain Movie IDs")
    return ordered


def _query_sort_key(key: int | str) -> tuple[int, int | str]:
    return (0, key) if isinstance(key, int) else (1, key)


def _query_hash(queries: Sequence[PoolQuery]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        digest.update(
            _canonical([query.key, query.subject_id, query.history]).encode("utf-8") + b"\n"
        )
    return digest.hexdigest()


def _validate_pool(
    pool: tuple[Candidate, ...], query: PoolQuery, catalog: frozenset[int], depth: int
) -> None:
    if not isinstance(pool, tuple) or len(pool) > depth:
        raise ValueError("Candidate Pool must be a bounded tuple")
    seen: set[int] = set()
    history = set(query.history)
    for rank, candidate in enumerate(pool, 1):
        if not isinstance(candidate, Candidate) or candidate.rank != rank:
            raise ValueError("Candidate Pool ranks must be contiguous")
        if candidate.movie_id not in catalog or candidate.movie_id in history:
            raise ValueError("Candidate Pool contains an ineligible Movie")
        if candidate.movie_id in seen:
            raise ValueError("Candidate Pool contains a duplicate Movie")
        seen.add(candidate.movie_id)
        score = float(candidate.score)
        if not math.isfinite(score) or score != candidate.score:
            raise ValueError("Candidate Pool score is not finite and lossless float64")


def _pools_identical(left: Sequence[Candidate], right: Sequence[Candidate]) -> bool:
    return len(left) == len(right) and all(
        one.movie_id == two.movie_id
        and one.rank == two.rank
        and struct.pack("!d", float(one.score)) == struct.pack("!d", float(two.score))
        for one, two in zip(left, right, strict=True)
    )


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    value = {**manifest, "checksum": _digest(manifest)}
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(_canonical(value), encoding="utf-8")
    _sync_file(temp)
    os.replace(temp, path)
    _sync_directory(path.parent)


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def _touch(path: Path) -> None:
    os.utime(path, None)
