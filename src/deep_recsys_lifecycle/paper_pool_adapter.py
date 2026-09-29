"""Join validation-only per-source Candidate Pool streams by deterministic Query key."""

from __future__ import annotations

import hashlib
import pickle
import re
from collections.abc import Callable, Collection, Generator, Iterator, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from .fusion import FUSION_BANKS, FusionQueryMode
from .models import Candidate
from .paper_pool_cache import PoolCache, PoolCacheResult, PoolQuery, PoolScope
from .retriever import CandidateRetriever, candidate_pool_for_query


@dataclass(frozen=True, slots=True)
class PoolSourceCacheInfo:
    key: str | None
    cache_hit: bool
    stored_depth: int | None


class _HashWriter:
    def __init__(self, digest: Any) -> None:
        self.digest = digest

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        return len(data)


def retriever_state_sha256(retriever: object) -> str:
    """Hash fitted dataclass state without copying its serialized bytes into RAM.

    Excluding ``init=False`` fields keeps lazy scoring accelerators out of the key.
    A retriever without picklable dataclass state needs a caller-supplied fingerprint.
    """

    if not is_dataclass(retriever):
        raise TypeError("retriever state fingerprint requires a dataclass")
    digest = hashlib.sha256(b"paper-pool-retriever-state-v1\0")
    state = (
        type(retriever).__module__,
        type(retriever).__qualname__,
        tuple(
            (field.name, getattr(retriever, field.name))
            for field in fields(retriever)
            if field.init
        ),
    )
    pickle.dump(state, _HashWriter(digest), protocol=5)
    return digest.hexdigest()


class PartitionPoolStream:
    """Yield one Query and its source pools without retaining a whole partition.

    ``cache_info`` is available as soon as the source entries are prepared. It is
    refreshed on access because a cache entry can be reclaimed and rebuilt between
    preparation and iteration.
    """

    def __init__(
        self,
        queries: tuple[PoolQuery, ...],
        source_names: tuple[str, ...],
        retrievers: Mapping[str, CandidateRetriever],
        results: Mapping[str, PoolCacheResult],
        depth: int,
    ) -> None:
        self.queries = queries
        self.source_names = source_names
        self._retrievers = retrievers
        self._results = results
        self._depth = depth

    @property
    def cache_info(self) -> dict[str, PoolSourceCacheInfo]:
        return {
            name: PoolSourceCacheInfo(
                key=result.key if result is not None else None,
                cache_hit=result.cache_hit if result is not None else False,
                stored_depth=result.stored_depth if result is not None else None,
            )
            for name in self.source_names
            for result in (self._results.get(name),)
        }

    def __iter__(self) -> Iterator[tuple[PoolQuery, dict[str, tuple[Candidate, ...]]]]:
        with ExitStack() as stack:
            streams = {
                name: stack.enter_context(
                    closing(
                        self._results[name].iter_pools()
                        if name in self._results
                        else _uncached_pools(self.queries, self._retrievers[name], self._depth)
                    )
                )
                for name in self.source_names
            }
            for expected in self.queries:
                pools: dict[str, tuple[Candidate, ...]] = {}
                for name, stream in streams.items():
                    try:
                        query, pool = next(stream)
                    except StopIteration as error:
                        raise ValueError(f"{name} Candidate Pool stream ended early") from error
                    if query != expected:
                        raise ValueError(f"{name} Candidate Pool Query order differs")
                    pools[name] = pool
                yield expected, pools
            for name, stream in streams.items():
                if next(stream, None) is not None:
                    raise ValueError(f"{name} Candidate Pool stream contains extra Queries")


def partition_pools(
    cache: PoolCache | None,
    *,
    mode: FusionQueryMode,
    partition: str,
    cohort_fingerprint: str,
    fit_fingerprint: str,
    code_fingerprint: str,
    retrievers: Mapping[str, CandidateRetriever],
    source_configs: Mapping[str, Mapping[str, object]],
    model_fingerprints: Mapping[str, str],
    seed: int,
    device_by_source: Mapping[str, str],
    checkpoint_by_source: Mapping[str, str],
    catalog_movie_ids: Collection[int],
    queries: Sequence[PoolQuery],
    depth: int,
    verify_deeper_prefix: bool = False,
) -> PartitionPoolStream:
    """Prepare one source shard stream per inner/validation partition.

    The fitted model fingerprint is mandatory even if no lossless model serializer
    exists. The cache receives only Query keys, Subject identity when applicable,
    and histories; the adapter has no Gold Set or test-scoring argument.
    """

    if mode not in FUSION_BANKS:
        raise ValueError("unknown paper Query mode")
    names = FUSION_BANKS[mode]
    required = set(names)
    for label, provided in (
        ("retrievers", retrievers),
        ("source configs", source_configs),
        ("model fingerprints", model_fingerprints),
        ("devices", device_by_source),
        ("checkpoint policies", checkpoint_by_source),
    ):
        if set(provided) != required:
            raise ValueError(f"{label} must match the {mode} source bank")
    for name in names:
        if retrievers[name].name != name:
            raise ValueError(f"retriever mapping key differs from source name: {name}")
        if not model_fingerprints[name]:
            raise ValueError(f"{name} requires a fitted-model content fingerprint")
    if partition != "validation" and not re.fullmatch(r"inner:[A-Za-z0-9_-]+", partition):
        raise ValueError("Candidate Pools are limited to inner and validation partitions")
    if isinstance(depth, bool) or depth < 1:
        raise ValueError("depth must be positive")

    ordered = tuple(queries)
    keys = [query.key for query in ordered]
    if any(isinstance(key, bool) or not isinstance(key, int | str) for key in keys):
        raise ValueError("Query keys must be integers or strings")
    if len(keys) != len(set(keys)) or keys != sorted(keys, key=_key_order):
        raise ValueError("Candidate Pool Queries require unique deterministic key order")
    if mode == "history_only" and any(query.subject_id is not None for query in ordered):
        raise ValueError("History-Only Queries cannot carry Subject identity")
    if mode == "known_user" and any(
        isinstance(query.subject_id, bool) or not isinstance(query.subject_id, int)
        for query in ordered
    ):
        raise ValueError("Known-User Queries require Subject identity")

    catalog = tuple(sorted(catalog_movie_ids))
    results: dict[str, PoolCacheResult] = {}
    if cache is not None:
        for name in names:
            retriever = retrievers[name]
            scope = PoolScope(
                mode=mode,
                partition=partition,
                cohort_fingerprint=cohort_fingerprint,
                fit_fingerprint=fit_fingerprint,
                model_fingerprint=model_fingerprints[name],
                code_fingerprint=code_fingerprint,
                source_name=name,
                source_config=source_configs[name],
                seed=seed,
                device=device_by_source[name],
                checkpoint_policy=checkpoint_by_source[name],
                catalog_movie_ids=catalog,
            )
            results[name] = cache.iter_or_build(
                scope,
                ordered,
                depth,
                _source_scorer(retriever),
                verify_deeper_prefix=verify_deeper_prefix,
            )
    return PartitionPoolStream(ordered, names, retrievers, results, depth)


def _uncached_pools(
    queries: tuple[PoolQuery, ...],
    retriever: CandidateRetriever,
    depth: int,
) -> Generator[tuple[PoolQuery, tuple[Candidate, ...]]]:
    for query in queries:
        yield query, candidate_pool_for_query(retriever, query.subject_id, query.history, depth)


def _source_scorer(
    retriever: CandidateRetriever,
) -> Callable[[PoolQuery, int], tuple[Candidate, ...]]:
    def score(query: PoolQuery, depth: int) -> tuple[Candidate, ...]:
        return candidate_pool_for_query(retriever, query.subject_id, query.history, depth)

    return score


def _key_order(key: int | str) -> tuple[int, int | str]:
    return (0, key) if isinstance(key, int) else (1, key)
