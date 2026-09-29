from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from deep_recsys_lifecycle.models import Candidate
from deep_recsys_lifecycle.paper_pool_adapter import partition_pools, retriever_state_sha256
from deep_recsys_lifecycle.paper_pool_cache import PoolCache, PoolQuery


@dataclass
class _Retriever:
    name: str
    offset: int
    fail: bool = False
    calls: list[tuple[int | None, tuple[int, ...], int]] = field(default_factory=list)

    def candidate_pool(self, history: tuple[int, ...], limit: int = 200) -> tuple[Candidate, ...]:
        return self._pool(None, history, limit)

    def _pool(
        self, subject_id: int | None, history: tuple[int, ...], limit: int
    ) -> tuple[Candidate, ...]:
        if self.fail:
            raise AssertionError(f"{self.name} should not rescore a cache hit")
        self.calls.append((subject_id, history, limit))
        available = [movie_id for movie_id in (10, 20, 30, 40) if movie_id not in history]
        return tuple(
            Candidate(movie_id, self.offset + 0.1234567890123456 / movie_id, rank)
            for rank, movie_id in enumerate(available[:limit], 1)
        )


@dataclass
class _SubjectRetriever(_Retriever):
    def candidate_pool(self, history: tuple[int, ...], limit: int = 200) -> tuple[Candidate, ...]:
        raise AssertionError("LightGCN requires Subject identity")

    def candidate_pool_for_subject(
        self, subject_id: int, history: tuple[int, ...], limit: int = 200
    ) -> tuple[Candidate, ...]:
        return self._pool(subject_id, history, limit)


def _bank(known: bool) -> dict[str, _Retriever]:
    values: dict[str, _Retriever] = {
        "popularity": _Retriever("popularity", 0),
        "itemknn": _Retriever("itemknn", 1),
        "multivae": _Retriever("multivae", 2),
    }
    if known:
        values["lightgcn"] = _SubjectRetriever("lightgcn", 3)
    return values


def _options(retrievers: dict[str, _Retriever], queries: tuple[PoolQuery, ...]) -> dict[str, Any]:
    return {
        "partition": "validation",
        "cohort_fingerprint": "cohort-sha",
        "fit_fingerprint": "ordered-fit-events-sha",
        "code_fingerprint": "scoring-code-sha",
        "retrievers": retrievers,
        "source_configs": {
            name: {"weight": retriever.offset} for name, retriever in retrievers.items()
        },
        "model_fingerprints": {name: f"model-{name}-sha" for name in retrievers},
        "seed": 42,
        "device_by_source": {name: "cpu" for name in retrievers},
        "checkpoint_by_source": {name: "last" for name in retrievers},
        "catalog_movie_ids": (10, 20, 30, 40),
        "queries": queries,
        "depth": 2,
    }


def test_cached_and_uncached_partition_streams_are_exact(tmp_path: Path) -> None:
    retrievers = _bank(known=True)
    queries = (PoolQuery(1, 1, (10,)), PoolQuery(2, 2, (20,)))
    options = _options(retrievers, queries)
    plain = list(partition_pools(None, mode="known_user", **options))
    assert [query.key for query, _ in plain] == [1, 2]
    assert set(plain[0][1]) == set(retrievers)
    assert retrievers["lightgcn"].calls == [(1, (10,), 2), (2, (20,), 2)]

    cache = PoolCache(tmp_path, max_bytes=1_000_000, min_free_bytes=0, shard_queries=1)
    first = partition_pools(cache, mode="known_user", **options)
    assert all(info.key and not info.cache_hit for info in first.cache_info.values())
    assert list(first) == plain
    for retriever in retrievers.values():
        retriever.fail = True
    second = partition_pools(cache, mode="known_user", **options)
    assert all(info.cache_hit for info in second.cache_info.values())
    assert list(second) == plain


def test_history_only_bank_has_no_subject_identity_and_rejects_bad_order(tmp_path: Path) -> None:
    retrievers = _bank(known=False)
    queries = (PoolQuery("inner-0001", None, (10,)), PoolQuery("inner-0002", None, (20,)))
    options = _options(retrievers, queries)
    options["partition"] = "inner:0"
    rows = list(partition_pools(None, mode="history_only", **options))
    assert all(set(pools) == {"popularity", "itemknn", "multivae"} for _, pools in rows)
    assert all(
        subject_id is None for source in retrievers.values() for subject_id, _, _ in source.calls
    )

    cache = PoolCache(tmp_path, max_bytes=1_000_000, min_free_bytes=0)
    assert list(partition_pools(cache, mode="history_only", **options)) == rows
    with pytest.raises(ValueError, match="deterministic key order"):
        partition_pools(None, mode="history_only", **{**options, "queries": queries[::-1]})
    with pytest.raises(ValueError, match="cannot carry Subject identity"):
        partition_pools(
            None,
            mode="history_only",
            **{**options, "queries": (PoolQuery("inner-0001", 1, (10,)),)},
        )
    with pytest.raises(ValueError, match="source bank"):
        partition_pools(None, mode="history_only", **_options(_bank(known=True), queries))


def test_source_model_fingerprint_invalidates_only_that_source(tmp_path: Path) -> None:
    retrievers = _bank(known=True)
    queries = (PoolQuery(1, 1, (10,)),)
    options = _options(retrievers, queries)
    cache = PoolCache(tmp_path, max_bytes=1_000_000, min_free_bytes=0)
    first = partition_pools(cache, mode="known_user", **options)
    assert list(first)

    changed_models = {**options["model_fingerprints"], "itemknn": "new-model-state-sha"}
    changed = partition_pools(
        cache, mode="known_user", **{**options, "model_fingerprints": changed_models}
    )
    assert not changed.cache_info["itemknn"].cache_hit
    assert changed.cache_info["itemknn"].key != first.cache_info["itemknn"].key
    for name in ("popularity", "multivae", "lightgcn"):
        assert changed.cache_info[name].cache_hit
        assert changed.cache_info[name].key == first.cache_info[name].key
    assert list(changed) == list(first)


def test_mismatched_source_stream_query_is_rejected() -> None:
    class _BadResult:
        key = "bad-key"
        cache_hit = True
        stored_depth = 2

        def iter_pools(self) -> Any:
            yield PoolQuery(9, 9, ()), (Candidate(10, 1.0, 1),)

    class _BadCache:
        def iter_or_build(self, *_args: Any, **_kwargs: Any) -> _BadResult:
            return _BadResult()

    retrievers = _bank(known=True)
    options = _options(retrievers, (PoolQuery(1, 1, ()),))
    stream = partition_pools(_BadCache(), mode="known_user", **options)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Query order differs"):
        list(stream)


def test_missing_model_fingerprint_and_test_partition_are_rejected() -> None:
    retrievers = _bank(known=True)
    options = _options(retrievers, (PoolQuery(1, 1, ()),))
    with pytest.raises(ValueError, match="fitted-model content fingerprint"):
        partition_pools(
            None,
            mode="known_user",
            **{**options, "model_fingerprints": {**options["model_fingerprints"], "itemknn": ""}},
        )
    with pytest.raises(ValueError, match="inner and validation"):
        partition_pools(None, mode="known_user", **{**options, "partition": "test"})


def test_retriever_state_hash_excludes_lazy_scorer_fields() -> None:
    @dataclass
    class _State:
        name: str
        weights: tuple[float, ...]
        lazy_scorer: object | None = field(default=None, init=False)

    first = _State("multivae", (0.1234567890123456,))
    digest = retriever_state_sha256(first)
    first.lazy_scorer = object()
    assert retriever_state_sha256(first) == digest
    second = _State("multivae", (0.1234567890123457,))
    assert retriever_state_sha256(second) != digest
