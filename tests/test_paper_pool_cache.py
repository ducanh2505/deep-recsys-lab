from __future__ import annotations

import fcntl
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from deep_recsys_lifecycle.models import Candidate
from deep_recsys_lifecycle.paper_pool_cache import (
    PoolCache,
    PoolCacheCapacityError,
    PoolCacheCorruptionError,
    PoolQuery,
    PoolScope,
)


def _scope(**changes: object) -> PoolScope:
    base = PoolScope(
        mode="known_user",
        partition="validation",
        cohort_fingerprint="cohort-source-split-sha",
        fit_fingerprint="ordered-fit-events-sha",
        model_fingerprint="fitted-state-sha",
        code_fingerprint="source-and-scoring-code-sha",
        source_name="itemknn",
        source_config={"neighbors": 20, "similarity": "cosine"},
        seed=42,
        device="cpu",
        checkpoint_policy="last",
        catalog_movie_ids=(10, 20, 30, 40),
    )
    return replace(base, **changes)


def _queries() -> tuple[PoolQuery, ...]:
    return (
        PoolQuery(3, 3, (30,)),
        PoolQuery(1, 1, (10,)),
        PoolQuery(2, 2, (20,)),
    )


def _score(query: PoolQuery, depth: int) -> tuple[Candidate, ...]:
    values = [movie_id for movie_id in (10, 20, 30, 40) if movie_id not in query.history]
    return tuple(
        Candidate(movie_id, math.nextafter(1.0 / movie_id, math.inf), rank)
        for rank, movie_id in enumerate(values[:depth], 1)
    )


def _cache(root: Path, **changes: int) -> PoolCache:
    return PoolCache(root, max_bytes=1_000_000, min_free_bytes=0, shard_queries=2, **changes)


def test_streamed_shards_preserve_order_scores_and_cache_hits(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    built = cache.iter_or_build(_scope(), _queries(), 3, _score)
    assert not built.cache_hit
    assert [query.key for query, _ in built.iter_pools()] == [1, 2, 3]
    assert len(list(tmp_path.rglob("shard-*.parquet"))) == 2
    expected = {query.key: _score(query, 3) for query in _queries()}
    assert dict((query.key, pool) for query, pool in built.iter_pools()) == expected

    def fail_if_scored(_query: PoolQuery, _depth: int) -> tuple[Candidate, ...]:
        raise AssertionError("exact cache hit must not rescore")

    hit = cache.iter_or_build(_scope(), reversed(_queries()), 3, fail_if_scored)
    assert hit.cache_hit
    assert dict((query.key, pool) for query, pool in hit.iter_pools()) == expected


def test_every_score_input_changes_the_content_key(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    baseline = cache.iter_or_build(_scope(), _queries(), 2, _score).key
    variants = (
        (
            _scope(mode="history_only", source_name="itemknn"),
            tuple(PoolQuery(str(query.key), None, query.history) for query in _queries()),
        ),
        (_scope(partition="inner:0"), _queries()),
        (_scope(cohort_fingerprint="another-cohort"), _queries()),
        (_scope(fit_fingerprint="another-fit"), _queries()),
        (_scope(model_fingerprint="another-model"), _queries()),
        (_scope(code_fingerprint="another-code"), _queries()),
        (_scope(source_name="popularity"), _queries()),
        (_scope(source_config={"neighbors": 40}), _queries()),
        (_scope(seed=43), _queries()),
        (_scope(device="mps"), _queries()),
        (_scope(checkpoint_policy="best_validation"), _queries()),
        (_scope(catalog_movie_ids=(10, 20, 30, 40, 50)), _queries()),
        (_scope(), _queries()[:2]),
        (
            _scope(),
            (
                PoolQuery(3, 3, (20,)),
                *_queries()[1:],
            ),
        ),
    )
    keys = {baseline}
    for scope, queries in variants:
        result = cache.iter_or_build(scope, queries, 2, _score)
        assert not result.cache_hit
        assert result.key not in keys
        keys.add(result.key)
    assert cache.iter_or_build(_scope(), _queries(), 1, _score).key not in keys


def test_deeper_pool_is_used_only_after_exact_per_query_prefix_verification(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)
    cache.iter_or_build(_scope(), _queries(), 3, _score)
    calls: list[int] = []

    def verified_score(query: PoolQuery, depth: int) -> tuple[Candidate, ...]:
        calls.append(query.key)
        return _score(query, depth)

    result = cache.iter_or_build(_scope(), _queries(), 2, verified_score, verify_deeper_prefix=True)
    assert not result.cache_hit
    assert result.stored_depth == 3
    assert calls == [1, 2, 3]
    assert all(pool == _score(query, 2) for query, pool in result.iter_pools())
    assert len(list(tmp_path.rglob("shard-*.parquet"))) == 2

    reused = cache.iter_or_build(
        _scope(),
        _queries(),
        2,
        lambda _query, _depth: pytest.fail("verified prefix should be reused"),
    )
    assert reused.cache_hit
    assert reused.stored_depth == 3


def test_non_prefix_scoring_builds_its_own_depth(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.iter_or_build(_scope(), _queries(), 3, _score)

    def depth_dependent_score(query: PoolQuery, depth: int) -> tuple[Candidate, ...]:
        pool = _score(query, depth)
        return tuple(Candidate(item.movie_id, item.score + depth, item.rank) for item in pool)

    shallow = cache.iter_or_build(
        _scope(), _queries(), 2, depth_dependent_score, verify_deeper_prefix=True
    )
    assert shallow.stored_depth == 2
    assert all(pool == depth_dependent_score(query, 2) for query, pool in shallow.iter_pools())
    assert len(list(tmp_path.rglob("shard-*.parquet"))) == 4


def test_corrupt_shard_and_manifest_are_rejected(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.iter_or_build(_scope(), _queries(), 2, _score)
    shard = next(tmp_path.rglob("shard-*.parquet"))
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(PoolCacheCorruptionError):
        cache.iter_or_build(_scope(), _queries(), 2, _score)

    separate = _cache(tmp_path / "manifest-case")
    separate.iter_or_build(_scope(), _queries(), 2, _score)
    manifest = next((tmp_path / "manifest-case").rglob("manifest.json"))
    content = json.loads(manifest.read_text(encoding="utf-8"))
    content["query_count"] += 1
    manifest.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(PoolCacheCorruptionError):
        separate.iter_or_build(_scope(), _queries(), 2, _score)


def test_test_partition_gold_config_and_history_only_identity_are_rejected(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)
    with pytest.raises(ValueError, match="only inner or validation"):
        cache.iter_or_build(_scope(partition="test"), _queries(), 2, _score)
    with pytest.raises(ValueError, match="score-affecting"):
        cache.iter_or_build(_scope(source_config={"gold_movie_ids": [20]}), _queries(), 2, _score)
    with pytest.raises(ValueError, match="cannot carry Subject identity"):
        cache.iter_or_build(_scope(mode="history_only"), _queries(), 2, _score)
    with pytest.raises(ValueError, match="ineligible"):
        cache.get_or_build(
            _scope(),
            PoolQuery(7, 7, (10,)),
            1,
            lambda _query, _depth: (Candidate(10, 1.0, 1),),
        )


def test_bounded_eviction_removes_old_recomputable_pool_only(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    first = cache.iter_or_build(_scope(), _queries(), 2, _score)
    first_size = (
        sum(path.stat().st_size for path in tmp_path.rglob("*.parquet"))
        + next(tmp_path.rglob("manifest.json")).stat().st_size
    )
    bounded = PoolCache(
        tmp_path,
        max_bytes=first_size + 100,
        min_free_bytes=0,
        shard_queries=2,
    )
    second = bounded.iter_or_build(_scope(model_fingerprint="second-model"), _queries(), 2, _score)
    assert list(second.iter_pools())
    assert not list((tmp_path / "entries").glob(f"*/{first.key}"))
    assert list(first.iter_pools())
    assert not first.cache_hit  # Rebuilt after eviction between lookup and iteration.
    assert sum(path.stat().st_size for path in tmp_path.rglob("manifest.json")) < bounded.max_bytes


def test_active_alias_keeps_its_deeper_data_during_eviction(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    deep = cache.iter_or_build(_scope(), _queries(), 3, _score)
    alias = cache.iter_or_build(_scope(), _queries(), 2, _score, verify_deeper_prefix=True)
    assert alias.stored_depth == 3
    with cache._lock(alias.key, exclusive=False):
        with pytest.raises(PoolCacheCapacityError):
            cache._ensure_space(cache.max_bytes + 1, protect=frozenset())
        assert list((tmp_path / "entries").glob(f"*/{deep.key}"))


def test_capacity_failure_leaves_no_committed_partial_entry(tmp_path: Path) -> None:
    cache = PoolCache(tmp_path, max_bytes=100, min_free_bytes=0)
    with pytest.raises(PoolCacheCapacityError):
        cache.iter_or_build(_scope(), _queries(), 2, _score)
    assert not list(tmp_path.rglob("manifest.json"))


def test_stale_staging_is_reclaimed_but_active_build_is_counted(tmp_path: Path) -> None:
    cache = PoolCache(tmp_path, max_bytes=100, min_free_bytes=0)
    orphan = cache.staging / "orphan"
    orphan.mkdir()
    (orphan / ".lease").touch()
    (orphan / "partial.parquet").write_bytes(b"x" * 200)
    cache._ensure_space(0, protect=frozenset())
    assert not orphan.exists()

    active = cache.staging / "active"
    active.mkdir()
    (active / "partial.parquet").write_bytes(b"x" * 200)
    with (active / ".lease").open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        try:
            with pytest.raises(PoolCacheCapacityError):
                cache._ensure_space(0, protect=frozenset())
            assert active.exists()
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)
