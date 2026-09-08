from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from recsys.core.hashing import canonical_json, digest_file, digest_token, digest_value, normalize
from recsys.core.io import read_json, write_json, write_jsonl
from recsys.core.paths import WorkspacePaths
from recsys.core.registry import Registry


@dataclass
class Example:
    path: Path
    values: set[int]


def test_canonical_hashing_normalizes_structures(tmp_path: Path) -> None:
    left = {"b": Example(Path("a/b"), {3, 1}), "a": [2, 1]}
    right = {"a": [2, 1], "b": {"path": "a/b", "values": [1, 3]}}
    assert normalize(left) == right
    assert canonical_json(left) == canonical_json(right)
    assert digest_value(left) == digest_value(right)
    with pytest.raises(ValueError, match="non-finite"):
        digest_value(float("nan"))
    with pytest.raises(TypeError, match="cannot normalize"):
        normalize(object())

    payload = tmp_path / "payload"
    payload.write_bytes(b"content")
    assert digest_file(payload).startswith("sha256:")
    assert len(digest_token(digest_file(payload))) == 64
    with pytest.raises(ValueError, match="sha256"):
        digest_token("bad")


def test_atomic_json_helpers_and_workspace_paths(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "value.json"
    write_json(target, {"b": 2, "a": 1})
    assert read_json(target) == {"a": 1, "b": 2}
    jsonl = tmp_path / "events.jsonl"
    write_jsonl(jsonl, [{"id": 1}, {"id": 2}])
    assert jsonl.read_text().count("\n") == 2
    paths = WorkspacePaths.from_value(tmp_path).ensure()
    assert paths.raw_data.is_dir()
    assert paths.models.is_dir()
    assert WorkspacePaths.from_value().root == Path.cwd().resolve()


def test_registry_is_deterministic_and_rejects_duplicates() -> None:
    registry: Registry[int] = Registry("example")
    registry.register("second-value", 2)
    registry.register("first", 1)
    assert registry.names() == ("first", "second_value")
    assert registry.get("second-value") == 2
    assert "FIRST" in registry
    assert list(registry) == [("first", 1), ("second_value", 2)]
    with pytest.raises(ValueError, match="duplicate"):
        registry.register("first", 3)
    with pytest.raises(ValueError, match="empty"):
        registry.register(" ", 0)
    with pytest.raises(KeyError, match="available"):
        registry.get("missing")
