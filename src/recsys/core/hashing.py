"""Canonical hashing used by datasets, runs, and immutable artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any


def normalize(value: Any) -> Any:
    """Return a JSON-safe value with stable ordering and representations."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return normalize(dataclasses.asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): normalize(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, set | frozenset):
        return sorted((normalize(item) for item in value), key=canonical_json)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [normalize(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite floats cannot be hashed")
        return value
    if value is None or isinstance(value, bool | int | str):
        return value
    if hasattr(value, "item"):
        return normalize(value.item())
    raise TypeError(f"cannot normalize value of type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a normalized value without insignificant whitespace."""

    return json.dumps(normalize(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def digest_value(value: Any) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def digest_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def digest_token(value: str) -> str:
    """Return the filesystem-safe hexadecimal part of a SHA-256 identifier."""

    prefix, separator, token = value.partition(":")
    if prefix != "sha256" or separator != ":" or len(token) != 64:
        raise ValueError("expected a sha256:<digest> identifier")
    return token
