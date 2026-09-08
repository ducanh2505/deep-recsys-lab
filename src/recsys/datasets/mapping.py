"""Strict identifiers and deterministic index mappings."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from recsys.core.types import Identifier


def validate_identifier(value: Any, *, field: str) -> Identifier:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{field} cannot be boolean")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{field} must be a non-empty string or integer")


def identifier_key(value: Identifier) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def deterministic_mapping(values: Iterable[Any], *, field: str) -> tuple[Identifier, ...]:
    validated = {validate_identifier(value, field=field) for value in values}
    return tuple(sorted(validated, key=identifier_key))


def encode_identifier(value: Identifier) -> dict[str, str | int]:
    return {"type": "integer" if isinstance(value, int) else "string", "value": value}


def decode_identifier(value: dict[str, Any]) -> Identifier:
    kind = value.get("type")
    raw = value.get("value")
    if kind == "integer" and isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if kind == "string" and isinstance(raw, str) and raw:
        return raw
    raise ValueError("invalid serialized identifier")
