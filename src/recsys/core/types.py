"""Framework-neutral domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

type Identifier = str | int


class QueryMode(StrEnum):
    """Inputs an artifact can use to produce recommendations."""

    KNOWN_USER = "known_user"
    HISTORY = "history"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ranked item emitted by a model or retrieval component."""

    item_index: int
    score: float
    metadata: dict[str, object] = field(default_factory=dict)
