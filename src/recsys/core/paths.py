"""Workspace-relative paths for all generated state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    root: Path

    @classmethod
    def from_value(cls, value: str | Path | None = None) -> WorkspacePaths:
        root = Path.cwd() if value is None else Path(value)
        return cls(root.expanduser().resolve())

    @property
    def var(self) -> Path:
        return self.root / "var"

    @property
    def raw_data(self) -> Path:
        return self.var / "data" / "raw"

    @property
    def prepared_data(self) -> Path:
        return self.var / "data" / "prepared"

    @property
    def runs(self) -> Path:
        return self.var / "runs"

    @property
    def models(self) -> Path:
        return self.var / "models"

    @property
    def caches(self) -> Path:
        return self.var / "cache"

    def ensure(self) -> WorkspacePaths:
        for path in (self.raw_data, self.prepared_data, self.runs, self.models, self.caches):
            path.mkdir(parents=True, exist_ok=True)
        return self
