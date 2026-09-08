"""Small deterministic plugin registry."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

PluginT = TypeVar("PluginT")


class Registry[PluginT]:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._plugins: dict[str, PluginT] = {}

    def register(self, name: str, plugin: PluginT) -> None:
        normalized = name.strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError(f"{self.kind} plugin name cannot be empty")
        if normalized in self._plugins:
            raise ValueError(f"duplicate {self.kind} plugin: {normalized}")
        self._plugins[normalized] = plugin

    def get(self, name: str) -> PluginT:
        normalized = name.strip().lower().replace("-", "_")
        try:
            return self._plugins[normalized]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"unknown {self.kind} plugin {name!r}; available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.replace("-", "_").lower() in self._plugins

    def __iter__(self) -> Iterator[tuple[str, PluginT]]:
        for name in self.names():
            yield name, self._plugins[name]
