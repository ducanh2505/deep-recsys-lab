"""Resumable run-local stage state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from recsys.core.io import read_json, write_json

ResultT = TypeVar("ResultT")


@dataclass(slots=True)
class RunState:
    path: Path
    value: dict[str, Any]

    @classmethod
    def open(cls, path: Path) -> RunState:
        value = cast(dict[str, Any], read_json(path)) if path.exists() else {"stages": {}}
        return cls(path, value)

    def stage(
        self,
        name: str,
        operation: Callable[[], ResultT],
        *,
        encode: Callable[[ResultT], Any] | None = None,
        decode: Callable[[Any], ResultT] | None = None,
    ) -> ResultT:
        stages = cast(dict[str, Any], self.value.setdefault("stages", {}))
        previous = stages.get(name)
        if (
            isinstance(previous, dict)
            and previous.get("status") == "complete"
            and "result" in previous
            and decode is not None
        ):
            return decode(previous["result"])
        stages[name] = {"status": "running", "started_at": datetime.now(UTC).isoformat()}
        write_json(self.path, self.value)
        try:
            result = operation()
        except BaseException as exc:
            stages[name] = {
                **cast(dict[str, Any], stages[name]),
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(self.path, self.value)
            raise
        stages[name] = {
            **cast(dict[str, Any], stages[name]),
            "status": "complete",
            "finished_at": datetime.now(UTC).isoformat(),
        }
        if encode is not None:
            cast(dict[str, Any], stages[name])["result"] = encode(result)
        write_json(self.path, self.value)
        return result
