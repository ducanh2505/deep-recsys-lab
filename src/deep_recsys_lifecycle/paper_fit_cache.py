"""Bounded, content-addressed train-only retriever fits for paper screening."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

from .event_store import DataSnapshot
from .retriever import CandidateRetriever

T = TypeVar("T", bound=CandidateRetriever)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FitCache:
    """Reuse unchanged source fits without persisting labels or test queries."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = 4 * 1024**3,
        min_free_bytes: int = 8 * 1024**3,
    ) -> None:
        if max_bytes < 1 or min_free_bytes < 0:
            raise ValueError("fit cache limits must be positive")
        self.root = Path(root)
        self.entries = self.root / "entries"
        self.entries.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    def get_or_fit(
        self,
        *,
        source: str,
        snapshot: DataSnapshot,
        source_config: Mapping[str, object],
        seed: int,
        device: str,
        code_fingerprint: str,
        fit: Callable[[], T],
    ) -> T:
        """Load an exact fit or fit once; checksum and bound every stored artifact."""

        identity = {
            "schema_version": 1,
            "source": source,
            "fit_snapshot_fingerprint": snapshot.fingerprint,
            "fit_event_count": snapshot.event_count,
            "source_config": dict(source_config),
            "seed": seed,
            "device": device,
            "code_fingerprint": code_fingerprint,
        }
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        entry = self.entries / key
        # One lock also protects readers from eviction by another screening process.
        lock_path = self.root / ".fit-cache.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            artifact = entry / "retriever.pkl"
            manifest = entry / "manifest.json"
            if artifact.is_file() and manifest.is_file():
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                if metadata.get("identity") != identity or metadata.get(
                    "artifact_sha256"
                ) != _hash_file(artifact):
                    raise ValueError("stored retriever fit failed integrity verification")
                with artifact.open("rb") as handle:
                    retriever = pickle.load(handle)  # noqa: S301 - checksum-verified local fit
                if retriever.name != source:
                    raise ValueError("stored retriever source differs")
                self._hits += 1
                os.utime(entry)
                return retriever  # type: ignore[no-any-return]

            fitted = fit()
            if fitted.name != source:
                raise ValueError("fitted retriever source differs")
            descriptor, temp_name = tempfile.mkstemp(prefix=f".{key}.", dir=self.root)
            temp_path = Path(temp_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    pickle.dump(fitted, handle, protocol=5)
                    handle.flush()
                    os.fsync(handle.fileno())
                artifact_size = temp_path.stat().st_size
                self._make_room(artifact_size)
                temp_entry = Path(tempfile.mkdtemp(prefix=f".{key}.staging.", dir=self.root))
                try:
                    os.replace(temp_path, temp_entry / "retriever.pkl")
                    (temp_entry / "manifest.json").write_text(
                        json.dumps(
                            {
                                "identity": identity,
                                "artifact_sha256": _hash_file(temp_entry / "retriever.pkl"),
                            },
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    os.replace(temp_entry, entry)
                finally:
                    if temp_entry.exists():
                        for child in temp_entry.iterdir():
                            child.unlink()
                        temp_entry.rmdir()
            finally:
                temp_path.unlink(missing_ok=True)
            self._misses += 1
            return fitted

    def _make_room(self, incoming_bytes: int) -> None:
        if incoming_bytes > self.max_bytes:
            raise OSError("one retriever fit exceeds the fit cache limit")
        entries = sorted(
            (path for path in self.entries.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
        )
        used = sum(
            child.stat().st_size
            for entry in entries
            for child in entry.iterdir()
            if child.is_file()
        )
        free = os.statvfs(self.root).f_bavail * os.statvfs(self.root).f_frsize
        for entry in entries:
            if (
                used + incoming_bytes <= self.max_bytes
                and free - incoming_bytes >= self.min_free_bytes
            ):
                break
            size = sum(child.stat().st_size for child in entry.iterdir() if child.is_file())
            for child in entry.iterdir():
                child.unlink()
            entry.rmdir()
            used -= size
            free += size
        if used + incoming_bytes > self.max_bytes or free - incoming_bytes < self.min_free_bytes:
            raise OSError("fit cache cannot preserve its size and free-space reserves")
