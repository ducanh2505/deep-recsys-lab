"""Deterministic NPZ serialization without executable payloads."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(arrays):
                if not name.isidentifier():
                    raise ValueError(f"invalid NPZ array name: {name}")
                buffer = io.BytesIO()
                np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
                entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o600 << 16
                archive.writestr(entry, buffer.getvalue())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except Exception as exc:
        raise ValueError(f"invalid NPZ payload: {path.name}") from exc
