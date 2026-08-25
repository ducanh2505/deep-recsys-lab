"""GroupLens-only MovieLens-20M download and checksum verification."""

from __future__ import annotations

import hashlib
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

GROUPLENS_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)


def _read_expected_md5(text: str) -> str:
    match = re.search(r"\b([0-9a-fA-F]{32})\b", text)
    if not match:
        raise ValueError("GroupLens checksum sidecar did not contain an MD5 digest")
    return match.group(1).lower()


def download_movielens20m(
    destination_dir: str | Path,
    *,
    url: str = GROUPLENS_URL,
    force: bool = False,
) -> dict[str, str | int]:
    """Download the archive and GroupLens MD5 sidecar, then verify both hashes."""

    destination = Path(destination_dir)
    archive = destination / "ml-20m.zip"
    md5_path = destination / "ml-20m.zip.md5"
    if force or not archive.exists():
        _download(url, archive)
    if force or not md5_path.exists():
        _download(url + ".md5", md5_path)
    expected_md5 = _read_expected_md5(md5_path.read_text(encoding="utf-8"))
    actual_md5 = _hash_file(archive, "md5")
    if actual_md5 != expected_md5:
        raise ValueError(f"MovieLens MD5 mismatch: expected {expected_md5}, got {actual_md5}")
    return {
        "url": url,
        "archive": str(archive),
        "size_bytes": archive.stat().st_size,
        "md5": actual_md5,
        "sha256": _hash_file(archive, "sha256"),
    }


def extract_movielens_archive(archive_path: str | Path, destination_dir: str | Path) -> Path:
    """Safely extract the GroupLens archive and return the dataset directory."""

    archive = Path(archive_path)
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)
    extracted = destination / "ml-20m"
    return extracted if extracted.exists() else destination
