from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .event_store import DataSnapshot
from .models import PositiveInteraction
from .popularity import PopularityModel, fit_popularity


class ServingArtifact:
    """An immutable, self-describing Popularity serving bundle."""

    REQUIRED_MANIFEST_KEYS = {
        "artifact_version",
        "data_snapshot_fingerprint",
        "event_count",
        "positive_interaction_count",
        "implicit_signal_threshold",
        "retriever",
        "query_modes",
        "model_sha256",
        "manifest_sha256",
    }

    def __init__(self, path: Path, manifest: dict[str, Any], model: PopularityModel) -> None:
        self.path = path
        self.manifest = manifest
        self.model = model

    @classmethod
    def build(
        cls,
        path: Path,
        snapshot: DataSnapshot,
        interactions: tuple[PositiveInteraction, ...],
        threshold: float = 4.0,
    ) -> ServingArtifact:
        model = fit_popularity(snapshot, interactions)
        return cls(
            path=path,
            manifest={
                "artifact_version": 1,
                "data_snapshot_fingerprint": snapshot.fingerprint,
                "event_count": snapshot.event_count,
                "positive_interaction_count": len(interactions),
                "implicit_signal_threshold": threshold,
                "retriever": "popularity",
                "query_modes": ["known_user", "history_only", "empty_history"],
            },
            model=model,
        )

    def save(self) -> None:
        if self.path.exists():
            raise FileExistsError(f"Serving Artifact path already exists: {self.path}")

        self.path.mkdir(parents=True)
        model_payload = json.dumps(
            self.model.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        model_sha256 = hashlib.sha256(model_payload).hexdigest()
        manifest = {**self.manifest, "model_sha256": model_sha256}
        manifest["manifest_sha256"] = self._manifest_checksum(manifest)
        (self.path / "model.json").write_bytes(model_payload + b"\n")
        (self.path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.manifest = manifest

    @classmethod
    def load(cls, path: Path) -> ServingArtifact:
        manifest_path = path / "manifest.json"
        model_path = path / "model.json"
        if not manifest_path.is_file() or not model_path.is_file():
            raise ValueError(f"Serving Artifact is incomplete: {path}")

        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_value = json.loads(model_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_value, dict) or not isinstance(model_value, dict):
            raise ValueError("Serving Artifact files must contain JSON objects")
        missing = cls.REQUIRED_MANIFEST_KEYS - manifest_value.keys()
        if missing:
            raise ValueError(f"Serving Artifact manifest is missing: {sorted(missing)}")
        if cls._manifest_checksum(manifest_value) != manifest_value["manifest_sha256"]:
            raise ValueError("Serving Artifact manifest checksum does not match its contents")

        model_bytes = json.dumps(model_value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(model_bytes).hexdigest() != manifest_value["model_sha256"]:
            raise ValueError("Serving Artifact model checksum does not match its manifest")

        return cls(path=path, manifest=manifest_value, model=PopularityModel.from_dict(model_value))

    @staticmethod
    def _manifest_checksum(manifest: dict[str, Any]) -> str:
        unsigned_manifest = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        payload = json.dumps(unsigned_manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()
