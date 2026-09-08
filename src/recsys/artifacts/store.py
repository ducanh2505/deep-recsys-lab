"""Artifact source of truth independent from any serving framework."""

from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from pydantic import ValidationError

from recsys.core.hashing import digest_file, digest_token, digest_value, normalize
from recsys.core.io import read_json, write_json
from recsys.core.types import Identifier, QueryMode
from recsys.datasets.mapping import decode_identifier, encode_identifier
from recsys.datasets.types import PreparedDataset

from .manifest import ArtifactIntegrityError, ArtifactManifest, PayloadEntry
from .npz import read_npz, write_npz


@dataclass(slots=True)
class RuntimeSpec:
    plugin: str
    runtime: str
    capabilities: tuple[QueryMode, ...]
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    root: Path
    manifest: ArtifactManifest
    arrays: dict[str, np.ndarray]
    user_ids: tuple[Identifier, ...]
    item_ids: tuple[Identifier, ...]
    item_metadata: tuple[dict[str, Any], ...]


def csr_arrays(name: str, matrix: sp.csr_matrix) -> dict[str, np.ndarray]:
    value = matrix.tocsr(copy=False)
    return {
        f"{name}_data": value.data.astype(np.float32, copy=False),
        f"{name}_indices": value.indices.astype(np.int64, copy=False),
        f"{name}_indptr": value.indptr.astype(np.int64, copy=False),
        f"{name}_shape": np.asarray(value.shape, dtype=np.int64),
    }


def arrays_csr(arrays: dict[str, np.ndarray], name: str) -> sp.csr_matrix:
    required = {f"{name}_{suffix}" for suffix in ("data", "indices", "indptr", "shape")}
    if not required.issubset(arrays):
        raise ArtifactIntegrityError(f"artifact is missing sparse matrix {name!r}")
    shape_values = arrays[f"{name}_shape"].astype(np.int64).tolist()
    if len(shape_values) != 2:
        raise ArtifactIntegrityError(f"sparse matrix {name!r} has an invalid shape")
    return sp.csr_matrix(
        (
            arrays[f"{name}_data"],
            arrays[f"{name}_indices"],
            arrays[f"{name}_indptr"],
        ),
        shape=(int(shape_values[0]), int(shape_values[1])),
    )


def _safe_metadata(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        return normalize(value)
    except TypeError:
        return str(value)


def _catalog(dataset: PreparedDataset) -> dict[str, Any]:
    metadata = dataset.item_metadata
    lookup: dict[Identifier, dict[str, Any]] = {}
    if not metadata.empty and "item_id" in metadata:
        for record in metadata.to_dict(orient="records"):
            item_id = cast(Identifier, record.pop("item_id"))
            lookup[item_id] = {str(key): _safe_metadata(value) for key, value in record.items()}
    return {
        "users": [encode_identifier(value) for value in dataset.user_ids],
        "items": [
            {"id": encode_identifier(value), "metadata": lookup.get(value, {})}
            for value in dataset.item_ids
        ],
    }


def _identity_basis(
    spec: RuntimeSpec,
    dataset: PreparedDataset,
    config: dict[str, Any],
    payloads: dict[str, PayloadEntry],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "plugin": spec.plugin,
        "runtime": spec.runtime,
        "capabilities": [mode.value for mode in spec.capabilities],
        "dataset_digest": dataset.dataset_digest,
        "config": config,
        "metadata": normalize(spec.metadata),
        "payloads": {name: entry.sha256 for name, entry in sorted(payloads.items())},
    }


def write_artifact(
    parent: Path,
    spec: RuntimeSpec,
    dataset: PreparedDataset,
    config: dict[str, Any],
) -> LoadedArtifact:
    plugin_parent = parent / spec.plugin
    plugin_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".artifact-", dir=plugin_parent))
    try:
        arrays = {**csr_arrays("train", dataset.train), **spec.arrays}
        write_npz(temporary / "payload.npz", arrays)
        write_json(temporary / "catalog.json", _catalog(dataset))
        for name, source in sorted(spec.files.items()):
            if Path(name).name != name or name in {"payload.npz", "catalog.json", "manifest.json"}:
                raise ValueError(f"invalid artifact payload name: {name}")
            shutil.copyfile(source, temporary / name)
        payload_names = ["payload.npz", "catalog.json", *sorted(spec.files)]
        payloads = {
            name: PayloadEntry(
                sha256=digest_file(temporary / name), bytes=(temporary / name).stat().st_size
            )
            for name in payload_names
        }
        normalized_config = cast(dict[str, Any], normalize(config))
        artifact_id = digest_value(_identity_basis(spec, dataset, normalized_config, payloads))
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            created_at=datetime.now(UTC),
            plugin=spec.plugin,
            runtime=spec.runtime,
            capabilities=list(spec.capabilities),
            dataset_digest=dataset.dataset_digest,
            config=normalized_config,
            payloads=payloads,
            schema={
                "interaction": {
                    "required": ["user_id", "item_id"],
                    "optional": ["value", "timestamp"],
                },
                "recommendation": ["item_id", "score", "metadata"],
            },
            catalog_size=len(dataset.item_ids),
            metadata=cast(dict[str, Any], normalize(spec.metadata)),
        )
        write_json(temporary / "manifest.json", manifest.model_dump(mode="json", by_alias=True))
        target = plugin_parent / digest_token(artifact_id)
        if target.exists():
            shutil.rmtree(temporary)
            return load_artifact(target)
        temporary.replace(target)
        return load_artifact(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_manifest(root: Path) -> ArtifactManifest:
    try:
        return ArtifactManifest.model_validate(read_json(root / "manifest.json"))
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        raise ArtifactIntegrityError("artifact manifest is missing or invalid") from exc


def load_artifact(root: Path, *, verify: bool = True) -> LoadedArtifact:
    root = root.expanduser().resolve()
    manifest = _read_manifest(root)
    if root.name != digest_token(manifest.artifact_id):
        raise ArtifactIntegrityError("artifact directory does not match artifact_id")
    if verify:
        for name, entry in manifest.payloads.items():
            if Path(name).name != name:
                raise ArtifactIntegrityError("artifact manifest contains an unsafe payload path")
            payload = root / name
            if not payload.is_file() or payload.stat().st_size != entry.bytes:
                raise ArtifactIntegrityError(f"artifact payload is missing or truncated: {name}")
            if digest_file(payload) != entry.sha256:
                raise ArtifactIntegrityError(f"artifact payload checksum mismatch: {name}")
        expected_id = digest_value(
            {
                "schema_version": manifest.schema_version,
                "plugin": manifest.plugin,
                "runtime": manifest.runtime,
                "capabilities": [mode.value for mode in manifest.capabilities],
                "dataset_digest": manifest.dataset_digest,
                "config": manifest.config,
                "metadata": manifest.metadata,
                "payloads": {
                    name: entry.sha256 for name, entry in sorted(manifest.payloads.items())
                },
            }
        )
        if expected_id != manifest.artifact_id:
            raise ArtifactIntegrityError("artifact identity verification failed")
    try:
        arrays = read_npz(root / "payload.npz")
        catalog = cast(dict[str, Any], read_json(root / "catalog.json"))
        user_ids = tuple(decode_identifier(value) for value in catalog["users"])
        item_ids = tuple(decode_identifier(value["id"]) for value in catalog["items"])
        metadata = tuple(
            cast(dict[str, Any], value.get("metadata", {})) for value in catalog["items"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError("artifact catalog or numeric payload is invalid") from exc
    if len(item_ids) != manifest.catalog_size:
        raise ArtifactIntegrityError("artifact catalog size does not match its manifest")
    train = arrays_csr(arrays, "train")
    if train.shape != (len(user_ids), len(item_ids)):
        raise ArtifactIntegrityError("artifact interaction matrix has an incompatible shape")
    return LoadedArtifact(root, manifest, arrays, user_ids, item_ids, metadata)
