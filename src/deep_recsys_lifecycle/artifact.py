from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .fusion import LearnedHybridFusion, untrained_fusion_classifier
from .serving import (
    ItemKNNRetriever,
    LightGCNRetriever,
    MultVAERetriever,
    PopularityRetriever,
)

if TYPE_CHECKING:
    from .event_store import DataSnapshot
    from .models import PositiveInteraction, RatingEvent


ARTIFACT_SCHEMA_VERSION = 1
SUPPORTED_QUERY_MODES = ("known_user", "history_only", "empty_history")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ArtifactValidationError(ValueError):
    """Raised when a Serving Artifact cannot be trusted by the runtime."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for artifact identity and integrity hashes."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value cannot be represented as canonical JSON") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_configuration_checksum(configuration: Mapping[str, Any]) -> str:
    """Hash configuration independently of dictionary insertion order."""

    return sha256_bytes(canonical_json_bytes(dict(configuration)))


def deterministic_dataset_checksum(events: Iterable[RatingEvent]) -> str:
    """Hash source events independent of source path, order, and dict serialization order."""

    canonical_events = [event.to_dict() for event in events]
    canonical_events.sort(key=canonical_json_bytes)
    return sha256_bytes(canonical_json_bytes(canonical_events))


def resolve_source_revision(cwd: Path | None = None) -> str:
    """Resolve the source revision without baking a repository-specific SHA into code."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = completed.stdout.strip()
    return revision or "unknown"


@dataclass(frozen=True, slots=True)
class PayloadDescriptor:
    path: str
    role: str
    sha256: str

    @classmethod
    def from_dict(cls, value: object) -> PayloadDescriptor:
        if not isinstance(value, Mapping):
            raise ArtifactValidationError("payload inventory entries must be objects")
        path = _required_string(value, "path")
        role = _required_string(value, "role")
        checksum = _required_string(value, "sha256")
        _validate_sha256(checksum, f"payload {path!r} checksum")
        return cls(path=path, role=role, sha256=checksum)

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "role": self.role, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Typed, validated view of the JSON manifest contract."""

    artifact_schema_version: int
    artifact_version: int
    artifact_id: str
    lifecycle_stage: str
    data_cutoff_percentage: int
    source_event_count: int
    event_count: int
    positive_interaction_count: int
    implicit_signal_threshold: float
    dataset_checksum: str
    data_snapshot_fingerprint: str
    retriever: str
    query_modes: tuple[str, ...]
    configuration: dict[str, Any]
    configuration_sha256: str
    random_seed: int
    training_device: str
    serving_device: str
    multivae_training: dict[str, Any]
    lightgcn_training: dict[str, Any]
    fusion_training: dict[str, Any]
    timings: dict[str, float]
    evaluation_metrics: dict[str, Any]
    payload_inventory: tuple[PayloadDescriptor, ...]
    source_revision: str
    model_sha256: str
    bundle_integrity_sha256: str
    manifest_sha256: str
    _raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)

    @classmethod
    def from_dict(cls, value: object) -> ArtifactManifest:
        if not isinstance(value, Mapping):
            raise ArtifactValidationError("Serving Artifact manifest must be a JSON object")

        schema_version = _required_int(value, "artifact_schema_version")
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactValidationError(f"unsupported artifact schema version: {schema_version}")
        artifact_version = _required_int(value, "artifact_version")
        if artifact_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactValidationError(f"unsupported artifact version: {artifact_version}")

        required = {
            "artifact_id",
            "lifecycle_stage",
            "data_cutoff_percentage",
            "source_event_count",
            "event_count",
            "positive_interaction_count",
            "implicit_signal_threshold",
            "dataset_checksum",
            "data_snapshot_fingerprint",
            "retriever",
            "query_modes",
            "configuration",
            "configuration_sha256",
            "random_seed",
            "training_device",
            "serving_device",
            "timings",
            "evaluation_metrics",
            "payload_inventory",
            "source_revision",
            "model_sha256",
            "bundle_integrity_sha256",
            "manifest_sha256",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ArtifactValidationError(f"Serving Artifact manifest is missing: {missing}")

        artifact_id = _required_string(value, "artifact_id")
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise ArtifactValidationError("artifact_id must be a safe relative directory name")
        lifecycle_stage = _required_string(value, "lifecycle_stage")
        data_cutoff = _required_int(value, "data_cutoff_percentage")
        if not 0 <= data_cutoff <= 100:
            raise ArtifactValidationError("data_cutoff_percentage must be between 0 and 100")

        source_event_count = _nonnegative_int(value, "source_event_count")
        event_count = _nonnegative_int(value, "event_count")
        positive_count = _nonnegative_int(value, "positive_interaction_count")
        if event_count > source_event_count:
            raise ArtifactValidationError("event_count cannot exceed source_event_count")
        if positive_count > event_count:
            raise ArtifactValidationError("positive_interaction_count cannot exceed event_count")
        threshold = _required_number(value, "implicit_signal_threshold")
        dataset_checksum = _required_string(value, "dataset_checksum")
        snapshot_fingerprint = _required_string(value, "data_snapshot_fingerprint")
        _validate_sha256(dataset_checksum, "dataset_checksum")
        _validate_sha256(snapshot_fingerprint, "data_snapshot_fingerprint")

        retriever = _required_string(value, "retriever")
        if retriever != "popularity":
            raise ArtifactValidationError(f"unsupported serving retriever: {retriever}")
        raw_query_modes = value["query_modes"]
        if (
            not isinstance(raw_query_modes, list)
            or any(not isinstance(mode, str) for mode in raw_query_modes)
            or set(raw_query_modes) != set(SUPPORTED_QUERY_MODES)
            or len(raw_query_modes) != len(SUPPORTED_QUERY_MODES)
        ):
            raise ArtifactValidationError("manifest query_modes are incompatible with serving")

        raw_configuration = value["configuration"]
        if not isinstance(raw_configuration, Mapping):
            raise ArtifactValidationError("manifest configuration must be an object")
        configuration = dict(raw_configuration)
        configuration_checksum = _required_string(value, "configuration_sha256")
        _validate_sha256(configuration_checksum, "configuration_sha256")
        if configuration.get("retriever") != retriever:
            raise ArtifactValidationError("manifest configuration is incompatible with retriever")
        itemknn_enabled = configuration.get("itemknn_enabled", False)
        if not isinstance(itemknn_enabled, bool):
            raise ArtifactValidationError("manifest itemknn_enabled must be boolean")

        random_seed = _required_int(value, "random_seed")
        training_device = _required_string(value, "training_device")
        serving_device = _required_string(value, "serving_device")
        if serving_device != "cpu":
            raise ArtifactValidationError("unsupported serving device: expected cpu")
        if configuration.get("serving_device") != serving_device:
            raise ArtifactValidationError(
                "manifest configuration is incompatible with serving device"
            )
        if (
            configuration.get("data_cutoff_percentage") != data_cutoff
            or configuration.get("random_seed") != random_seed
            or configuration.get("training_device") != training_device
            or configuration.get("implicit_signal_threshold") != threshold
            or configuration.get("query_modes") != raw_query_modes
        ):
            raise ArtifactValidationError("manifest configuration is incompatible with serving")

        raw_multivae_training = value.get("multivae_training", {})
        if configuration.get("multivae_enabled") and "multivae_training" not in value:
            raise ArtifactValidationError("manifest is missing Mult-VAE training provenance")
        if raw_multivae_training and configuration.get("multivae_enabled") is not True:
            raise ArtifactValidationError("manifest Mult-VAE configuration is incomplete")
        if configuration.get("multivae_enabled") is True and not isinstance(
            configuration.get("multivae"), Mapping
        ):
            raise ArtifactValidationError("manifest Mult-VAE configuration is missing")
        if not isinstance(raw_multivae_training, Mapping):
            raise ArtifactValidationError("manifest multivae_training must be an object")
        if raw_multivae_training:
            _validate_multivae_training(raw_multivae_training, random_seed)
        multivae_training = dict(raw_multivae_training)

        raw_lightgcn_training = value.get("lightgcn_training", {})
        if configuration.get("lightgcn_enabled") and "lightgcn_training" not in value:
            raise ArtifactValidationError("manifest is missing LightGCN training provenance")
        if raw_lightgcn_training and configuration.get("lightgcn_enabled") is not True:
            raise ArtifactValidationError("manifest LightGCN configuration is incomplete")
        if configuration.get("lightgcn_enabled") is True and not isinstance(
            configuration.get("lightgcn"), Mapping
        ):
            raise ArtifactValidationError("manifest LightGCN configuration is missing")
        if not isinstance(raw_lightgcn_training, Mapping):
            raise ArtifactValidationError("manifest lightgcn_training must be an object")
        if raw_lightgcn_training:
            _validate_lightgcn_training(raw_lightgcn_training, random_seed)
        lightgcn_training = dict(raw_lightgcn_training)

        fusion_enabled = configuration.get("fusion_enabled", False)
        if not isinstance(fusion_enabled, bool):
            raise ArtifactValidationError("manifest fusion_enabled must be boolean")
        if fusion_enabled:
            for mode in ("known_user_fusion", "history_only_fusion"):
                if not isinstance(configuration.get(mode), Mapping):
                    raise ArtifactValidationError(f"manifest configuration is missing {mode}")
        raw_fusion_training = value.get("fusion_training", {})
        if fusion_enabled and "fusion_training" not in value:
            raise ArtifactValidationError("manifest is missing fusion training provenance")
        if not isinstance(raw_fusion_training, Mapping):
            raise ArtifactValidationError("manifest fusion_training must be an object")
        if fusion_enabled and set(raw_fusion_training) != {"known_user", "history_only"}:
            raise ArtifactValidationError("manifest fusion training must contain both query modes")
        fusion_training = {
            str(mode): dict(metadata)
            for mode, metadata in raw_fusion_training.items()
            if isinstance(mode, str) and isinstance(metadata, Mapping)
        }

        raw_timings = value["timings"]
        if not isinstance(raw_timings, Mapping):
            raise ArtifactValidationError("manifest timings must be an object")
        timings = {
            key: _nonnegative_number(raw_timings, key) for key in ("training", "export", "smoke")
        }
        for key, _raw_timing in raw_timings.items():
            if key not in timings:
                timings[str(key)] = _nonnegative_number(raw_timings, key)

        raw_metrics = value["evaluation_metrics"]
        if not isinstance(raw_metrics, Mapping):
            raise ArtifactValidationError("manifest evaluation_metrics must be an object")
        evaluation_metrics = dict(raw_metrics)
        _validate_metric_tree(evaluation_metrics, "evaluation_metrics")

        raw_inventory = value["payload_inventory"]
        if not isinstance(raw_inventory, list) or not raw_inventory:
            raise ArtifactValidationError("payload_inventory must be a non-empty list")
        payload_inventory = tuple(PayloadDescriptor.from_dict(item) for item in raw_inventory)
        if len({payload.path for payload in payload_inventory}) != len(payload_inventory):
            raise ArtifactValidationError("payload_inventory contains duplicate paths")

        source_revision = _required_string(value, "source_revision")
        model_sha256 = _required_string(value, "model_sha256")
        bundle_integrity_sha256 = _required_string(value, "bundle_integrity_sha256")
        manifest_sha256 = _required_string(value, "manifest_sha256")
        _validate_sha256(model_sha256, "model_sha256")
        _validate_sha256(bundle_integrity_sha256, "bundle_integrity_sha256")
        _validate_sha256(manifest_sha256, "manifest_sha256")

        return cls(
            artifact_schema_version=schema_version,
            artifact_version=artifact_version,
            artifact_id=artifact_id,
            lifecycle_stage=lifecycle_stage,
            data_cutoff_percentage=data_cutoff,
            source_event_count=source_event_count,
            event_count=event_count,
            positive_interaction_count=positive_count,
            implicit_signal_threshold=threshold,
            dataset_checksum=dataset_checksum,
            data_snapshot_fingerprint=snapshot_fingerprint,
            retriever=retriever,
            query_modes=tuple(raw_query_modes),
            configuration=configuration,
            configuration_sha256=configuration_checksum,
            random_seed=random_seed,
            training_device=training_device,
            serving_device=serving_device,
            multivae_training=multivae_training,
            lightgcn_training=lightgcn_training,
            fusion_training=fusion_training,
            timings=timings,
            evaluation_metrics=evaluation_metrics,
            payload_inventory=payload_inventory,
            source_revision=source_revision,
            model_sha256=model_sha256,
            bundle_integrity_sha256=bundle_integrity_sha256,
            manifest_sha256=manifest_sha256,
            _raw=dict(value),
        )

    def to_dict(self) -> dict[str, Any]:
        if self._raw:
            return dict(self._raw)
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "artifact_version": self.artifact_version,
            "artifact_id": self.artifact_id,
            "lifecycle_stage": self.lifecycle_stage,
            "data_cutoff_percentage": self.data_cutoff_percentage,
            "source_event_count": self.source_event_count,
            "event_count": self.event_count,
            "positive_interaction_count": self.positive_interaction_count,
            "implicit_signal_threshold": self.implicit_signal_threshold,
            "dataset_checksum": self.dataset_checksum,
            "data_snapshot_fingerprint": self.data_snapshot_fingerprint,
            "retriever": self.retriever,
            "query_modes": list(self.query_modes),
            "configuration": self.configuration,
            "configuration_sha256": self.configuration_sha256,
            "random_seed": self.random_seed,
            "training_device": self.training_device,
            "serving_device": self.serving_device,
            "multivae_training": self.multivae_training,
            "lightgcn_training": self.lightgcn_training,
            "fusion_training": self.fusion_training,
            "timings": self.timings,
            "evaluation_metrics": self.evaluation_metrics,
            "payload_inventory": [item.to_dict() for item in self.payload_inventory],
            "source_revision": self.source_revision,
            "model_sha256": self.model_sha256,
            "bundle_integrity_sha256": self.bundle_integrity_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


def validate_manifest(value: object) -> ArtifactManifest:
    """Validate manifest types, supported configuration, and internal checksums."""

    manifest = ArtifactManifest.from_dict(value)
    raw = manifest.to_dict()
    if _manifest_checksum(raw) != manifest.manifest_sha256:
        raise ArtifactValidationError(
            "Serving Artifact manifest checksum does not match its contents"
        )
    if canonical_configuration_checksum(manifest.configuration) != manifest.configuration_sha256:
        raise ArtifactValidationError("manifest configuration checksum does not match its contents")
    if _bundle_integrity_checksum(raw) != manifest.bundle_integrity_sha256:
        raise ArtifactValidationError("Serving Artifact bundle integrity checksum is invalid")
    return manifest


class ServingArtifact:
    """An immutable, self-describing CPU serving bundle."""

    def __init__(
        self,
        path: Path,
        manifest: dict[str, Any],
        model: PopularityRetriever,
        itemknn: ItemKNNRetriever | None = None,
        multivae: MultVAERetriever | None = None,
        lightgcn: LightGCNRetriever | None = None,
        known_user_fusion: LearnedHybridFusion | None = None,
        history_only_fusion: LearnedHybridFusion | None = None,
    ) -> None:
        self.path = path
        self.manifest = manifest
        self.model = model
        self.itemknn = itemknn
        self.multivae = multivae
        self.lightgcn = lightgcn
        self.known_user_fusion = known_user_fusion
        self.history_only_fusion = history_only_fusion

    @property
    def artifact_id(self) -> str:
        return str(self.manifest["artifact_id"])

    @property
    def artifact_fingerprint(self) -> str:
        identity = {
            key: value
            for key, value in self.manifest.items()
            if key not in {"timings", "bundle_integrity_sha256", "manifest_sha256"}
        }
        return sha256_bytes(canonical_json_bytes(identity))

    @classmethod
    def build(
        cls,
        path: Path,
        snapshot: DataSnapshot,
        interactions: tuple[PositiveInteraction, ...],
        threshold: float = 4.0,
        *,
        source_events: Iterable[RatingEvent] | None = None,
        lifecycle_stage: str = "fast",
        data_cutoff_percentage: int = 50,
        configuration: Mapping[str, Any] | None = None,
        random_seed: int = 42,
        training_device: str | None = None,
        serving_device: str = "cpu",
        timings: Mapping[str, float] | None = None,
        evaluation_metrics: Mapping[str, Any] | None = None,
        source_revision: str | None = None,
        itemknn_model: ItemKNNRetriever | None = None,
        multivae_model: MultVAERetriever | None = None,
        lightgcn_model: LightGCNRetriever | None = None,
        known_user_fusion: LearnedHybridFusion | None = None,
        history_only_fusion: LearnedHybridFusion | None = None,
        manifest_metadata: Mapping[str, Any] | None = None,
    ) -> ServingArtifact:
        # Keep fitting behind the export seam; the runtime loader imports only serving.py.
        from .itemknn import fit_itemknn
        from .multivae import fit_multivae
        from .popularity import fit_popularity

        model = fit_popularity(snapshot, interactions)
        fitted_itemknn = itemknn_model or fit_itemknn(snapshot, interactions).to_serving()
        fitted_multivae = multivae_model or fit_multivae(snapshot, interactions, seed=random_seed)
        fitted_known_user_fusion = known_user_fusion or untrained_fusion_classifier(
            "known_user", source_snapshot_fingerprint=snapshot.fingerprint
        )
        fitted_history_only_fusion = history_only_fusion or untrained_fusion_classifier(
            "history_only", source_snapshot_fingerprint=snapshot.fingerprint
        )
        multivae_metadata = dict(fitted_multivae.training_metadata)
        # Preserve the #36 top-level field as the CPU artifact/runtime device.  The actual
        # neural training device is recorded precisely in multivae_training.actual_device.
        resolved_training_device = training_device or "cpu"
        source_event_values = tuple(source_events) if source_events is not None else snapshot.events
        resolved_configuration = {
            "profile": "fast",
            "retriever": "popularity",
            "data_cutoff_percentage": data_cutoff_percentage,
            "future_percentage": 60,
            "implicit_signal_threshold": threshold,
            "candidate_pool_limit": 200,
            "query_modes": list(SUPPORTED_QUERY_MODES),
            "random_seed": random_seed,
            "training_device": resolved_training_device,
            "serving_device": serving_device,
            "multivae_enabled": True,
            "multivae": dict(fitted_multivae.configuration),
            "lightgcn_enabled": lightgcn_model is not None,
            "itemknn_enabled": True,
            "fusion_enabled": True,
            "known_user_fusion": fitted_known_user_fusion.configuration,
            "history_only_fusion": fitted_history_only_fusion.configuration,
            **dict(configuration or {}),
        }
        if lightgcn_model is not None:
            resolved_configuration["lightgcn_enabled"] = True
            resolved_configuration["lightgcn"] = dict(lightgcn_model.configuration)
        configuration_checksum = canonical_configuration_checksum(resolved_configuration)
        source_revision_value = source_revision or resolve_source_revision()
        identity = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "lifecycle_stage": lifecycle_stage,
            "data_cutoff_percentage": data_cutoff_percentage,
            "dataset_checksum": deterministic_dataset_checksum(source_event_values),
            "data_snapshot_fingerprint": snapshot.fingerprint,
            "configuration_sha256": configuration_checksum,
            "random_seed": random_seed,
            "source_revision": source_revision_value,
            "retriever": "popularity",
        }
        artifact_id = f"artifact-{sha256_bytes(canonical_json_bytes(identity))[:24]}"
        resolved_timings = {
            "training": 0.0,
            "export": 0.0,
            "smoke": 0.0,
            **dict(timings or {}),
        }
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "lifecycle_stage": lifecycle_stage,
            "data_cutoff_percentage": data_cutoff_percentage,
            "source_event_count": len(source_event_values),
            # event_count remains the snapshot count for compatibility with Issue #35.
            "event_count": snapshot.event_count,
            "positive_interaction_count": len(interactions),
            "implicit_signal_threshold": threshold,
            "dataset_checksum": identity["dataset_checksum"],
            "data_snapshot_fingerprint": snapshot.fingerprint,
            "retriever": "popularity",
            "query_modes": list(SUPPORTED_QUERY_MODES),
            "configuration": resolved_configuration,
            "configuration_sha256": configuration_checksum,
            "random_seed": random_seed,
            "training_device": resolved_training_device,
            "serving_device": serving_device,
            "multivae_training": multivae_metadata,
            "lightgcn_training": (
                dict(lightgcn_model.training_metadata) if lightgcn_model is not None else {}
            ),
            "fusion_training": {
                "known_user": dict(fitted_known_user_fusion.training_metadata),
                "history_only": dict(fitted_history_only_fusion.training_metadata),
            },
            "timings": resolved_timings,
            "evaluation_metrics": dict(evaluation_metrics or {}),
            "payload_inventory": [],
            "source_revision": source_revision_value,
        }
        for key, value in (manifest_metadata or {}).items():
            if key in manifest:
                raise ValueError(f"manifest metadata cannot replace required field: {key}")
            manifest[key] = value
        return cls(
            path=path,
            manifest=manifest,
            model=model,
            itemknn=fitted_itemknn,
            multivae=fitted_multivae,
            lightgcn=lightgcn_model,
            known_user_fusion=fitted_known_user_fusion,
            history_only_fusion=fitted_history_only_fusion,
        )

    def save(self) -> None:
        """Write one new bundle; an existing directory is never modified."""

        if self.path.exists():
            raise FileExistsError(f"Serving Artifact path already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.mkdir()
        try:
            export_started = monotonic()
            payload: dict[str, object] = {
                "payload_schema_version": 4,
                "popularity": self.model.to_dict(),
                "itemknn": self.itemknn.to_dict() if self.itemknn is not None else None,
                "multivae": self.multivae.to_dict() if self.multivae is not None else None,
                "fusion": {
                    "known_user": (
                        self.known_user_fusion.to_dict()
                        if self.known_user_fusion is not None
                        else None
                    ),
                    "history_only": (
                        self.history_only_fusion.to_dict()
                        if self.history_only_fusion is not None
                        else None
                    ),
                },
            }
            payload["lightgcn"] = self.lightgcn.to_dict() if self.lightgcn is not None else None
            model_payload = canonical_json_bytes(payload)
            model_sha256 = sha256_bytes(model_payload)
            manifest = {**self.manifest}
            manifest["timings"] = {
                **dict(manifest["timings"]),
                "export": monotonic() - export_started,
            }
            manifest["model_sha256"] = model_sha256
            manifest["payload_inventory"] = [
                {
                    "path": "model.json",
                    "role": "popularity_retriever",
                    "sha256": model_sha256,
                }
            ]
            manifest["bundle_integrity_sha256"] = _bundle_integrity_checksum(manifest)
            manifest["manifest_sha256"] = _manifest_checksum(manifest)
            validate_manifest(manifest)
            _write_exclusive(self.path / "model.json", model_payload)
            _write_exclusive(
                self.path / "manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
                + b"\n",
            )
            self.manifest = manifest
        except BaseException:
            shutil.rmtree(self.path, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path: Path) -> ServingArtifact:
        """Load one explicit artifact directory, or an artifact store's active pointer."""

        if path.is_file() and path.name == "active.json":
            return ArtifactStore(path.parent).load_active()
        if (
            path.is_dir()
            and not (path / "manifest.json").is_file()
            and (path / "active.json").is_file()
        ):
            return ArtifactStore(path).load_active()
        if not path.is_dir():
            raise ArtifactValidationError(f"Serving Artifact directory does not exist: {path}")

        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ArtifactValidationError(f"Serving Artifact is incomplete: {path}")
        try:
            manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactValidationError("Serving Artifact manifest is unreadable") from error
        typed_manifest = validate_manifest(manifest_value)
        manifest = typed_manifest.to_dict()

        descriptors = {
            descriptor.path: descriptor for descriptor in typed_manifest.payload_inventory
        }
        payload_root = path.resolve()
        payload_paths = {
            descriptor.path: _safe_payload_path(payload_root, descriptor.path)
            for descriptor in typed_manifest.payload_inventory
        }
        model_descriptor = descriptors.get("model.json")
        if model_descriptor is None or model_descriptor.role != "popularity_retriever":
            raise ArtifactValidationError("Serving Artifact has no compatible popularity payload")
        if model_descriptor.sha256 != typed_manifest.model_sha256:
            raise ArtifactValidationError("model_sha256 does not match payload inventory")

        for descriptor in typed_manifest.payload_inventory:
            payload_path = payload_paths[descriptor.path]
            if not payload_path.is_file():
                raise ArtifactValidationError(
                    f"Serving Artifact payload is missing: {descriptor.path}"
                )
            if sha256_bytes(payload_path.read_bytes()) != descriptor.sha256:
                raise ArtifactValidationError(
                    f"Serving Artifact payload checksum does not match: {descriptor.path}"
                )

        model_path = _safe_payload_path(payload_root, model_descriptor.path)
        try:
            model_value = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactValidationError("Serving Artifact model payload is unreadable") from error
        if not isinstance(model_value, Mapping):
            raise ArtifactValidationError("Serving Artifact model payload must be a JSON object")
        try:
            payload_schema_version = model_value.get("payload_schema_version")
            if "payload_schema_version" in model_value:
                if payload_schema_version not in {2, 3, 4}:
                    raise ValueError("unsupported composite payload schema")
                raw_popularity = model_value.get("popularity")
                raw_multivae = model_value.get("multivae")
                if not isinstance(raw_popularity, Mapping) or not isinstance(raw_multivae, Mapping):
                    raise ValueError("composite payload is missing Popularity or Mult-VAE state")
                model = PopularityRetriever.from_dict(raw_popularity)
                multivae = MultVAERetriever.from_dict(raw_multivae)
                raw_lightgcn = model_value.get("lightgcn")
                if payload_schema_version in {3, 4}:
                    if payload_schema_version == 4 and raw_lightgcn is None:
                        lightgcn = None
                    elif not isinstance(raw_lightgcn, Mapping):
                        raise ValueError("composite payload is missing LightGCN state")
                    else:
                        lightgcn = LightGCNRetriever.from_dict(raw_lightgcn)
                else:
                    lightgcn = None
                if payload_schema_version == 4:
                    raw_itemknn = model_value.get("itemknn")
                    raw_fusion = model_value.get("fusion")
                    if not isinstance(raw_itemknn, Mapping) or not isinstance(raw_fusion, Mapping):
                        raise ValueError("composite payload is missing ItemKNN or fusion state")
                    raw_known_fusion = raw_fusion.get("known_user")
                    raw_history_fusion = raw_fusion.get("history_only")
                    if not isinstance(raw_known_fusion, Mapping) or not isinstance(
                        raw_history_fusion, Mapping
                    ):
                        raise ValueError("composite payload is missing both fusion classifiers")
                    itemknn = ItemKNNRetriever.from_dict(raw_itemknn)
                    known_user_fusion = LearnedHybridFusion.from_dict(raw_known_fusion)
                    history_only_fusion = LearnedHybridFusion.from_dict(raw_history_fusion)
                else:
                    itemknn = None
                    known_user_fusion = None
                    history_only_fusion = None
            else:
                model = PopularityRetriever.from_dict(model_value)
                itemknn = None
                multivae = None
                lightgcn = None
                known_user_fusion = None
                history_only_fusion = None
        except ValueError as error:
            raise ArtifactValidationError(
                f"Serving Artifact model payload is invalid: {error}"
            ) from error
        if multivae is not None:
            manifest_multivae_configuration = manifest.get("configuration", {}).get("multivae")
            if not isinstance(manifest_multivae_configuration, Mapping) or dict(
                manifest_multivae_configuration
            ) != dict(multivae.configuration):
                raise ArtifactValidationError(
                    "Serving Artifact Mult-VAE configuration is incompatible with its manifest"
                )
            manifest_training = manifest.get("multivae_training")
            if not isinstance(manifest_training, Mapping) or dict(manifest_training) != dict(
                multivae.training_metadata
            ):
                raise ArtifactValidationError(
                    "Serving Artifact Mult-VAE provenance is incompatible with its manifest"
                )
        if manifest.get("configuration", {}).get("multivae_enabled") and multivae is None:
            raise ArtifactValidationError("Serving Artifact is missing its Mult-VAE payload")
        if lightgcn is not None:
            manifest_lightgcn_configuration = manifest.get("configuration", {}).get("lightgcn")
            if not isinstance(manifest_lightgcn_configuration, Mapping) or dict(
                manifest_lightgcn_configuration
            ) != dict(lightgcn.configuration):
                raise ArtifactValidationError(
                    "Serving Artifact LightGCN configuration is incompatible with its manifest"
                )
            manifest_training = manifest.get("lightgcn_training")
            if not isinstance(manifest_training, Mapping) or dict(manifest_training) != dict(
                lightgcn.training_metadata
            ):
                raise ArtifactValidationError(
                    "Serving Artifact LightGCN provenance is incompatible with its manifest"
                )
        if manifest.get("configuration", {}).get("lightgcn_enabled") and lightgcn is None:
            raise ArtifactValidationError("Serving Artifact is missing its LightGCN payload")
        if manifest.get("configuration", {}).get("itemknn_enabled") and itemknn is None:
            raise ArtifactValidationError("Serving Artifact is missing its ItemKNN payload")
        if manifest.get("configuration", {}).get("fusion_enabled"):
            if known_user_fusion is None or history_only_fusion is None:
                raise ArtifactValidationError("Serving Artifact is missing its fusion classifiers")
            manifest_known_configuration = manifest.get("configuration", {}).get(
                "known_user_fusion"
            )
            manifest_history_configuration = manifest.get("configuration", {}).get(
                "history_only_fusion"
            )
            if (
                not isinstance(manifest_known_configuration, Mapping)
                or dict(manifest_known_configuration) != known_user_fusion.configuration
                or not isinstance(manifest_history_configuration, Mapping)
                or dict(manifest_history_configuration) != history_only_fusion.configuration
            ):
                raise ArtifactValidationError(
                    "Serving Artifact fusion configuration is incompatible"
                )
            raw_fusion_training = manifest.get("fusion_training")
            if not isinstance(raw_fusion_training, Mapping):
                raise ArtifactValidationError("Serving Artifact fusion provenance is missing")
            if dict(raw_fusion_training.get("known_user", {})) != dict(
                known_user_fusion.training_metadata
            ) or dict(raw_fusion_training.get("history_only", {})) != dict(
                history_only_fusion.training_metadata
            ):
                raise ArtifactValidationError("Serving Artifact fusion provenance is incompatible")
        return cls(
            path=path,
            manifest=manifest,
            model=model,
            itemknn=itemknn,
            multivae=multivae,
            lightgcn=lightgcn,
            known_user_fusion=known_user_fusion,
            history_only_fusion=history_only_fusion,
        )

    def health_metadata(self) -> dict[str, object]:
        return {
            "status": "ok",
            "artifact_id": self.artifact_id,
            "artifact_fingerprint": self.artifact_fingerprint,
            "data_snapshot_fingerprint": self.manifest["data_snapshot_fingerprint"],
            "retriever": self.manifest["retriever"],
            "query_modes": self.manifest["query_modes"],
            "serving_device": self.manifest["serving_device"],
            "multivae_loaded": self.multivae is not None,
            "lightgcn_loaded": self.lightgcn is not None,
            "itemknn_loaded": self.itemknn is not None,
            "known_user_fusion_loaded": self.known_user_fusion is not None,
            "history_only_fusion_loaded": self.history_only_fusion is not None,
        }


class ArtifactStore:
    """A local immutable artifact store with an atomic active pointer."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def active_pointer_path(self) -> Path:
        return self.root / "active.json"

    @property
    def staging_root(self) -> Path:
        return self.root / ".staging"

    def new_staging_path(self) -> Path:
        self.staging_root.mkdir(parents=True, exist_ok=True)
        return self.staging_root / f"artifact-{uuid4().hex}"

    def update_staging_timings(
        self,
        staging_path: Path,
        timings: Mapping[str, float],
    ) -> ServingArtifact:
        """Record measured phase timings while a bundle is still mutable staging state."""

        staging = self._validated_staging_path(staging_path)
        artifact = ServingArtifact.load(staging)
        manifest = {**artifact.manifest}
        updated_timings = {**dict(manifest["timings"])}
        for name, value in timings.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ArtifactValidationError(f"timing {name!r} must be a non-negative number")
            updated_timings[name] = float(value)
        manifest["timings"] = updated_timings
        manifest["bundle_integrity_sha256"] = _bundle_integrity_checksum(manifest)
        manifest["manifest_sha256"] = _manifest_checksum(manifest)
        validate_manifest(manifest)

        temporary = staging / f".manifest-{uuid4().hex}.tmp"
        try:
            _write_exclusive(
                temporary,
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
                + b"\n",
            )
            os.replace(temporary, staging / "manifest.json")
        finally:
            temporary.unlink(missing_ok=True)
        return ServingArtifact.load(staging)

    def publish(self, staging_path: Path) -> Path:
        """Validate and atomically rename a completed staging directory into the store."""

        staging = self._validated_staging_path(staging_path)

        artifact = ServingArtifact.load(staging)
        final_path = self.root / artifact.artifact_id
        if final_path.exists():
            raise FileExistsError(f"immutable artifact already exists: {artifact.artifact_id}")
        os.rename(staging, final_path)
        return final_path

    def discard_staging(self, staging_path: Path) -> None:
        staging = self._validated_staging_path(staging_path)
        if staging.exists():
            shutil.rmtree(staging)

    def _validated_staging_path(self, staging_path: Path) -> Path:
        staging = staging_path.resolve()
        staging_root = self.staging_root.resolve()
        try:
            relative = staging.relative_to(staging_root)
        except ValueError as error:
            raise ArtifactValidationError(
                "staging artifact must be inside the artifact store"
            ) from error
        if len(relative.parts) != 1:
            raise ArtifactValidationError("staging artifact must be a direct staging child")
        return staging

    def activate(
        self,
        artifact: Path | str,
        *,
        smoke_validator: Callable[[ServingArtifact], None] | None = None,
    ) -> ServingArtifact:
        """Validate, smoke-test, and atomically point active.json at one artifact."""

        loaded = self.load(artifact)
        if smoke_validator is None:
            from .validation import validate_smoke_queries

            smoke_validator = validate_smoke_queries
        smoke_validator(loaded)
        self._write_active_pointer(loaded)
        return loaded

    def _activate_loaded(self, artifact: Path | str) -> ServingArtifact:
        """Activate an artifact already smoke-tested by the lifecycle."""

        loaded = self.load(artifact)
        self._write_active_pointer(loaded)
        return loaded

    def load(self, artifact: Path | str) -> ServingArtifact:
        if isinstance(artifact, str):
            if not _ARTIFACT_ID_RE.fullmatch(artifact):
                raise ArtifactValidationError("artifact id is not a safe relative path")
            artifact_path = self.root / artifact
        else:
            artifact_path = artifact
        resolved = artifact_path.resolve()
        try:
            relative = resolved.relative_to(self.root.resolve())
        except ValueError as error:
            raise ArtifactValidationError("artifact must be inside the artifact store") from error
        if len(relative.parts) != 1 or not _ARTIFACT_ID_RE.fullmatch(relative.parts[0]):
            raise ArtifactValidationError("artifact path must name one immutable artifact")
        loaded = ServingArtifact.load(resolved)
        if loaded.artifact_id != relative.parts[0]:
            raise ArtifactValidationError(
                "artifact directory name does not match manifest artifact_id"
            )
        return loaded

    def load_active(self) -> ServingArtifact:
        pointer_path = self.active_pointer_path
        if not pointer_path.is_file():
            raise ArtifactValidationError(f"active artifact pointer does not exist: {pointer_path}")
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactValidationError("active artifact pointer is unreadable") from error
        if not isinstance(pointer, dict) or set(pointer) != {"artifact_id"}:
            raise ArtifactValidationError("active artifact pointer has an invalid shape")
        artifact_id = pointer["artifact_id"]
        if not isinstance(artifact_id, str) or not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise ArtifactValidationError("active artifact pointer contains an invalid artifact id")
        artifact_path = self.root / artifact_id
        if not artifact_path.is_dir():
            raise ArtifactValidationError(
                f"active artifact pointer targets a missing artifact: {artifact_id}"
            )
        return self.load(artifact_path)

    def _write_active_pointer(self, artifact: ServingArtifact) -> None:
        final_path = self.root / artifact.artifact_id
        if not final_path.is_dir():
            raise ArtifactValidationError("cannot activate an unpublished artifact")
        temporary = self.root / f".active-{uuid4().hex}.tmp"
        payload = canonical_json_bytes({"artifact_id": artifact.artifact_id}) + b"\n"
        try:
            _write_exclusive(temporary, payload)
            os.replace(temporary, self.active_pointer_path)
        finally:
            temporary.unlink(missing_ok=True)


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _required_string(value: Mapping[object, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise ArtifactValidationError(f"manifest field {key!r} must be a non-empty string")
    return raw


def _required_int(value: Mapping[object, object], key: str) -> int:
    raw = value.get(key)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ArtifactValidationError(f"manifest field {key!r} must be an integer")
    return raw


def _nonnegative_int(value: Mapping[object, object], key: str) -> int:
    result = _required_int(value, key)
    if result < 0:
        raise ArtifactValidationError(f"manifest field {key!r} cannot be negative")
    return result


def _required_number(value: Mapping[object, object], key: str) -> float:
    return _number_value(value.get(key), f"manifest field {key!r}")


def _nonnegative_number(value: Mapping[object, object], key: object) -> float:
    result = _number_value(value.get(key), f"timing {key!r}")
    if result < 0:
        raise ArtifactValidationError(f"timing {key!r} cannot be negative")
    return result


def _number_value(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ArtifactValidationError(f"{label} must be a finite number")
    return float(value)


def _validate_metric_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_metric_tree(child, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_metric_tree(child, f"{label}[{index}]")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ArtifactValidationError(f"{label} must contain only finite numeric values")


def _validate_multivae_training(value: Mapping[object, object], random_seed: int) -> None:
    required = {
        "requested_device",
        "actual_device",
        "duration_seconds",
        "seed",
        "fallback_reason",
        "hyperparameters",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ArtifactValidationError(f"manifest multivae_training is missing: {missing}")
    requested_device = value.get("requested_device")
    actual_device = value.get("actual_device")
    if requested_device not in {"cpu", "mps"}:
        raise ArtifactValidationError("manifest Mult-VAE requested_device is invalid")
    if actual_device not in {"cpu", "mps"}:
        raise ArtifactValidationError("manifest Mult-VAE actual_device is invalid")
    if _number_value(value.get("duration_seconds"), "Mult-VAE duration_seconds") < 0:
        raise ArtifactValidationError("Mult-VAE duration_seconds cannot be negative")
    seed = value.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed != random_seed:
        raise ArtifactValidationError("manifest Mult-VAE seed is incompatible with random_seed")
    fallback_reason = value.get("fallback_reason")
    if fallback_reason is not None and (
        not isinstance(fallback_reason, str) or not fallback_reason
    ):
        raise ArtifactValidationError("manifest Mult-VAE fallback_reason must be null or text")
    if not isinstance(value.get("hyperparameters"), Mapping):
        raise ArtifactValidationError("manifest Mult-VAE hyperparameters must be an object")


def _validate_lightgcn_training(value: Mapping[object, object], random_seed: int) -> None:
    required = {
        "requested_device",
        "actual_device",
        "duration_seconds",
        "seed",
        "fallback_reason",
        "hyperparameters",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ArtifactValidationError(f"manifest LightGCN training is missing: {missing}")
    requested_device = value.get("requested_device")
    actual_device = value.get("actual_device")
    if requested_device not in {"cpu", "mps"}:
        raise ArtifactValidationError("manifest LightGCN requested_device is invalid")
    if actual_device not in {"cpu", "mps"}:
        raise ArtifactValidationError("manifest LightGCN actual_device is invalid")
    if _number_value(value.get("duration_seconds"), "LightGCN duration_seconds") < 0:
        raise ArtifactValidationError("LightGCN duration_seconds cannot be negative")
    seed = value.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed != random_seed:
        raise ArtifactValidationError("manifest LightGCN seed is incompatible with random_seed")
    fallback_reason = value.get("fallback_reason")
    if fallback_reason is not None and (
        not isinstance(fallback_reason, str) or not fallback_reason
    ):
        raise ArtifactValidationError("manifest LightGCN fallback_reason must be null or text")
    if not isinstance(value.get("hyperparameters"), Mapping):
        raise ArtifactValidationError("manifest LightGCN hyperparameters must be an object")
    benchmark_seconds = value.get("benchmark_seconds", {})
    if not isinstance(benchmark_seconds, Mapping):
        raise ArtifactValidationError("manifest LightGCN benchmark_seconds must be an object")
    _validate_metric_tree(benchmark_seconds, "LightGCN benchmark_seconds")


def _validate_sha256(value: str, label: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ArtifactValidationError(f"{label} must be a lowercase SHA-256 checksum")


def _bundle_integrity_checksum(manifest: Mapping[str, Any]) -> str:
    unsigned = {
        key: value
        for key, value in manifest.items()
        if key not in {"bundle_integrity_sha256", "manifest_sha256"}
    }
    return sha256_bytes(canonical_json_bytes(unsigned))


def _manifest_checksum(manifest: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return sha256_bytes(canonical_json_bytes(unsigned))


def _safe_payload_path(root: Path, payload_name: str) -> Path:
    windows_absolute = (
        len(payload_name) >= 3 and payload_name[1] == ":" and payload_name[2] in "/\\"
    )
    if not payload_name or windows_absolute or "\\" in payload_name or "\x00" in payload_name:
        raise ArtifactValidationError(f"invalid payload path: {payload_name!r}")
    pure_path = PurePosixPath(payload_name)
    if pure_path.is_absolute() or ".." in pure_path.parts or "." in pure_path.parts:
        raise ArtifactValidationError(f"invalid payload path: {payload_name!r}")
    target = (root / Path(*pure_path.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ArtifactValidationError(
            f"payload path escapes artifact directory: {payload_name!r}"
        ) from error
    return target
