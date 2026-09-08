"""Safe runtime loaders for every artifact representation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from recsys.core.registry import Registry
from recsys.core.types import QueryMode
from recsys.fusion import load_fusion
from recsys.fusion.base import FusionStrategy

from .manifest import ArtifactIntegrityError
from .store import LoadedArtifact, arrays_csr


@dataclass(frozen=True, slots=True)
class RuntimeQuery:
    mode: QueryMode
    vector: np.ndarray
    ordered_items: tuple[int, ...]
    user_index: int | None = None


class ArtifactRuntime(Protocol):
    def score(self, query: RuntimeQuery) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    root: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    files: dict[str, Path]
    n_users: int
    n_items: int


RuntimeLoader = Callable[[RuntimeContext], ArtifactRuntime]
RUNTIME_REGISTRY: Registry[RuntimeLoader] = Registry("runtime")


class PopularityRuntime:
    def __init__(self, context: RuntimeContext) -> None:
        self.scores = _vector(context, "global_scores", context.n_items)

    def score(self, query: RuntimeQuery) -> np.ndarray:
        return self.scores.copy()


class SparseRuntime:
    def __init__(self, context: RuntimeContext) -> None:
        self.similarity = arrays_csr(context.arrays, "similarity")
        if self.similarity.shape != (context.n_items, context.n_items):
            raise ArtifactIntegrityError("sparse runtime matrix has an incompatible shape")
        if not np.isfinite(self.similarity.data).all():
            raise ArtifactIntegrityError("sparse runtime matrix contains non-finite values")
        self.aggregation = str(context.metadata.get("aggregation", "weighted_history"))
        if self.aggregation not in {"weighted_history", "last_item"}:
            raise ArtifactIntegrityError("sparse runtime aggregation is unsupported")

    def score(self, query: RuntimeQuery) -> np.ndarray:
        if self.aggregation == "last_item" and query.ordered_items:
            return np.asarray(
                self.similarity.getrow(query.ordered_items[-1]).toarray().ravel(),
                dtype=np.float32,
            )
        return np.asarray(query.vector @ self.similarity, dtype=np.float32).ravel()


class EmbeddingRuntime:
    def __init__(self, context: RuntimeContext) -> None:
        self.users = np.asarray(context.arrays.get("user_embeddings"), dtype=np.float32)
        self.items = np.asarray(context.arrays.get("item_embeddings"), dtype=np.float32)
        if self.items.ndim != 2 or self.items.shape[0] != context.n_items:
            raise ArtifactIntegrityError("item embeddings have an incompatible shape")
        if self.users.ndim != 2 or self.users.shape[0] != context.n_users:
            raise ArtifactIntegrityError("user embeddings have an incompatible shape")
        if self.users.shape[1] != self.items.shape[1]:
            raise ArtifactIntegrityError("user and item embedding dimensions differ")
        if not np.isfinite(self.users).all() or not np.isfinite(self.items).all():
            raise ArtifactIntegrityError("embedding runtime contains non-finite values")

    def score(self, query: RuntimeQuery) -> np.ndarray:
        if query.mode is QueryMode.KNOWN_USER and query.user_index is not None:
            profile = self.users[query.user_index]
        else:
            indices = np.flatnonzero(query.vector)
            if not len(indices):
                raise ValueError("embedding history query cannot be empty")
            weights = query.vector[indices].astype(np.float32)
            denominator = float(np.abs(weights).sum()) or 1.0
            profile = (self.items[indices] * weights[:, None]).sum(axis=0) / denominator
        return np.asarray(profile @ self.items.T, dtype=np.float32)


class SequentialRuntime:
    def __init__(self, context: RuntimeContext) -> None:
        self.transition = arrays_csr(context.arrays, "similarity")
        self.last_items = np.asarray(context.arrays.get("last_items"), dtype=np.int64)
        self.fallback = _vector(context, "global_scores", context.n_items)
        if self.last_items.shape != (context.n_users,):
            raise ArtifactIntegrityError("sequential known-user state has an incompatible shape")
        if ((self.last_items < -1) | (self.last_items >= context.n_items)).any():
            raise ArtifactIntegrityError(
                "sequential known-user state contains invalid item indices"
            )

    def score(self, query: RuntimeQuery) -> np.ndarray:
        last = query.ordered_items[-1] if query.ordered_items else -1
        if query.mode is QueryMode.KNOWN_USER and query.user_index is not None:
            last = int(self.last_items[query.user_index])
        if last < 0:
            return self.fallback.copy()
        scores = self.transition.getrow(last).toarray().ravel()
        return scores if np.count_nonzero(scores) else self.fallback.copy()


class OnnxDenseRuntime:
    def __init__(self, context: RuntimeContext) -> None:
        import onnxruntime as ort

        path = context.files.get("model.onnx", context.root / "model.onnx")
        try:
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ArtifactIntegrityError(f"unable to initialize ONNX runtime: {exc}") from exc
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ArtifactIntegrityError("dense ONNX runtime requires one input and one output")
        self.input_name = str(context.metadata.get("input", inputs[0].name))
        self.output_name = str(context.metadata.get("output", outputs[0].name))
        self.n_items = context.n_items

    def score(self, query: RuntimeQuery) -> np.ndarray:
        batch = np.ascontiguousarray(query.vector.reshape(1, self.n_items), dtype=np.float32)
        try:
            output = self.session.run([self.output_name], {self.input_name: batch})[0]
        except Exception as exc:
            raise ArtifactIntegrityError(f"ONNX inference failed: {exc}") from exc
        scores = np.asarray(output, dtype=np.float32)
        if scores.shape != batch.shape or not np.isfinite(scores).all():
            raise ArtifactIntegrityError("ONNX runtime returned invalid scores")
        return np.asarray(scores[0], dtype=np.float32)


class SequentialOnnxRuntime:
    def __init__(self, context: RuntimeContext) -> None:
        import onnxruntime as ort

        path = context.files.get("model.onnx", context.root / "model.onnx")
        try:
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ArtifactIntegrityError(
                f"unable to initialize sequential ONNX runtime: {exc}"
            ) from exc
        self.input_name = str(context.metadata.get("input", "sequence"))
        self.output_name = str(context.metadata.get("output", "scores"))
        self.max_length = int(context.metadata.get("max_length", 0))
        self.known = np.asarray(context.arrays.get("known_sequences"), dtype=np.int64)
        self.n_items = context.n_items
        if self.max_length < 1 or self.known.shape != (context.n_users, self.max_length):
            raise ArtifactIntegrityError("sequential ONNX state has an incompatible shape")

    def score(self, query: RuntimeQuery) -> np.ndarray:
        if query.mode is QueryMode.KNOWN_USER and query.user_index is not None:
            sequence = self.known[query.user_index].copy()
        else:
            sequence = np.zeros(self.max_length, dtype=np.int64)
            selected = query.ordered_items[-self.max_length :]
            if selected:
                sequence[-len(selected) :] = np.asarray(selected, dtype=np.int64) + 1
        try:
            output = self.session.run(
                [self.output_name], {self.input_name: sequence.reshape(1, self.max_length)}
            )[0]
        except Exception as exc:
            raise ArtifactIntegrityError(f"sequential ONNX inference failed: {exc}") from exc
        scores = np.asarray(output, dtype=np.float32)
        if scores.shape != (1, self.n_items) or not np.isfinite(scores).all():
            raise ArtifactIntegrityError("sequential ONNX runtime returned invalid scores")
        return np.asarray(scores[0], dtype=np.float32)


class HybridRuntime:
    def __init__(self, context: RuntimeContext) -> None:
        components = context.metadata.get("components")
        if not isinstance(components, list) or not components:
            raise ArtifactIntegrityError("hybrid artifact has no components")
        self.components: list[tuple[str, ArtifactRuntime]] = []
        for component in components:
            if not isinstance(component, dict):
                raise ArtifactIntegrityError("hybrid component metadata is invalid")
            prefix = str(component["array_prefix"])
            arrays = {
                name[len(prefix) :]: value
                for name, value in context.arrays.items()
                if name.startswith(prefix)
            }
            file_map = cast(dict[str, str], component.get("files", {}))
            files = {logical: context.root / payload for logical, payload in file_map.items()}
            child = RuntimeContext(
                root=context.root,
                arrays=arrays,
                metadata=cast(dict[str, Any], component.get("metadata", {})),
                files=files,
                n_users=context.n_users,
                n_items=context.n_items,
            )
            runtime = load_runtime(str(component["runtime"]), child)
            self.components.append((str(component["name"]), runtime))
        fusion = cast(dict[str, Any], context.metadata.get("fusion", {}))
        name = str(fusion.get("name", "rrf"))
        try:
            self.fusion: FusionStrategy = load_fusion(name, fusion, context.root)
        except (KeyError, OSError, ValueError) as exc:
            raise ArtifactIntegrityError(f"unsupported hybrid fusion runtime: {name}") from exc

    def score(self, query: RuntimeQuery) -> np.ndarray:
        rankings = {name: runtime.score(query) for name, runtime in self.components}
        return np.asarray(self.fusion.fuse(rankings), dtype=np.float32)


def _vector(context: RuntimeContext, name: str, size: int) -> np.ndarray:
    value = np.asarray(context.arrays.get(name), dtype=np.float32)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ArtifactIntegrityError(f"runtime vector {name!r} has an incompatible shape")
    return value


_RUNTIMES: dict[str, RuntimeLoader] = {
    "popularity": PopularityRuntime,
    "sparse": SparseRuntime,
    "embedding": EmbeddingRuntime,
    "sequential": SequentialRuntime,
    "onnx_dense": OnnxDenseRuntime,
    "sequential_onnx": SequentialOnnxRuntime,
    "hybrid": HybridRuntime,
}
for _name, _runtime in _RUNTIMES.items():
    RUNTIME_REGISTRY.register(_name, _runtime)


def load_runtime(name: str, context: RuntimeContext) -> ArtifactRuntime:
    try:
        loader = RUNTIME_REGISTRY.get(name)
    except KeyError as exc:
        raise ArtifactIntegrityError(f"unsupported artifact runtime: {name}") from exc
    return loader(context)


def runtime_from_artifact(artifact: LoadedArtifact) -> ArtifactRuntime:
    context = RuntimeContext(
        root=artifact.root,
        arrays=artifact.arrays,
        metadata=artifact.manifest.metadata,
        files={name: artifact.root / name for name in artifact.manifest.payloads},
        n_users=len(artifact.user_ids),
        n_items=len(artifact.item_ids),
    )
    return load_runtime(artifact.manifest.runtime, context)
