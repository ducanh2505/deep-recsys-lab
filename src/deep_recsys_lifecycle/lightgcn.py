from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from time import monotonic
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .event_store import DataSnapshot
from .models import PositiveInteraction
from .positive import history_movie_ids
from .serving import LightGCNRetriever


@dataclass(frozen=True, slots=True)
class LightGCNConfig:
    """Fixed fast-profile LightGCN configuration.

    The profile deliberately exposes the graph dimensions, propagation depth, training budget,
    and negative-sampling policy so the exported artifact explains exactly what was fitted.
    """

    seed: int = 42
    embedding_dim: int = 16
    layers: int = 2
    epochs: int = 8
    batch_size: int = 64
    negative_samples: int = 1
    negative_sampling: str = "uniform"
    learning_rate: float = 0.01
    regularization: float = 1e-4
    mps_slowdown_threshold: float = 1.25

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("LightGCN seed must be an integer")
        if self.embedding_dim < 1 or self.layers < 1:
            raise ValueError("LightGCN embedding_dim and layers must be positive")
        if self.epochs < 1 or self.batch_size < 1 or self.negative_samples < 1:
            raise ValueError("LightGCN epochs, batch_size, and negative_samples must be positive")
        if self.negative_sampling != "uniform":
            raise ValueError("LightGCN supports only uniform negative sampling")
        if self.learning_rate <= 0 or self.regularization < 0:
            raise ValueError(
                "LightGCN learning_rate must be positive and regularization non-negative"
            )
        if self.mps_slowdown_threshold < 1:
            raise ValueError("LightGCN mps_slowdown_threshold must be at least 1")

    def to_dict(self) -> dict[str, int | float | str]:
        return {key: value for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class _GraphData:
    subject_ids: tuple[int, ...]
    movie_ids: tuple[int, ...]
    subject_histories: dict[int, tuple[int, ...]]
    positive_edges: tuple[tuple[int, int], ...]
    source_nodes: tuple[int, ...]
    destination_nodes: tuple[int, ...]
    normalized_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _TrainedEmbeddings:
    subject_embeddings: tuple[tuple[float, ...], ...]
    movie_embeddings: tuple[tuple[float, ...], ...]


class _LightGCNNetwork(nn.Module):
    def __init__(self, graph: _GraphData, config: LightGCNConfig) -> None:
        super().__init__()
        self.subject_embedding = nn.Embedding(len(graph.subject_ids), config.embedding_dim)
        self.movie_embedding = nn.Embedding(len(graph.movie_ids), config.embedding_dim)
        nn.init.normal_(self.subject_embedding.weight, std=0.1)
        nn.init.normal_(self.movie_embedding.weight, std=0.1)
        self.layers = config.layers
        self.subject_count = len(graph.subject_ids)
        self.register_buffer(
            "source_nodes",
            torch.tensor(graph.source_nodes, dtype=torch.long),
        )
        self.register_buffer(
            "destination_nodes",
            torch.tensor(graph.destination_nodes, dtype=torch.long),
        )
        self.register_buffer(
            "normalized_weights",
            torch.tensor(graph.normalized_weights, dtype=torch.float32),
        )

    def propagate(self) -> tuple[Tensor, Tensor]:
        all_embeddings = torch.cat(
            (self.subject_embedding.weight, self.movie_embedding.weight), dim=0
        )
        layer_embeddings = [all_embeddings]
        source_nodes = cast(Tensor, self.source_nodes)
        destination_nodes = cast(Tensor, self.destination_nodes)
        normalized_weights = cast(Tensor, self.normalized_weights)
        for _layer in range(self.layers):
            messages = all_embeddings[source_nodes] * normalized_weights.unsqueeze(1)
            propagated = torch.zeros_like(all_embeddings)
            propagated.index_add_(0, destination_nodes, messages)
            all_embeddings = propagated
            layer_embeddings.append(all_embeddings)
        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return (
            final_embeddings[: self.subject_count],
            final_embeddings[self.subject_count :],
        )


def fit_lightgcn(
    snapshot: DataSnapshot,
    interactions: Iterable[PositiveInteraction],
    *,
    config: LightGCNConfig | None = None,
    seed: int | None = None,
    device_preference: str = "auto",
    mps_probe: Callable[[], bool] | None = None,
    device_benchmark: Callable[[str], float] | None = None,
) -> LightGCNRetriever:
    """Fit LightGCN on snapshot-bounded Positive Interactions.

    ``auto`` and ``mps`` request MPS.  MPS first runs a representative propagation/full-catalog
    scoring workload and is retained only when it is compatible and no slower than the declared
    threshold relative to CPU.  Every fallback reason is carried into the serving payload.
    ``mps_probe`` and ``device_benchmark`` are deterministic test seams for hosts without MPS.
    """

    if device_preference not in {"auto", "cpu", "mps"}:
        raise ValueError("LightGCN device_preference must be one of: auto, cpu, mps")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("LightGCN seed must be an integer")

    resolved_config = config or LightGCNConfig()
    resolved_seed = resolved_config.seed if seed is None else seed
    if resolved_seed != resolved_config.seed:
        resolved_config = replace(resolved_config, seed=resolved_seed)
    graph = _build_graph(snapshot, interactions)

    requested_device = "mps" if device_preference in {"auto", "mps"} else "cpu"
    fallback_reason: str | None = None
    benchmark_seconds: dict[str, float] = {}
    training_seconds = 0.0
    actual_device = "cpu"

    def train_on_device(device_name: str) -> _TrainedEmbeddings:
        nonlocal training_seconds
        fit_started = monotonic()
        try:
            return _train_once(graph, resolved_config, resolved_seed, device_name)
        finally:
            training_seconds += monotonic() - fit_started

    if requested_device == "mps":
        probe = mps_probe or _mps_is_available
        try:
            mps_available = bool(probe())
        except Exception as error:  # pragma: no cover - defensive accelerator boundary
            mps_available = False
            fallback_reason = f"MPS availability probe failed: {type(error).__name__}: {error}"
        if not mps_available and fallback_reason is None:
            fallback_reason = "MPS is unavailable on this host"

        if mps_available:
            try:
                benchmark_seconds["mps"] = _run_device_benchmark(
                    graph, resolved_config, "mps", device_benchmark
                )
                benchmark_seconds["cpu"] = _run_device_benchmark(
                    graph, resolved_config, "cpu", device_benchmark
                )
                if benchmark_seconds["mps"] > (
                    benchmark_seconds["cpu"] * resolved_config.mps_slowdown_threshold
                ):
                    fallback_reason = (
                        "MPS representative workload was slower than CPU: "
                        f"{benchmark_seconds['mps']:.6f}s vs "
                        f"{benchmark_seconds['cpu']:.6f}s "
                        f"(threshold {resolved_config.mps_slowdown_threshold:.2f}x)"
                    )
                    trained = train_on_device("cpu")
                else:
                    trained = train_on_device("mps")
                    actual_device = "mps"
            except Exception as error:  # MPS compatibility is a per-approach seam.
                if fallback_reason is None:
                    fallback_reason = (
                        f"MPS benchmark or training failed: {type(error).__name__}: {error}"
                    )
                trained = train_on_device("cpu")
        else:
            trained = train_on_device("cpu")
    else:
        trained = train_on_device("cpu")
    metadata: dict[str, Any] = {
        "requested_device": requested_device,
        "actual_device": actual_device,
        "duration_seconds": float(training_seconds),
        "seed": resolved_seed,
        "fallback_reason": fallback_reason,
        "hyperparameters": resolved_config.to_dict(),
        "benchmark_seconds": benchmark_seconds,
        "benchmark_policy": (
            "fallback when MPS representative workload exceeds CPU by "
            f"{resolved_config.mps_slowdown_threshold:.2f}x"
        ),
    }
    configuration: dict[str, Any] = {
        **resolved_config.to_dict(),
        "profile_semantics": "snapshot_positive_interaction_bipartite_graph",
        "graph_edges": "deduplicated_subject_movie_positive_interactions",
        "candidate_scoring": "full_snapshot_movie_catalog",
        "unknown_subject_policy": "reject_without_history_embedding_inference",
        "observed_movie_policy": "exclude_before_top_200",
    }
    return LightGCNRetriever(
        subject_ids=graph.subject_ids,
        movie_ids=graph.movie_ids,
        subject_histories=graph.subject_histories,
        positive_edges=graph.positive_edges,
        subject_embeddings=trained.subject_embeddings,
        movie_embeddings=trained.movie_embeddings,
        configuration=configuration,
        training_metadata=metadata,
    )


def _build_graph(
    snapshot: DataSnapshot,
    interactions: Iterable[PositiveInteraction],
) -> _GraphData:
    snapshot_event_ids = {event.event_id for event in snapshot.events}
    movie_ids = tuple(sorted({event.movie_id for event in snapshot.events}))
    if not movie_ids:
        raise ValueError("LightGCN requires a non-empty snapshot catalog")
    catalog_ids = set(movie_ids)
    positive_pairs: set[tuple[int, int]] = set()
    histories: defaultdict[int, list[PositiveInteraction]] = defaultdict(list)
    for interaction in interactions:
        if (
            interaction.event_id in snapshot_event_ids
            and interaction.movie_id in catalog_ids
            and interaction.rating >= 4.0
        ):
            positive_pairs.add((interaction.subject_id, interaction.movie_id))
            histories[interaction.subject_id].append(interaction)
    positive_edges = tuple(sorted(positive_pairs))
    if not positive_edges:
        raise ValueError("LightGCN requires at least one snapshot Positive Interaction")

    subject_ids = tuple(sorted({subject_id for subject_id, _movie_id in positive_edges}))
    subject_index = {subject_id: index for index, subject_id in enumerate(subject_ids)}
    movie_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
    subject_histories = {
        subject_id: history_movie_ids(histories[subject_id]) for subject_id in subject_ids
    }

    node_count = len(subject_ids) + len(movie_ids)
    degrees = [0] * node_count
    directed_edges: list[tuple[int, int]] = []
    for subject_id, movie_id in positive_edges:
        subject_node = subject_index[subject_id]
        movie_node = len(subject_ids) + movie_index[movie_id]
        directed_edges.extend(((subject_node, movie_node), (movie_node, subject_node)))
        degrees[subject_node] += 1
        degrees[movie_node] += 1

    source_nodes = tuple(source for source, _destination in directed_edges)
    destination_nodes = tuple(destination for _source, destination in directed_edges)
    normalized_weights = tuple(
        1.0 / math.sqrt(degrees[source] * degrees[destination])
        for source, destination in directed_edges
    )
    return _GraphData(
        subject_ids=subject_ids,
        movie_ids=movie_ids,
        subject_histories=subject_histories,
        positive_edges=positive_edges,
        source_nodes=source_nodes,
        destination_nodes=destination_nodes,
        normalized_weights=normalized_weights,
    )


def _run_device_benchmark(
    graph: _GraphData,
    config: LightGCNConfig,
    device_name: str,
    benchmark: Callable[[str], float] | None,
) -> float:
    value = (
        benchmark(device_name)
        if benchmark is not None
        else _measure_workload(graph, config, device_name)
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{device_name} benchmark must be numeric")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{device_name} benchmark must be positive and finite")
    return seconds


def _measure_workload(graph: _GraphData, config: LightGCNConfig, device_name: str) -> float:
    started = monotonic()
    device = torch.device(device_name)
    all_embeddings = torch.ones(
        (len(graph.subject_ids) + len(graph.movie_ids), config.embedding_dim),
        dtype=torch.float32,
        device=device,
    )
    source_nodes = torch.tensor(graph.source_nodes, dtype=torch.long, device=device)
    destination_nodes = torch.tensor(graph.destination_nodes, dtype=torch.long, device=device)
    normalized_weights = torch.tensor(graph.normalized_weights, dtype=torch.float32, device=device)
    for _layer in range(config.layers):
        messages = all_embeddings[source_nodes] * normalized_weights.unsqueeze(1)
        propagated = torch.zeros_like(all_embeddings)
        propagated.index_add_(0, destination_nodes, messages)
        all_embeddings = propagated
    # The serving contract scores every Movie in the catalog for a Known Subject.
    _ = all_embeddings[: len(graph.subject_ids)] @ all_embeddings[len(graph.subject_ids) :].T
    _synchronize(device_name)
    return max(monotonic() - started, 1e-9)


def _train_once(
    graph: _GraphData,
    config: LightGCNConfig,
    seed: int,
    device_name: str,
) -> _TrainedEmbeddings:
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    model = _LightGCNNetwork(graph, config).to(device)
    triplets = _sample_triplets(graph, config, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    for _epoch in range(config.epochs):
        random.shuffle(triplets)
        for start in range(0, len(triplets), config.batch_size):
            batch = triplets[start : start + config.batch_size]
            subject_indices = torch.tensor(
                [item[0] for item in batch], dtype=torch.long, device=device
            )
            positive_indices = torch.tensor(
                [item[1] for item in batch], dtype=torch.long, device=device
            )
            negative_indices = torch.tensor(
                [item[2] for item in batch], dtype=torch.long, device=device
            )
            subject_embeddings, movie_embeddings = model.propagate()
            subject_values = subject_embeddings[subject_indices]
            positive_values = movie_embeddings[positive_indices]
            negative_values = movie_embeddings[negative_indices]
            positive_scores = (subject_values * positive_values).sum(dim=1)
            negative_scores = (subject_values * negative_values).sum(dim=1)
            bpr_loss = -functional.logsigmoid(positive_scores - negative_scores).mean()
            regularization = config.regularization * (
                model.subject_embedding.weight[subject_indices].square().mean()
                + model.movie_embedding.weight[positive_indices].square().mean()
                + model.movie_embedding.weight[negative_indices].square().mean()
            )
            loss = bpr_loss + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

    model.eval()
    with torch.no_grad():
        subject_embeddings, movie_embeddings = model.propagate()
    _synchronize(device_name)
    return _TrainedEmbeddings(
        subject_embeddings=_tensor_matrix(subject_embeddings.cpu()),
        movie_embeddings=_tensor_matrix(movie_embeddings.cpu()),
    )


def _sample_triplets(
    graph: _GraphData,
    config: LightGCNConfig,
    seed: int,
) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    subject_index = {subject_id: index for index, subject_id in enumerate(graph.subject_ids)}
    movie_index = {movie_id: index for index, movie_id in enumerate(graph.movie_ids)}
    observed_by_subject: defaultdict[int, set[int]] = defaultdict(set)
    for subject_id, movie_id in graph.positive_edges:
        observed_by_subject[subject_id].add(movie_id)

    triplets: list[tuple[int, int, int]] = []
    for subject_id, movie_id in graph.positive_edges:
        negatives = [
            candidate
            for candidate in graph.movie_ids
            if candidate not in observed_by_subject[subject_id]
        ]
        for _sample in range(config.negative_samples):
            if negatives:
                triplets.append(
                    (
                        subject_index[subject_id],
                        movie_index[movie_id],
                        movie_index[rng.choice(negatives)],
                    )
                )
    if not triplets:
        raise ValueError("LightGCN requires at least one available uniform negative sample")
    return triplets


def _tensor_matrix(value: Tensor) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(float(item) for item in row) for row in value.tolist())
    if not all(math.isfinite(item) for row in rows for item in row):
        raise ValueError("LightGCN produced a non-finite embedding")
    return rows


def _mps_is_available() -> bool:
    try:
        return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())
    except Exception:  # pragma: no cover - depends on installed torch build
        return False


def _synchronize(device_name: str) -> None:
    if device_name == "mps":
        torch.mps.synchronize()
