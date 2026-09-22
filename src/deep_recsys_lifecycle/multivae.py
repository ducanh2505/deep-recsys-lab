from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from math import isfinite
from time import monotonic
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from .event_store import DataSnapshot
from .models import PositiveInteraction
from .positive import history_movie_ids
from .serving import MultVAERetriever


@dataclass(frozen=True, slots=True)
class MultVAEConfig:
    """Small fixed fast-profile configuration for the Mult-VAE vertical slice."""

    hidden_dim: int = 32
    latent_dim: int = 16
    dropout: float = 0.0
    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 0.01
    kl_beta: float = 0.2

    def __post_init__(self) -> None:
        if self.hidden_dim < 1 or self.latent_dim < 1:
            raise ValueError("Mult-VAE dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("Mult-VAE dropout must be in [0, 1)")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Mult-VAE epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.kl_beta < 0:
            raise ValueError("Mult-VAE learning_rate must be positive and kl_beta non-negative")

    def to_dict(self) -> dict[str, int | float]:
        return {key: value for key, value in asdict(self).items()}


class _MultiVAE(nn.Module):
    def __init__(self, item_count: int, config: MultVAEConfig) -> None:
        super().__init__()
        self.dropout_probability = config.dropout
        self.encoder = nn.Linear(item_count, config.hidden_dim)
        self.mean = nn.Linear(config.hidden_dim, config.latent_dim)
        self.log_variance = nn.Linear(config.hidden_dim, config.latent_dim)
        self.decoder = nn.Linear(config.latent_dim, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, item_count)

    def forward(self, interactions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        normalized = functional.normalize(interactions.float(), p=2, dim=1)
        hidden = torch.tanh(
            self.encoder(
                functional.dropout(normalized, p=self.dropout_probability, training=self.training)
            )
        )
        mean = self.mean(hidden)
        log_variance = self.log_variance(hidden)
        if self.training:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        decoded = torch.tanh(self.decoder(latent))
        return self.output(decoded), mean, log_variance


def fit_multivae(
    snapshot: DataSnapshot,
    interactions: Iterable[PositiveInteraction],
    *,
    config: MultVAEConfig | None = None,
    seed: int = 42,
    device_preference: str = "auto",
    mps_probe: Callable[[], bool] | None = None,
) -> MultVAERetriever:
    """Fit Mult-VAE from snapshot-bounded binary Positive Interactions.

    ``auto`` prefers MPS when it reports availability.  Any MPS initialization or training
    failure reruns the same fixed profile on CPU and records the reason in the exported training
    metadata.  ``mps_probe`` is an injection seam for deterministic fallback tests.
    """

    resolved_config = config or MultVAEConfig()
    if device_preference not in {"auto", "cpu", "mps"}:
        raise ValueError("device_preference must be one of: auto, cpu, mps")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Mult-VAE seed must be an integer")

    catalog = tuple(sorted({event.movie_id for event in snapshot.events}))
    if not catalog:
        raise ValueError("Mult-VAE requires a non-empty snapshot catalog")
    catalog_ids = set(catalog)
    snapshot_event_ids = {event.event_id for event in snapshot.events}
    histories: defaultdict[int, list[PositiveInteraction]] = defaultdict(list)
    for interaction in interactions:
        if (
            interaction.event_id not in snapshot_event_ids
            or interaction.movie_id not in catalog_ids
        ):
            continue
        histories[interaction.subject_id].append(interaction)
    subject_histories = {
        subject_id: history_movie_ids(subject_interactions)
        for subject_id, subject_interactions in sorted(histories.items())
    }
    profiles = tuple(
        tuple(float(movie_id in set(history)) for movie_id in catalog)
        for _subject_id, history in subject_histories.items()
    )

    requested_device = "mps" if device_preference in {"auto", "mps"} else "cpu"
    fallback_reason: str | None = None
    actual_device = "cpu"
    training_started = monotonic()
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
                trained_model = _train_once(profiles, len(catalog), resolved_config, seed, "mps")
                actual_device = "mps"
            except Exception as error:  # MPS compatibility is intentionally a per-approach seam.
                fallback_reason = f"MPS training failed: {type(error).__name__}: {error}"
                trained_model = _train_once(profiles, len(catalog), resolved_config, seed, "cpu")
        else:
            trained_model = _train_once(profiles, len(catalog), resolved_config, seed, "cpu")
    else:
        trained_model = _train_once(profiles, len(catalog), resolved_config, seed, "cpu")
    training_seconds = monotonic() - training_started

    metadata: dict[str, Any] = {
        "requested_device": requested_device,
        "actual_device": actual_device,
        "duration_seconds": float(training_seconds),
        "seed": seed,
        "fallback_reason": fallback_reason,
        "hyperparameters": resolved_config.to_dict(),
    }
    configuration: dict[str, Any] = {
        **resolved_config.to_dict(),
        "profile_semantics": "binary_positive_interaction_by_catalog_index",
        "unknown_movie_policy": "ignore_in_profile_and_exclude_from_candidates",
        "empty_profile_behavior": "catalog_id_order_with_zero_scores",
    }
    return MultVAERetriever(
        catalog=catalog,
        subject_histories=subject_histories,
        encoder_weight=_state_matrix(trained_model, "encoder.weight"),
        encoder_bias=_state_vector(trained_model, "encoder.bias"),
        mean_weight=_state_matrix(trained_model, "mean.weight"),
        mean_bias=_state_vector(trained_model, "mean.bias"),
        decoder_weight=_state_matrix(trained_model, "decoder.weight"),
        decoder_bias=_state_vector(trained_model, "decoder.bias"),
        output_weight=_state_matrix(trained_model, "output.weight"),
        output_bias=_state_vector(trained_model, "output.bias"),
        configuration=configuration,
        training_metadata=metadata,
    )


def _mps_is_available() -> bool:
    try:
        return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())
    except Exception:  # pragma: no cover - depends on the installed torch build
        return False


def _train_once(
    profiles: tuple[tuple[float, ...], ...],
    item_count: int,
    config: MultVAEConfig,
    seed: int,
    device_name: str,
) -> _MultiVAE:
    random.seed(seed)
    torch.manual_seed(seed)
    model = _MultiVAE(item_count, config).to(torch.device(device_name))
    if profiles:
        target = torch.tensor(profiles, dtype=torch.float32, device=torch.device(device_name))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        model.train()
        for epoch in range(config.epochs):
            for start in range(0, len(profiles), config.batch_size):
                batch = target[start : start + config.batch_size]
                logits, mean, log_variance = model(batch)
                negative_log_likelihood = (
                    -(functional.log_softmax(logits, dim=1) * batch).sum(dim=1).mean()
                )
                divergence = (
                    -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(dim=1).mean()
                )
                beta = config.kl_beta * (epoch + 1) / config.epochs
                loss = negative_log_likelihood + beta * divergence
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    model.eval().cpu()
    return model


def _state_vector(model: _MultiVAE, name: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in model.state_dict()[name].detach().cpu().flatten())
    if not all(isfinite(value) for value in values):
        raise ValueError(f"Mult-VAE state {name!r} contains a non-finite value")
    return values


def _state_matrix(model: _MultiVAE, name: str) -> tuple[tuple[float, ...], ...]:
    tensor = model.state_dict()[name].detach().cpu()
    values = tuple(tuple(float(value) for value in row) for row in tensor.tolist())
    if not all(isfinite(value) for row in values for value in row):
        raise ValueError(f"Mult-VAE state {name!r} contains a non-finite value")
    return values
