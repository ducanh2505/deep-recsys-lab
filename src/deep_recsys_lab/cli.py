"""Typer command line interface for the complete workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, cast

import torch
import typer
from torch.utils.data import DataLoader

from .config import compose_hydra_config, to_app_config
from .data.dataset import InteractionDataset, load_prepared_data
from .data.download import download_movielens20m, extract_movielens_archive
from .data.preprocess import prepare_movielens20m
from .device import select_device
from .evaluation.evaluator import evaluate_model
from .lightgcn.config import compose_lightgcn_config
from .lightgcn.data import load_lightgcn_data, prepare_lightgcn_data
from .lightgcn.experiment import reproduce_lightgcn
from .lightgcn.report import export_verified_report
from .model import build_model
from .model.base import BaseRecommender
from .serving.model_store import register_bento_model
from .training.checkpoint import load_checkpoint
from .training.trainer import Trainer

app = typer.Typer(
    help="Reproducible Multi-VAE collaborative filtering workflow.", no_args_is_help=True
)
data_app = typer.Typer(help="Download and prepare MovieLens data.", no_args_is_help=True)
model_app = typer.Typer(help="Manage immutable BentoML models.", no_args_is_help=True)
lightgcn_app = typer.Typer(
    help="Prepare and reproduce the isolated LightGCN research experiment.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")
app.add_typer(lightgcn_app, name="lightgcn")


def _model_from_config(config: Any, n_items: int) -> BaseRecommender:
    model_cfg = config.model
    return build_model(
        model_cfg.name,
        n_items=n_items,
        hidden_dims=tuple(model_cfg.hidden_dims),
        latent_dim=model_cfg.latent_dim,
        dropout=model_cfg.dropout,
        input_normalization=model_cfg.input_normalization,
    )


def _checkpoint_model(checkpoint: Path, n_items: int, device: torch.device) -> BaseRecommender:
    payload = cast(dict[str, Any], torch.load(checkpoint, map_location=device, weights_only=False))
    model_config = payload.get("model_config") or payload.get("config", {}).get("model", {})
    model_config = dict(model_config)
    model_name = model_config.pop("name", "multvae")
    model_config["n_items"] = n_items
    model = build_model(model_name, **model_config).to(device)
    load_checkpoint(checkpoint, model, device=device, restore_rng=False)
    return model


@data_app.command("download")
def data_download(
    root: Annotated[
        Path, typer.Option(help="Directory for the archive and checksum sidecar.")
    ] = Path("data/raw"),
    force: Annotated[bool, typer.Option(help="Re-download and re-verify existing files.")] = False,
) -> None:
    result = download_movielens20m(root, force=force)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@data_app.command("prepare")
def data_prepare(
    input_dir: Annotated[
        Path, typer.Option(help="Extracted ml-20m directory or its parent.")
    ] = Path("data/raw"),
    output_dir: Annotated[Path, typer.Option(help="Versioned processed-data directory.")] = Path(
        "data/processed"
    ),
    min_positive: Annotated[
        int | None, typer.Option(help="Override the minimum positive ratings per user.")
    ] = None,
    validation_users: Annotated[
        int | None, typer.Option(help="Override validation-user count.")
    ] = None,
    test_users: Annotated[int | None, typer.Option(help="Override test-user count.")] = None,
    strict_user_counts: Annotated[
        bool, typer.Option(help="Require the configured held-out user counts.")
    ] = True,
) -> None:
    config = to_app_config(compose_hydra_config(overrides=["dataset=movielens20m"])).dataset
    values = {**config.__dict__, "strict_user_counts": strict_user_counts}
    if min_positive is not None:
        values["min_positive_ratings"] = min_positive
    if validation_users is not None:
        values["n_validation_users"] = validation_users
    if test_users is not None:
        values["n_test_users"] = test_users
    if not (input_dir / "ratings.csv").exists() and (input_dir / "ml-20m.zip").exists():
        input_dir = extract_movielens_archive(input_dir / "ml-20m.zip", input_dir)
    prepared = prepare_movielens20m(input_dir, output_dir, config=type(config)(**values))
    typer.echo(
        f"prepared {prepared.n_train_users:,} train users x {prepared.n_items:,} items "
        f"at {prepared.root}"
    )


@app.command("train")
def train(
    config_name: Annotated[
        str, typer.Option("--config-name", help="Hydra config name.")
    ] = "config",
    overrides: Annotated[
        list[str] | None, typer.Argument(help="Hydra overrides, e.g. trainer.epochs=1.")
    ] = None,
) -> None:
    hydra_config = compose_hydra_config(config_name=config_name, overrides=overrides or [])
    config = to_app_config(hydra_config)
    prepared = load_prepared_data(config.data_dir)
    device = select_device(config.device)
    model = _model_from_config(config, prepared.n_items)
    train_loader = DataLoader(
        InteractionDataset(prepared.train),
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.num_workers,
    )
    validation_loader = DataLoader(
        InteractionDataset(
            prepared.validation_fold_in,
            eval=True,
            fold_in=prepared.validation_fold_in,
            fold_out=prepared.validation_fold_out,
        ),
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
    )
    typer.echo("Starting training with the following configuration:")
    for key, value in config.__dict__.items():
        typer.echo(f"  {key}: {value}")
    trainer = Trainer(
        model,
        train_loader,
        validation_loader,
        output_dir=config.output_dir,
        device=device,
        config=config,
        epochs=config.trainer.epochs,
        learning_rate=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
        beta_cap=config.trainer.beta_cap,
        total_anneal_steps=config.trainer.total_anneal_steps,
        validation_ks=config.evaluation.ks,
        max_train_batches=config.trainer.max_train_batches,
        max_eval_batches=config.trainer.max_eval_batches,
        gradient_clip_norm=config.trainer.gradient_clip_norm,
        resume_from=config.trainer.resume_from,
        seed=config.seed,
    )
    history = trainer.run()
    typer.echo(f"training complete: {len(history)} epoch records at {config.output_dir}")


@app.command("evaluate")
def evaluate(
    checkpoint: Annotated[Path, typer.Option(help="Selected best or resumable checkpoint.")],
    data_dir: Annotated[Path, typer.Option(help="Prepared-data directory.")] = Path(
        "data/processed"
    ),
    device: Annotated[str, typer.Option(help="cpu, cuda, mps, or auto.")] = "cpu",
    max_batches: Annotated[
        int | None, typer.Option(help="Bound evaluation for a smoke run.")
    ] = None,
) -> None:
    prepared = load_prepared_data(data_dir)
    selected_device = select_device(device)
    model = _checkpoint_model(checkpoint, prepared.n_items, selected_device)
    test_loader = DataLoader(
        InteractionDataset(
            prepared.test_fold_in,
            eval=True,
            fold_in=prepared.test_fold_in,
            fold_out=prepared.test_fold_out,
        ),
        batch_size=500,
        shuffle=False,
    )
    metrics = evaluate_model(
        model,
        test_loader,
        device=selected_device,
        ks=(20, 50, 100),
        max_batches=max_batches,
    )
    output = checkpoint.parent / "test_metrics.json"
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


@model_app.command("register")
def model_register(
    checkpoint: Annotated[Path, typer.Option(help="Best checkpoint to register.")],
    model_version: Annotated[
        str,
        typer.Option(help="Required immutable Bento version; 'latest' is rejected."),
    ],
    data_dir: Annotated[Path, typer.Option(help="Prepared-data directory.")] = Path(
        "data/processed"
    ),
) -> None:
    registered = register_bento_model(
        checkpoint,
        data_dir,
        model_version=model_version,
    )
    typer.echo(str(registered.tag))


@lightgcn_app.command("prepare")
def lightgcn_prepare(
    config_name: Annotated[
        str, typer.Option("--config-name", help="Dedicated Hydra config name.")
    ] = "lightgcn",
    overrides: Annotated[
        list[str] | None, typer.Argument(help="Hydra overrides for the LightGCN config.")
    ] = None,
    force: Annotated[
        bool, typer.Option(help="Rebuild even when a verified artifact already exists.")
    ] = False,
) -> None:
    """Prepare the deterministic implicit MovieLens split and checksums."""

    config = compose_lightgcn_config(config_name, overrides)
    output_dir = Path(config.data_dir).resolve()
    if (output_dir / "manifest.json").exists() and not force:
        prepared = load_lightgcn_data(output_dir)
    else:
        raw_dir = Path(config.data.raw_dir).resolve()
        if (
            not (raw_dir / "ratings.csv").exists()
            and not (raw_dir / "ml-20m" / "ratings.csv").exists()
            and (raw_dir / "ml-20m.zip").exists()
        ):
            extract_movielens_archive(raw_dir / "ml-20m.zip", raw_dir)
        prepared = prepare_lightgcn_data(
            raw_dir,
            output_dir,
            config=config.data,
            seed=config.seed,
        )
    typer.echo(json.dumps(prepared.manifest, indent=2, sort_keys=True))


@lightgcn_app.command("reproduce")
def lightgcn_reproduce(
    config_name: Annotated[
        str, typer.Option("--config-name", help="Dedicated Hydra config name.")
    ] = "lightgcn",
    overrides: Annotated[
        list[str] | None, typer.Argument(help="Hydra overrides; existing runs reject drift.")
    ] = None,
) -> None:
    """Run or resume the bounded sweep, retrain, and one-time test evaluation."""

    config = compose_lightgcn_config(config_name, overrides)
    prepared = load_lightgcn_data(Path(config.data_dir).resolve())
    result = reproduce_lightgcn(prepared, config, Path(config.output_dir).resolve())
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") != "verified":
        raise typer.Exit(code=2)


@lightgcn_app.command("report")
def lightgcn_report(
    config_name: Annotated[
        str, typer.Option("--config-name", help="Dedicated Hydra config name.")
    ] = "lightgcn",
    overrides: Annotated[
        list[str] | None, typer.Argument(help="Hydra overrides matching the verified run.")
    ] = None,
) -> None:
    """Verify the run and export a deployment-safe aggregate JSON report."""

    config = compose_lightgcn_config(config_name, overrides)
    prepared = load_lightgcn_data(Path(config.data_dir).resolve())
    report = export_verified_report(
        Path(config.output_dir).resolve(),
        prepared,
        Path(config.report.output).resolve(),
    )
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    app()
