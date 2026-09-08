# recsys-platform

`recsys-platform` is a Python toolkit for building recommendation systems as composable
software. It provides generic interaction-data contracts, deterministic preparation,
model and retriever plugins, hybrid fusion, immutable artifacts, offline evaluation,
and a framework-neutral serving engine.

The repository is intentionally dataset-agnostic. Local MovieLens and Yelp readers are
available as integrations, while the core accepts any CSV or Parquet interaction table
that can be mapped to `user_id` and `item_id`.

## Architecture

```text
interaction table
      │
      ▼
prepare ──► deterministic mappings + declared split
      │
      ▼
train ────► model, retriever, or composed hybrid plugin
      │
      ▼
artifact ─► schema + checksums + content-derived SHA-256 identity
      │
      ├────────► evaluate against held-out interactions
      └────────► serve through RecommendationEngine and optional BentoML
```

Generated data, runs, artifacts, and caches live under `var/` and are never source
artifacts. The project homepage is an independent vanilla Vite app under `apps/site`.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/) are supported. Choose one accelerator
extra and add only the capabilities you use:

```bash
uv sync --extra cpu --extra train --extra hybrid --extra serve --extra dev
```

Use `mps` or `cu126` instead of `cpu` for the matching PyTorch runtime.

## Quickstart

Run a bounded synthetic train–evaluate pipeline without downloading a dataset:

```bash
uv run recsys plugins list
uv run recsys --workspace-root . run \
  dataset=synthetic \
  model=multivae \
  training.epochs=1
```

The command writes run state under `var/runs/` and an immutable artifact under
`var/models/multivae/<sha256-digest>/`. It does not create a `latest` alias.

Prepare a generic table with an external configuration:

```bash
uv run recsys --config examples/configs/tabular.yaml data prepare
```

Configuration precedence is packaged defaults, optional external YAML, then repeatable
CLI dotlist overrides (`--set key=value`) or trailing Hydra-style overrides on commands
that compose configuration.

## Serve an artifact

Install the `serve` extra, then pass the artifact directory explicitly:

```bash
uv run recsys serve --artifact var/models/multivae/<sha256-digest>
```

The API accepts exactly one of a known `user_id` or a non-empty interaction history.
It exposes `POST /recommend`, `GET /model`, `/livez`, `/readyz`, and `/metrics`.
Set `RECSYS_API_KEY` to require `X-API-Key` on recommendation and model endpoints.

## Documentation

- [Architecture](docs/architecture.md)
- [Data contract](docs/data-contract.md)
- [Configuration and plugins](docs/configuration-and-plugins.md)
- [Serving](docs/serving.md)
- [Built-in models and retrievers](docs/models.md)

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest --cov=recsys --cov-report=term-missing

cd apps/site
npm ci
npm run build
npm run check
```

See [NOTICE](NOTICE) for required attribution covering adapted hybrid-retrieval material.
