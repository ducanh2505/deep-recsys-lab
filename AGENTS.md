# Repository Guidelines

## Project Structure & Module Organization

`lightgcn/` contains dataset loading, graph construction, the model, and training/evaluation. `scripts/prepare_ml20m.py` turns MovieLens ratings into the splits consumed by training; `main.py` runs that preparation script. `docs/` explains the data pipeline and model, and `paper/` holds the LightGCN reference PDF. Raw inputs and processed Parquet files live under `data/`; training outputs go under `artifacts/lightgcn_ml20m/`. There is currently no `tests/` directory.

## Build, Test, and Development Commands

- `uv sync`: install the Python 3.11 dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python scripts/prepare_ml20m.py`: rebuild the processed splits from `data/raw/ml-20m/ratings.csv`. This replaces the files in `data/processed/ml20m_lightgcn/`.
- `uv run python -m lightgcn.train --max-epochs 30`: train and evaluate LightGCN using the processed splits. See `docs/lightgcn.md` for the documented batch size and evaluation options.
- `uv run python -m compileall lightgcn scripts main.py`: check Python syntax without running the large dataset pipeline.

There is no separate build step. Data preparation and training can be expensive; use small, purpose-built fixtures when checking code changes.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for modules, functions, variables, and file names; `PascalCase` for classes; and `UPPER_CASE` for module constants. Add type hints and short docstrings for new public functions. Keep data paths anchored to the repository root, as the existing scripts do. No formatter or linter is configured; keep edits consistent with the surrounding module.

## Testing Guidelines

No test framework or coverage threshold is configured. For new behavior, add focused tests under `tests/` named `test_*.py`, especially for split rules, ID mapping, negative sampling, and ranking metrics. Document and install any test runner you introduce. Before a full training run, check syntax and validate changed data outputs against `data/processed/ml20m_lightgcn/manifest.json`.

## Commit & Pull Request Guidelines

This checkout has no Git history, so no existing commit-message convention can be verified. Use short, imperative subjects such as `Fix negative sampling for dense users`. In pull requests, describe the behavior change, commands run, and any effect on data splits or Recall@20/NDCG@20. Link a related issue when one exists. Keep large datasets and generated checkpoints out of routine commits; include metrics or manifest changes only when they are relevant and reproducible.
