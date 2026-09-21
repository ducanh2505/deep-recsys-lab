# Movie Recommender Lifecycle Showcase

A local-first portfolio project that demonstrates the first Popularity-only tracer bullet for
an implicit-feedback movie recommender: Kafka ingestion, deduplicated snapshots, Positive
Interactions, a loadable Serving Artifact, FastAPI recommendations, and a static HTML report.

This orphan branch is intentionally independent from the repository's `main` history. Ticket
#34 implements the first Popularity-only tracer bullet; later retrievers and evaluation stages
remain intentionally out of scope here.

## Documents

- [Domain language](CONTEXT.md)
- [Product and engineering specification](docs/spec.md)
- [Implementation roadmap](docs/roadmap.md)
- [Architecture decisions](docs/adr/)
- [Tracking issue #34](https://github.com/ducanh2505/deep-recsys-lab/issues/34)

## Scope at a glance

- A small deterministic MovieLens-shaped fixture becomes immutable Rating Events.
- Ratings of at least 4.0 become Positive Interactions; the original rating is retained.
- The events cross an official Apache Kafka broker in Docker Compose and are stored as
  append-only Parquet batches.
- Snapshot materialization deduplicates deterministic Event IDs after replay.
- Popularity is the only Candidate Retriever in this tracer bullet.
- FastAPI exposes Known-User, History-Only, and Empty-History recommendation routes with
  history exclusion.
- A self-contained HTML report records the Data Snapshot, threshold, Popularity artifact, and
  three query examples.

Downloaded MovieLens data and generated artifacts are never committed.

## Run the Popularity tracer bullet

Requirements: Python 3.12, `uv`, and Docker Desktop (the command starts the official Apache
Kafka Compose service automatically when `localhost:9092` is unavailable).

```bash
uv sync --dev
uv run movie-recsys fast --output artifacts/fast
```

The command creates `artifacts/fast/serving_artifact/` and
`artifacts/fast/report.html`. It intentionally publishes the fixture twice to exercise
at-least-once replay; the materialized snapshot remains unchanged by the duplicate batch.

To serve the loadable artifact:

```bash
uv run movie-recsys serve --artifact artifacts/fast/serving_artifact
curl -s -X POST http://127.0.0.1:8000/recommendations \
  -H 'content-type: application/json' -d '{"subject_id": 1, "top_n": 5}'
```

Run the deterministic tests without Docker:

```bash
uv run --group dev pytest
```

The real Kafka integration test is skipped unless the Compose broker is reachable. Start it
explicitly with `docker compose up -d kafka` when you want to run that test against the real
boundary.
