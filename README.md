# Movie Recommender Lifecycle Showcase

A local-first portfolio project that demonstrates a time-respecting evaluation stage for an
implicit-feedback movie recommender: Kafka ingestion, a 50% Data Snapshot, a withheld 50–60%
Future Window, Popularity, sparse ItemKNN, and Mult-VAE Candidate Pools, immutable Serving
Artifacts, FastAPI recommendations, and a static HTML report.

This orphan branch is intentionally independent from the repository's `main` history. Ticket
#34 implements the ingestion and Popularity tracer bullet; #35 adds the leakage-free
evaluation stage; #36 adds immutable activation; and #37 adds the Mult-VAE vertical slice.
Learned fusion, LightGCN, rolling snapshots, and latency benchmarking remain out of scope here.

## Documents

- [Domain language](CONTEXT.md)
- [Product and engineering specification](docs/spec.md)
- [Implementation roadmap](docs/roadmap.md)
- [Architecture decisions](docs/adr/)
- [Serving Artifact contract](docs/artifacts.md)
- [Tracking issue #36](https://github.com/ducanh2505/deep-recsys-lab/issues/36)
- [Tracking issue #37](https://github.com/ducanh2505/deep-recsys-lab/issues/37)

## Scope at a glance

- A small deterministic MovieLens-shaped fixture becomes immutable Rating Events.
- Ratings of at least 4.0 become Positive Interactions; the original rating is retained.
- The events cross an official Apache Kafka broker in Docker Compose and are stored as
  append-only Parquet batches.
- Snapshot materialization deduplicates deterministic Event IDs after replay.
- Popularity, sparse binary ItemKNN, and small seeded Mult-VAE are evaluated through one
  Candidate Retriever contract.
- Mult-VAE uses one binary catalog-index profile for Known-User and History-Only Queries; it
  prefers MPS, records the actual device, and retries on CPU with a recorded fallback reason.
- RRF is reported as a heuristic baseline and Oracle Union is reported only as a coverage ceiling.
- FastAPI exposes Known-User, History-Only, and Empty-History recommendation routes with
  history exclusion.
- A self-contained HTML report separates the 50% Data Snapshot, 50–60% Future Window, retrieval
  coverage, final ranking quality, Popularity, ItemKNN, Mult-VAE, RRF, Oracle Union, and
  Mult-VAE resource diagnostics.

Downloaded MovieLens data and generated artifacts are never committed.

## Run the fast evaluation lifecycle

Requirements: Python 3.12, `uv`, and Docker Desktop (the command starts the official Apache
Kafka Compose service automatically when `localhost:9092` is unavailable).

```bash
uv sync --dev
uv run movie-recsys fast --output artifacts/fast
```

The command creates an artifact store at `artifacts/fast/`, containing an immutable
`artifact-<id>/` directory, `active.json`, `report.html`, `data_snapshot.json`, and
`future_window.json`. It intentionally publishes the fixture twice to exercise at-least-once
replay; the full materialized snapshot remains unchanged by the duplicate batch. Fitting uses
only the first 50% of chronologically ordered events. The next 10% supplies Gold Candidates and
metrics, never model state or features.

The report compares Top-200 Popularity, ItemKNN, and Mult-VAE pools on the same deterministic
cohort with `Coverage@200`, `ConditionalRecall@10`, `EndToEndRecall@10`, `NDCG@10`, RRF, and
diagnostic Oracle Union coverage. The public API still returns Popularity until Learned Hybrid
Fusion is implemented; the loaded Mult-VAE payload is exercised through the serving smoke seam
for Known-User and History-Only inference.

To serve the loadable artifact:

```bash
uv run movie-recsys serve --artifact artifacts/fast
curl -s -X POST http://127.0.0.1:8000/recommendations \
  -H 'content-type: application/json' -d '{"subject_id": 1, "top_n": 5}'
```

`serve` accepts either an artifact store (which follows `active.json`) or one explicit immutable
artifact directory. The API loads persisted runtime payloads only; it does not fit a model,
replay Kafka, or read the Event Store. Restarting the API with the same store therefore keeps
the artifact identity and recommendation responses stable.

Run the deterministic tests without Docker:

```bash
uv run --group dev pytest
```

The real Kafka integration test is skipped unless the Compose broker is reachable. Start it
explicitly with `docker compose up -d kafka` when you want to run that test against the real
boundary. Mult-VAE training does not require Docker or an MPS device; an unavailable or
incompatible MPS path is recorded and retried on CPU.
