# Movie Recommender Lifecycle Showcase

A local-first portfolio project that demonstrates the lifecycle of an implicit-feedback
movie recommender: Kafka ingestion, temporal snapshots, four collaborative candidate
retrievers, learned hybrid fusion, leakage-safe evaluation, CPU serving, and a static HTML
report.

This orphan branch is intentionally independent from the repository's `main` history. It is
currently in the requirements and planning phase; implementation has not started.

## Documents

- [Domain language](CONTEXT.md)
- [Product and engineering specification](docs/spec.md)
- [Implementation roadmap](docs/roadmap.md)
- [Architecture decisions](docs/adr/)
- [Tracking issue #33](https://github.com/ducanh2505/deep-recsys-lab/issues/33)

## Scope at a glance

- MovieLens 20M ratings become `rating_recorded` events.
- Ratings of at least 4.0 become positive implicit interactions.
- A single local Kafka broker feeds an append-only Parquet Event Store.
- Data Snapshots advance from 50% to 100% in 10% increments.
- Popularity, ItemKNN, Mult-VAE, and LightGCN produce candidate pools.
- Learned Hybrid Fusion creates the final recommendation order for known-user and
  history-only queries.
- FastAPI serves Top-N recommendations on CPU.
- A self-contained HTML report presents retrieval coverage, ranking quality, cold-user
  behaviour, training cost, and serving latency.

Downloaded MovieLens data and generated artifacts are never committed.
