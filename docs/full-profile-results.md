# Full MovieLens 20M profile: measured results

This evidence sheet records the completed `full` profile run. It is intentionally compact so the
repository can retain the measured result without committing the downloaded dataset or the 23 GB
artifact store.

## Run identity

| Field | Measured value |
| --- | --- |
| Completed | 2026-09-23 12:58:28 +07:00 |
| Source revision | `6c96ac1af1f21c068eb289b70042560a84591079` (clean) |
| Profile | `full`, seed 42, stages 50/60/70/80/90/100 |
| Host | Apple M4, 24 GiB unified memory, macOS 26.6.2, Python 3.12.13 |
| Source ratings | 20,000,263 |
| Positive Interactions (`rating >= 4.0`) | 9,995,410 |
| Source archive SHA-256 | `96f243c338a8665f6bcc89c53edf6ee39162a846940de6b7c8c48aeada765ff3` |
| Published archive MD5 verified by the loader | `cd245b17a1ae2cc31bb14903e1204af3` |
| Active artifact | `artifact-9e48bb9c91ec1f9ca336b00e`, cutoff 100% |
| Active artifact SHA-256 | `c69fa02574c2d38fa0ed8d8bada2da0dde699de7bc9cad89df6c55396db900a3` |
| Artifact bundle integrity SHA-256 | `42d8d5d9181366f9c9f076d32d979c330c302cce42ac138a5d952a8993db933d` |

The input is the [official MovieLens 20M archive](https://grouplens.org/datasets/movielens/20m/)
and remains subject to the
[GroupLens usage terms](https://files.grouplens.org/datasets/movielens/ml-20m-README.html).
The loader checks the published MD5 before preparing the chronological Parquet source; lifecycle
provenance uses the archive SHA-256 above.

The observed end-to-end wall time was about 94 minutes. This is a local reproducibility result,
not a production throughput claim.

## Evaluation contract

The headline quality result uses the model trained at the 90% Data Snapshot and the withheld
90→100% Future Window. Each eligible Subject contributes at most one Query and its first future
Positive Interaction as the Gold Candidate. Model fitting, LHF labels, feature selection, and
serving state cannot read that Future Window.

- Known-User: Subject has snapshot identity and Positive-Interaction history.
- History-Only: the same supplied history is evaluated without a persisted Subject identity;
  LightGCN is unavailable by contract.
- Empty-History: Subject has no Positive Interaction in the Data Snapshot and first becomes
  positive in the Future Window. Popularity is the defined fallback.
- Cohorts are selected deterministically and capped at 5,000 Subjects per mode.
- `ConditionalRecall@10` uses covered Queries as its denominator. The other quality metrics in
  the table use the full cohort denominator.

## Final 90→100 quality

| Query mode and system | n | Coverage@200 | ConditionalRecall@10 | EndToEndRecall@10 | NDCG@10 | CatalogCoverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Known-User, validation-selected best single (Mult-VAE) | 4,908 | 0.2400 | 0.1358 | 0.0326 | 0.0165 | 0.0642 |
| Known-User, LHF | 4,908 | 0.2412 | 0.1537 | 0.0371 | 0.0186 | 0.0672 |
| Known-User, diagnostic Oracle Union | 4,908 | 0.3062 | 0.1084 | 0.0332 | 0.0195 | 0.1815 |
| History-Only, validation-selected best single (Mult-VAE) | 4,908 | 0.2400 | 0.1358 | 0.0326 | 0.0165 | 0.0642 |
| History-Only, LHF | 4,908 | 0.2414 | 0.1570 | 0.0379 | 0.0194 | 0.0648 |
| History-Only, diagnostic Oracle Union | 4,908 | 0.2989 | 0.1295 | 0.0387 | 0.0222 | 0.0445 |
| Empty-History, Popularity fallback | 5,000 | 0.1718 | 0.5518 | 0.0948 | 0.0655 | 0.0006 |

Against the retriever selected on prior validation, LHF improved EndToEndRecall@10 by 13.75% for
Known-User and 16.25% for History-Only. NDCG@10 improved by 13.27% and 17.66%, respectively.
These are relative improvements for this one deterministic run. “Best single” is fixed from prior
validation rather than chosen after seeing the 90→100 Future Window.

Retrieval remains the main quality constraint: the Gold Candidate was outside the diagnostic
Oracle Union for 3,405/4,908 Known-User Queries and 3,441/4,908 History-Only Queries. The Oracle
Union is a diagnostic ceiling and is never served.

## Full-positive neural training evidence

The active 100% manifest records the following observed training facts:

| Retriever | Positive interactions seen | Graph/profile edges | Actual device | Duration |
| --- | ---: | ---: | --- | ---: |
| Mult-VAE | 9,995,410 | 9,995,410 binary profile edges | MPS | 36.41 s |
| LightGCN | 9,995,410 | 9,995,410 deduplicated graph edges/triplets | CPU | 19.13 s |

LightGCN requested MPS and selected CPU after its representative benchmark measured 4.493 s on
MPS versus 2.889 s on CPU, exceeding the committed 1.25× slowdown threshold. The fallback and
both benchmark measurements are stored in the immutable manifest.

## Warm serving latency

The benchmark loaded the active artifact before timing and invoked the FastAPI path in-process at
concurrency 1 with 5 warm-ups and 30 successful samples per mode. It excludes TCP/network time,
startup, training, and artifact loading.

| Query mode | p50 | p95 | p99 | Throughput | p95 target | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Known-User | 54.99 ms | 55.38 ms | 55.78 ms | 18.23 req/s | 200 ms | pass |
| History-Only | 47.88 ms | 51.98 ms | 52.41 ms | 20.69 req/s | 200 ms | pass |
| Empty-History | 23.37 ms | 23.80 ms | 23.86 ms | 42.81 req/s | 20 ms | **fail** |

The Empty-History SLO miss is retained in the report and is the first performance follow-up.

## Reproduce

```bash
uv sync --dev
docker compose up -d kafka
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --no-sync movie-recsys full \
  --cache var/datasets --output artifacts/full
open artifacts/full/report.html
```

The completed local evidence lives in `artifacts/full/report.html`,
`artifacts/full/rolling_checkpoint.json`, `artifacts/full/evaluations/stage-090.json`,
`artifacts/full/latency_benchmark.json`, and the active artifact manifest. Dataset and generated
artifact directories are ignored by Git.

## Resume-ready claims

Use measured scope and denominators in interview discussion:

- Built a leakage-free, Kafka-backed recommender lifecycle for MovieLens 20M, replaying
  20,000,263 chronological ratings into six reproducible snapshots and training Mult-VAE and
  LightGCN on all 9,995,410 positive interactions.
- Designed a temporal 90→100 evaluation for known users, history-only requests, and new users;
  Learned Hybrid Fusion improved EndToEndRecall@10 by 13.8% and 16.3% versus the
  validation-selected Mult-VAE baseline on two 4,908-Query cohorts.
- Shipped checksum-verified immutable artifacts and a FastAPI serving path; the active full-data
  artifact measured 55.4 ms and 52.0 ms p95 for Known-User and History-Only requests on an Apple
  M4, while preserving the measured 23.8 ms Empty-History SLO miss in the report.

## Next steps, in order

1. Reduce Empty-History p95 below 20 ms by profiling request construction and caching the
   immutable Popularity Top-N after history exclusion; retain the same benchmark contract as a
   regression gate.
2. Improve retrieval coverage. Add a metadata/content retriever for interaction-new Movies or a
   two-tower/ANN retriever, then report incremental Oracle coverage before changing fusion.
3. Run multiple seeds and add subject-level bootstrap confidence intervals for the final temporal
   window. Publish the compact aggregate evidence as a CI artifact.
4. Diagnose LightGCN's low recall with controlled ablations for negative sampling, layer count,
   embedding size, and candidate budget; keep it out of headline claims until it earns inclusion.
5. Add loopback HTTP and concurrent load tests with RSS/CPU measurements. Keep the current
   in-process ASGI benchmark as the deterministic request-path regression test.

## Limits

- This is one deterministic local run with no confidence interval or online experiment.
- The latency evidence is warm, in-process ASGI at concurrency 1 and is not a capacity test.
- The collaborative retrievers do not make Movies with no snapshot interactions recommendable.
- Stage 100 is for artifact production and activation; it has no 100→110 Future Window and no
  future-quality claim.
