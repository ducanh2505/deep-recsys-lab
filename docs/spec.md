# Movie Recommender Lifecycle Showcase Specification

## Problem Statement

An AI/ML engineering portfolio needs to demonstrate more than an isolated recommendation
model. It should show how user events enter a system, become time-bounded training evidence,
produce comparable candidate retrievers, support cold-user degradation, become immutable
serving artifacts, and generate defensible quality and latency evidence. The system must be
small enough to build and run locally on an Apple-silicon Mac while remaining realistic enough
to discuss in an engineering interview.

## Solution

Build a local-first MovieLens 20M recommendation lifecycle. Rating Events are replayed through
a single Kafka broker into an append-only Event Store. Data Snapshots advance from 50% to 100%
of the chronological event stream. Popularity, ItemKNN, Mult-VAE, and LightGCN Candidate
Retrievers feed query-mode-specific Learned Hybrid Fusion classifiers. A CPU FastAPI service
returns unseen Top-N Candidates, and a self-contained HTML report explains retrieval coverage,
end-to-end ranking quality, user cold-start behaviour, training cost, and request latency.

The primary test seam is one high-level lifecycle command that starts from a verified source
archive and produces a loadable Serving Artifact plus a report. Kafka ingestion and the HTTP
contract are secondary seams where failures require narrower diagnosis.

## User Stories

1. As a portfolio reviewer, I want to understand the system architecture quickly, so that I
   can evaluate the candidate's ML engineering judgment.
2. As a portfolio reviewer, I want one static report, so that I can inspect results without
   operating the system interactively.
3. As a developer, I want the MovieLens archive downloaded and checksum-verified, so that the
   input is reproducible without redistributing the dataset.
4. As a developer, I want ratings replayed in chronological order, so that the simulated
   lifecycle respects the source timeline.
5. As a data engineer, I want Rating Events to cross Kafka, so that ingestion is an observable
   system boundary rather than a direct file read disguised as streaming.
6. As a data engineer, I want consumed events stored append-only, so that every Data Snapshot
   can be reproduced from retained evidence.
7. As a data engineer, I want deterministic event identifiers, so that at-least-once delivery
   does not duplicate training evidence.
8. As an ML engineer, I want explicit ratings converted by one declared threshold, so that all
   approaches solve the same implicit-feedback problem.
9. As an ML engineer, I want a 50% initial Data Snapshot, so that the first model has substantial
   evidence while leaving several future lifecycle stages to observe.
10. As an ML engineer, I want each Future Window evaluated before ingestion, so that reported
    quality contains no temporal leakage.
11. As an ML engineer, I want Popularity as a dependable baseline and fallback, so that every
    Query mode has defined behaviour.
12. As an ML engineer, I want ItemKNN as a memory-based collaborative baseline, so that neural
    approaches are compared with a strong simple retriever.
13. As an ML engineer, I want Mult-VAE to consume a supplied preference profile, so that both
    known and identity-free histories can receive learned recommendations.
14. As an ML engineer, I want LightGCN for known Subjects, so that graph collaborative retrieval
    is represented without pretending it supports unseen identities.
15. As an Apple-silicon developer, I want neural training to prefer MPS and fall back per
    approach, so that an unsupported operator does not block the lifecycle.
16. As an evaluator, I want every Candidate Retriever assessed on the same deterministic cohort,
    so that model comparisons are meaningful.
17. As an evaluator, I want Gold Candidates excluded from retrieval pools unless naturally
    found, so that coverage exposes the real retrieval ceiling.
18. As an evaluator, I want to compare the best single retriever, RRF, LHF, and the Oracle Union,
    so that complementarity and realized fusion headroom are visible.
19. As an evaluator, I want retrieval and conditional ranking reported separately, so that I can
    locate whether a failure occurred before or after candidate generation.
20. As an evaluator, I want results segmented by history length and Query mode, so that a global
    average does not hide cold-user behaviour.
21. As an evaluator, I want Interaction-New Candidates identified as a limitation, so that the
    report does not claim unsupported item cold-start capability.
22. As a serving client, I want to request recommendations using a known Subject, so that stored
    preference history can drive all eligible retrievers.
23. As a serving client, I want to supply Movie history without a Subject identity, so that the
    system can personalize for an unknown user.
24. As a serving client, I want a defined Empty-History response, so that the API degrades to
    popularity instead of failing or fabricating personalization.
25. As a serving client, I want previously observed Movies excluded, so that every returned
    Candidate is new to the supplied context.
26. As a serving client, I want a bounded Top-N contract, so that latency and output semantics
    remain predictable.
27. As an ML engineer, I want separate fusion artifacts for Known-User and History-Only Queries,
    so that each classifier learns from the retriever bank actually available at serving time.
28. As an ML engineer, I want LHF trained only from prior validation outcomes, so that its gains
    are not caused by future labels.
29. As an operator, I want every Lifecycle Stage to emit an immutable manifest, so that results
    can be traced to data, configuration, code, seed, and device.
30. As an operator, I want the 100% Serving Artifact activated only after export and smoke tests,
    so that the API never loads a partial lifecycle result.
31. As an operator, I want latency measured separately for each Query mode, so that a cheap
    fallback cannot conceal expensive personalized inference.
32. As a portfolio reviewer, I want failed latency targets displayed rather than suppressed, so
    that the report remains credible.
33. As a developer, I want a reduced fast profile, so that I can verify the complete path without
    waiting for a full MovieLens run.
34. As a developer, I want a full profile with the same contracts, so that the final report uses
    representative data without maintaining another system.
35. As a developer, I want one high-level command to run the lifecycle, so that setup and demo
    instructions remain simple.

## Implementation Decisions

- The repository starts from an orphan history and shares no ancestor with `main`.
- MovieLens 20M is downloaded from its official source, verified, acknowledged, and never
  committed with generated data or model artifacts.
- A Rating Event retains the source rating. Positive Interaction preparation applies a rating
  threshold of 4.0; lower ratings do not become negatives.
- A single Kafka broker, topic, and partition provide the local ingestion boundary and stable
  event ordering. The consumer accepts at-least-once delivery.
- Event batches are persisted append-only in Parquet. Snapshot materialization deduplicates by
  deterministic event identifier and uses DuckDB for local analytical reads.
- The initial Data Snapshot contains the earliest 50% of events. Later snapshots accumulate
  another 10% until 100%.
- Each snapshot from 50% through 90% is evaluated against the next Future Window before that
  window is ingested. The 100% artifact has no future-quality claim.
- One Gold Candidate per eligible Subject is the first Positive Interaction in the Future
  Window. Fast and full cohorts contain at most 1,000 and 5,000 deterministic Subjects.
- Every Candidate Retriever searches the available unseen Candidate Catalog and emits Top-200.
- Known-User Queries use Popularity, ItemKNN, Mult-VAE, and LightGCN. History-Only Queries omit
  LightGCN. Empty-History Queries use Popularity without fusion.
- LHF uses a gradient-boosted binary classifier over the union of constituent Candidate Pools.
  Its features comprise per-retriever ranks, scores and presence; agreement aggregates; history
  length and user-cold status; and snapshot-bounded item popularity and interaction-new status.
- Separate Known-User and History-Only fusion artifacts are trained because their retriever
  banks differ. LHF supplies the final MVP ordering; there is no downstream learned ranker.
- RRF is an executable heuristic baseline. The Oracle Union is diagnostic only and cannot be
  served as a ranker.
- Headline evaluation reports Retrieval Coverage@200, Oracle headroom realized,
  ConditionalRecall@10, end-to-end Recall@10, NDCG@10, and CatalogCoverage@10.
- The report segments Empty-History, 1-4, 5-19, and 20-plus interaction regimes and separates
  Known-User from History-Only Queries.
- The service exposes one recommendation operation with mutually exclusive known-subject and
  supplied-history inputs. Top-N defaults to 10 and is capped at 100.
- Popularity, ItemKNN, fusion, and serving execute on CPU. Mult-VAE prefers MPS. LightGCN tries
  MPS and falls back to CPU when compatibility or performance makes that necessary.
- Each Serving Artifact is immutable and identifies its snapshot cutoff, event count, dataset
  checksum, configuration, seed, device, timings, metrics, model payloads, and source commit.
- Snapshot 100% is activated after successful fit, export, and smoke tests. The 90% snapshot's
  evaluation against 90-100% supplies the final headline quality.
- Warm CPU latency targets are p95 at most 200 ms for Known-User and History-Only Queries and
  20 ms for Empty-History Queries at concurrency one. Failures are reported rather than hidden.
- The static report is self-contained and includes architecture, data flow, lifecycle quality,
  retrieval diagnosis, cold-user segments, resource use, latency, limitations, and provenance.
- The runtime stack is Python 3.12 with `uv`, Docker Compose, the official Apache Kafka image,
  `confluent-kafka`, Parquet, DuckDB, NumPy/SciPy, PyTorch, LightGBM, FastAPI/Uvicorn, Typer,
  Jinja2, and Plotly.
- Airflow, Dagster, Kubeflow, Kubernetes, MLflow, public hosting, a web UI, item-content
  retrieval, LLM reranking, and a second-stage learned ranker are not MVP dependencies.

## Testing Decisions

- The primary acceptance test exercises the highest seam: a reduced deterministic source fixture
  enters the lifecycle and produces a loadable Serving Artifact plus a valid static report.
- A Kafka integration test uses the real local broker contract to prove replay, consumption,
  offset progress, append-only persistence, and deduplication after repeated delivery.
- Snapshot tests assert temporal cutoffs, cumulative event membership, and the absence of Future
  Window evidence from training and fusion features.
- Candidate Retriever contract tests assert deterministic ordering, history exclusion, catalog
  membership, bounded output, and defined behaviour for unsupported Query modes.
- Query routing tests observe external recommendations and selected route, not internal class
  composition.
- LHF tests prove union construction, serving-time-only features, query-mode-specific retriever
  availability, and inability to recover a Gold Candidate absent from the union.
- Evaluation tests use hand-checkable fixtures to verify coverage, conditional recall,
  end-to-end decomposition, NDCG, catalog coverage, history segments, and Oracle headroom.
- Artifact tests prove immutability, manifest completeness, compatible loading, failed-export
  rejection, and activation only after smoke tests.
- API contract tests cover mutually exclusive inputs, unknown Subjects, empty history, Top-N
  bounds, observed-Candidate exclusion, response provenance, and stable validation errors.
- Report tests validate required sections and embedded data structurally; they do not snapshot
  exact HTML or chart pixels.
- Latency measurement is a benchmark, not a unit test. Its output records warm-up, sample count,
  concurrency, hardware, p50, p95, p99, throughput, and SLO status.
- No prior implementation exists on this orphan branch, so there is no local testing pattern to
  preserve. Tests should remain at the contracts above rather than mirror future module layout.

## Out of Scope

- New-item cold-start mitigation or a claim that collaborative retrieval can recommend unseen
  items with no interaction evidence.
- Content, semantic, dense, LLM, or cross-encoder retrieval and reranking.
- The paper's second downstream LightGBM ranker and its LLM comparisons.
- Online learning, continuous streaming retraining, automatic rollback, canary deployment, or a
  production model registry.
- Exactly-once Kafka processing, multi-broker operation, Schema Registry, Kafka Connect, or
  distributed storage.
- Kubernetes, cloud deployment, public endpoints, authentication, authorization, or multi-tenant
  operation.
- An interactive report, frontend application, or live controls embedded in the HTML output.
- Multi-seed significance testing, CPU/MPS bitwise equality, extensive hyperparameter search, or
  research-benchmark reproduction.
- Redistribution of MovieLens data or generated artifacts containing the source dataset.

## Further Notes

- The intended claim is "diagnose retrieval bottlenecks and provide graceful user cold-start
  routing," not "solve cold start."
- LHF is allowed to underperform the best single retriever. A leakage-free comparison and an
  honest explanation are completion criteria; a forced uplift is not.
- The full lifecycle may take hours on local hardware. The fast profile exists to keep the same
  system contract verifiable during development.
- The implementation roadmap is maintained separately and orders work as demoable tracer
  bullets rather than horizontal infrastructure layers.
