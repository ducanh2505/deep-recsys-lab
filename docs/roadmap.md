# Implementation Roadmap

This roadmap turns the approved specification into tracer-bullet slices. Every slice leaves a
demoable path working; neural approaches can be developed in parallel after the common
retrieval contract exists.

**Parent specification:** [GitHub issue #33](https://github.com/ducanh2505/deep-recsys-lab/issues/33)

## 01. Popularity lifecycle tracer bullet ([#34](https://github.com/ducanh2505/deep-recsys-lab/issues/34))

**Blocked by:** None.

**What it delivers:** A reduced MovieLens fixture is replayed through the local Kafka broker,
consumed into the Event Store, materialized as a Data Snapshot, converted to Positive
Interactions, fitted with Popularity, served through the recommendation API, and summarized in
a minimal static report.

**Acceptance:**

- One high-level fast command completes the path from source events to report.
- Replaying the same events does not change the materialized snapshot.
- Known, history-only, and empty-history requests return unseen Candidates through defined
  routes.

## 02. Time-respecting evaluation and ItemKNN ([#35](https://github.com/ducanh2505/deep-recsys-lab/issues/35))

**Blocked by:** [#34](https://github.com/ducanh2505/deep-recsys-lab/issues/34).

**What it delivers:** The 50% Data Snapshot and its 50-60% Future Window become an executable
evaluation stage, with ItemKNN adding a personalized Candidate Pool and the report separating
Retrieval Coverage from final ranking quality.

**Acceptance:**

- Temporal leakage checks prove that the Future Window cannot affect fitted state or features.
- Popularity and ItemKNN are compared on the same deterministic cohort.
- RRF and Oracle Union diagnostics establish the initial fusion headroom.

## 03. Immutable Serving Artifacts and activation ([#36](https://github.com/ducanh2505/deep-recsys-lab/issues/36))

**Blocked by:** [#35](https://github.com/ducanh2505/deep-recsys-lab/issues/35).

**What it delivers:** A Lifecycle Stage exports self-describing immutable artifacts, validates
them with smoke queries, activates a successful artifact, and lets the CPU API load it without
training dependencies.

**Acceptance:**

- Manifests trace data, configuration, seed, device, timings, metrics, payloads, and source
  revision.
- A partial or incompatible artifact cannot become active.
- Restarting the API preserves recommendation behaviour for the same artifact.

## 04. Mult-VAE vertical slice ([#37](https://github.com/ducanh2505/deep-recsys-lab/issues/37))

**Blocked by:** [#35](https://github.com/ducanh2505/deep-recsys-lab/issues/35).

**What it delivers:** Mult-VAE trains with automatic MPS preference and CPU fallback, exports a
serving-compatible payload, contributes Top-200 Candidates for Known-User and History-Only
Queries, and appears in quality and resource sections of the report.

**Acceptance:**

- Both query modes use the same declared profile semantics and exclude observed Candidates.
- The artifact records the actual device, duration, and fallback reason when applicable.
- The fast profile runs without requiring an MPS device.

## 05. LightGCN vertical slice ([#38](https://github.com/ducanh2505/deep-recsys-lab/issues/38))

**Blocked by:** [#35](https://github.com/ducanh2505/deep-recsys-lab/issues/35).

**What it delivers:** LightGCN trains with measured MPS compatibility and CPU fallback, exports
a CPU-serving payload, contributes Top-200 Candidates only for Known-User Queries, and appears
in quality and resource comparisons.

**Acceptance:**

- Known Subjects receive deterministic full-catalog Candidate scores with observed Movies
  excluded.
- History-Only and Empty-History Queries never pretend to have a LightGCN identity.
- Device compatibility or fallback is explicit rather than silently changing execution.

## 06. Query-mode Learned Hybrid Fusion ([#39](https://github.com/ducanh2505/deep-recsys-lab/issues/39))

**Blocked by:** [#36](https://github.com/ducanh2505/deep-recsys-lab/issues/36),
[#37](https://github.com/ducanh2505/deep-recsys-lab/issues/37), and
[#38](https://github.com/ducanh2505/deep-recsys-lab/issues/38).

**What it delivers:** Validation outcomes train separate Known-User and History-Only LHF
classifiers over the appropriate retriever unions. LHF produces the final Top-200 and Top-10
orders, with RRF, best-single, and Oracle Union comparisons in the report.

**Acceptance:**

- Every feature is reconstructible from the training snapshot, catalog, and request-time
  retriever outputs.
- A Gold Candidate absent from the union remains unrecoverable and is counted as a retrieval
  failure.
- Coverage, conditional success, end-to-end Recall, NDCG, and realized headroom agree on
  hand-checkable fixtures.

## 07. Rolling 50-to-100 lifecycle ([#40](https://github.com/ducanh2505/deep-recsys-lab/issues/40))

**Blocked by:** [#39](https://github.com/ducanh2505/deep-recsys-lab/issues/39).

**What it delivers:** One orchestration command evaluates, ingests, retrains, exports, and
reports the 50%, 60%, 70%, 80%, 90%, and 100% Lifecycle Stages with rolling LHF validation and
no future leakage.

**Acceptance:**

- Stages 50-90 are evaluated against their next Future Window before it is ingested.
- The 100% artifact is activated after smoke tests and carries no unsupported future-quality
  claim.
- Interrupted completed stages can be reused without mutating their artifacts.

## 08. Portfolio report and latency evidence ([#41](https://github.com/ducanh2505/deep-recsys-lab/issues/41))

**Blocked by:** [#40](https://github.com/ducanh2505/deep-recsys-lab/issues/40).

**What it delivers:** The final self-contained HTML report presents architecture, ingestion,
snapshot progression, retrieval bottlenecks, query-mode and cold-user quality, resource use,
serving latency, failure cases, limitations, and complete provenance.

**Acceptance:**

- Warm CPU benchmarks report p50, p95, p99, throughput, hardware, concurrency, and SLO status
  separately for all three Query modes.
- Missing metrics or failed SLOs remain visible and do not prevent report generation.
- A fresh checkout can follow the documented fast-profile path to reproduce a valid report.

## Dependency frontier

```text
01 → 02 → 03 ─┐
          ├─04 ├→ 06 → 07 → 08
          └─05 ┘
```

After ticket 02, tickets 03, 04, and 05 can proceed independently. Ticket 06 is the first point
that requires every final retriever and the artifact contract.
