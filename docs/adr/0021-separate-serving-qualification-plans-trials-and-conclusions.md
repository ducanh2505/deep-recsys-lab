# Separate serving qualification plans, trials, and conclusions

Serving readiness is modeled as an immutable Serving Qualification Specification frozen
before execution, multiple immutable Serving Trials that retain actual environments and raw
measurements, and a final immutable Serving Qualification that applies the predeclared rules.
A single mutable benchmark summary would require fewer records, but would permit post-hoc
criteria changes or cherry-picked executions and would hide performance variability; the
split accepts lifecycle and aggregation overhead in exchange for auditable qualification.
The Specification also freezes Trial eligibility, minimum samples and repetitions,
estimators, uncertainty handling, and any allowed violation rate. Every eligible Trial is
included; threshold criteria default to requiring each Trial's conservative estimate to
pass, while insufficient valid evidence produces an inconclusive Qualification rather than a
pass.
The Specification binds the expected inference runtime implementation as a clean Code
Snapshot or immutable external runtime/container version. Every decision-bearing Trial binds
the actual clean Code Snapshots for its inference runtime, Serving Adapter, load generator,
and measurement logic, or immutable tool/container versions when those live outside the
repository. Dirty executions remain development evidence so a Qualification cannot silently
combine measurements produced by different or unrecoverable implementations. A later
compatible runtime may execute the same Approach Artifact, but the new Artifact–runtime pair
requires fresh Trials and a new Qualification; the Artifact need not be copied or re-exported.
Before creating a decision-bearing Trial, preflight resolves every semantic document
referenced by the Artifact and verifies its content ID. Missing or mismatched references leave
local development serving possible when the strict runtime fields and payload are intact, but
make the Artifact ineligible for a Serving Trial or Qualification; no Trial is created merely
to record that precondition failure.
Decision-bearing Trials retain enough per-request timing and outcome evidence to recompute
queueing, service, end-to-end percentiles, and uncertainty, together with Trial-level resource
time series. They omit request payloads, subject identifiers, and recommendation content by
default; summary percentiles are derived evidence rather than the sole source of truth.
Trial Code Snapshots cover inference execution and collection; Qualification Code Snapshots
cover aggregation and decision logic. A collection defect invalidates the Trial and requires
fresh execution, while a semantics-preserving aggregation fix reuses the immutable raw Trials
to create a linked superseding Qualification. Deliberately changing estimator, uncertainty,
or acceptance semantics creates a new Specification and fresh Trials rather than post-hoc
reinterpretation.
