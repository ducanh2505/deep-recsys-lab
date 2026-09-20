# Declare external dependencies by owner and phase

External systems and resources may affect Dataset Adaptation, Fit or export, Artifact
inference, evaluation, or a Serving Adapter. Treating all of them as Artifact runtime
dependencies would make an offline-only dependency appear necessary for serving; leaving
them implicit in code or environment would make results difficult to interpret or reproduce.
The Lab therefore uses one lifecycle-neutral pair of value objects rather than parallel
Build Dependency and Runtime Dependency types.

An External Dependency Specification is embedded by exactly one domain owner rather than
registered globally. It declares a canonical key, exactly one usage phase, provider or
protocol, required capability or operation, exact stable semantic model or resource revision,
output-affecting configuration, and intentional stochastic behaviour and seed guarantee.
Merely accepting a seed argument does not establish seed control. Locations, credentials,
and secrets remain execution inputs and do not enter semantic identity. The same provider or
model used in multiple phases receives a separate declaration and canonical key for each
phase; multi-phase Specifications and implicit cross-phase sharing are unsupported. This
small amount of repeated configuration prevents an offline dependency change from silently
altering an Artifact's serving requirements.

The boundary is ownership of immutable bytes. When an Artifact contains every byte required
from a model or resource and those bytes participate in Artifact Identity, the resource is
payload rather than an inference-phase External Dependency; its original source revision is
retained as provenance. A URI, model name, or machine-local cache does not make an Artifact
self-contained. This permits larger portable Artifacts without pretending that an unresolved
external locator is packaged content.

Language packages, frameworks, system libraries, drivers, and hardware remain Execution
Environment conditions rather than External Dependencies. An Artifact may declare their
compatibility constraints, while the actual resolved versions are recorded in the Execution
Environment Snapshot. If an export strategy bundles runtime or package bytes into an
immutable container whose digest participates in Artifact Identity, those bytes are Artifact
payload; an execution-harness container that is not the Artifact remains environment.

Ownership determines lifecycle and identity. Dataset Adaptation owns preparation-phase
Specifications and retains resolved Snapshots as Dataset provenance; identical canonical
content and schema still have one Dataset identity. An Approach Specification declares
dependency slots and an Approach Variant binds exact Fit, export, and inference
Specifications. Changing a Fit or export Specification changes Variant identity, while an
inference Specification also enters the produced Artifact's identity and serving
requirements. An Evaluation Protocol owns evaluation-only Specifications. A Serving
Qualification Specification owns dependencies used only by its Adapter. This adds some
owner-specific plumbing but prevents offline dependencies from leaking across domain
boundaries and avoids duplicating nearly identical dependency schemas.

An External Dependency Snapshot belongs to the concrete execution of one Specification at
one usage phase. There is exactly one logical Snapshot for each combination of owning
execution identity, Specification identity, and usage phase; all invocations in that scope
reference it rather than creating per-invocation Snapshots. It records the resolved immutable
revision or digest, service/runtime version, relevant observed configuration, and requested
seed or sampling settings. Dataset provenance, a Run, a Refit Build, or a Serving Trial
retains it according to the owner. A different resolution within the same scope violates the
stable-dependency assumption and rejects the relevant phase rather than creating another
Snapshot and mixing evidence. A revision or configuration mismatch is not interpreted as
dependency unavailability. Raw requests and responses are excluded from the Snapshot and
from default evidence. A study that needs them for investigation predeclares them as
Diagnostic Observations with a versioned schema, request correlation, redaction, and
retention scope. Replaying a retained response does not constitute fresh inference or
Repeatability Evidence.

Call counts, token counts, latency, and other consumption are not Snapshot fields because
they accumulate during execution. Runs, Refit Builds, and Serving Trials retain required raw
resource evidence grouped by dependency canonical key and usage phase; totals are the default,
while per-invocation usage is retained only when the governing plan predeclares it. Missing a
required dimension makes the corresponding efficiency or derived monetary-cost claim
inconclusive without invalidating independent quality evidence. Monetary cost remains a
time-sensitive derivation rather than dependency identity.

The current Lab assumes a declared dependency stays reachable and behaviourally stable
during execution. Provider outage, mutable-alias drift, retry, timeout, circuit breaking,
dependency-failure degradation, time-scoped evidence, and cross-Variant temporal scheduling
are deferred until a concrete reliability use case exists. Dependencies without controllable
seeds remain supported, but their Runs forgo exact-reproduction claims and repeatability
requires multiple fresh planned invocations under the same stable Specification and aligned
Snapshots.

For inference, exact Run-local memoization keyed by the Run identity, every semantic input,
the Artifact, and resolved dependency Snapshots may avoid duplicate work or resume that same
Run; every reuse is recorded in the Inference Receipt. An outcome from one Run cannot
complete another Run even when all other hashes match. A memoized outcome is not fresh
inference and cannot support independent Repeatability or serving-performance evidence.
Recomputing Measurements from a retained Outcome Set performs no new inference and remains
allowed. TTL-based, approximate, stale, transport, and otherwise output-affecting caches are
deferred until a concrete workflow requires them. Serving Workloads and Qualifications
therefore do not model cache state or cache-hit mixtures.
