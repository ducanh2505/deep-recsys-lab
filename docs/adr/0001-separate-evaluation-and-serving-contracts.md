# Separate evaluation and serving contracts

Evaluation and serving share a Recommendation Approach's inference core but use separate
request and policy contracts. Reusing the public serving schema would simplify orchestration
and make parity obvious, but it would mix evaluation partitioning, judgment construction,
and sampled-candidate measurement with a production-facing API; separate contracts preserve
inference consistency while allowing each boundary to own its distinct policies. The
Experiment owns its Evaluation Protocol; the Recommendation Approach does not. The Lab
assumes every member of the exact Candidate Catalog is available for every serving and
evaluation invocation, so neither caller nor Protocol supplies request-specific inventory or
eligibility constraints. The shared inference implementation operates against the full
Catalog while remaining free to use an internal retrieval subset. This deliberately omits
business-domain availability policy from the Lab.
The Serving Contract is transport-neutral: HTTP, gRPC, batch, or framework integrations are
replaceable Serving Adapters that own wire-level concerns. Evaluation invokes the shared
Approach Inference Interface directly rather than using a transport adapter. A Serving
Contract belongs to one exact Problem Specification rather than one Approach: it declares
the capabilities required for multiple Approach Variants to conform, so Approach-specific
extensions require an explicit Contract version instead of changing client-visible fields
according to the loaded Artifact.

History windows have two distinct owners. By default an Evaluation Slice supplies all
eligible history before its cutoff; a deliberate query-history window is Evaluation Protocol
identity and is shared across compared Variants. An Approach may then consume only an
identity-bearing model-context window as part of its Variant semantics. Evaluation invokes
the deep interface directly and never inherits a Serving Adapter payload guard, while an
Adapter may reject an explicitly out-of-contract transport request but never truncate it.
The current scope declares no history-count guard, so the implementation's 5,000-event request
limit and evaluator truncation are conflicts to remove.
The current scope assumes declared External Dependencies are reachable and stable. Evaluation
and Serving Adapters therefore model neither dependency retries nor whole-request retry,
timeout, circuit-breaking policy, or output-affecting transport caches; these concerns are
deferred until a concrete reliability or cache-behaviour use case requires them at both
boundaries.
Operational latency is named by its measured boundary. Approach Inference Latency includes
the complete Approach call and its internal dependencies but excludes the Adapter;
End-to-End Serving Latency includes Adapter queueing and processing and is required for a
Serving Qualification, with external network time included only from a declared client-side
observation point. A core scorer microbenchmark remains diagnostic evidence and cannot be
presented as serving latency.
Capacity and SLO Qualifications use an open-loop arrival process so slower responses do not
silently reduce offered load and hide queueing. The Workload records offered load,
backpressure, and overload thresholds, while evidence reports offered rate, achieved
throughput, latency percentiles, failures, queueing, and saturation together. Closed-loop
tests remain valid for explicitly modeled interactive clients or component microbenchmarks,
but do not support an open-loop capacity claim.
