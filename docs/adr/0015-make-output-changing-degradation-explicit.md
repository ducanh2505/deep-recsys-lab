# Make output-changing degradation explicit

An output-changing alternative behaviour is a named Degradation Mode in the Approach
Specification, not an implicit choice in a Serving Adapter. The Approach Variant binds each
Mode, the Serving Contract declares which Modes callers may receive, the Inference Receipt
identifies the Mode actually used, and Evaluation and Serving Qualification evidence remains
separate by Mode. Silent substitution would report different recommendation semantics under
one apparent contract and transfer primary-mode evidence to untested output.

The current scope does not invoke Degradation Modes for External Dependency failure: declared
dependencies are assumed reachable and stable, while retry, timeout, circuit breaking, and
outage recovery are deferred. Modes remain available only for another explicitly researched
output behaviour, such as resource-pressure operation, and require their own evidence. If no
permitted qualified Mode can run, inference fails explicitly rather than changing semantics
without attribution.

A normal data condition inside otherwise valid input is different from degradation. An
Approach may declare an Algorithmic Backoff as part of its primary inference semantics—for
example, use global popularity when a valid last item has no outgoing transition—but its
Approach Variant must bind that rule, each invocation's Inference Receipt must identify an
applied Backoff by its stable key, and evaluation must report affected outcomes separately.
Missing required history is not such a condition: it is an unsupported query form unless the
Problem and Approach explicitly define a cold-start capability, so it may not be silently
converted into popularity output.

An Algorithmic Backoff does not change Evaluation Cohort membership. Comparative Measurements
continue to cover the full Cohort fixed before inference; its application rate and conditional
metrics are Variant-local outcome breakdowns derived from retained outcomes and may not be
presented as Direct Comparison evidence across different post-inference case sets.
