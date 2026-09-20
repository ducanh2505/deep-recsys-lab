# Declare approach capabilities before execution

Each Approach Variant declares its Problem-specific capability contract before a Run, and
the resulting Approach Artifact validates and exposes that contract rather than discovering
its public capabilities after training. Automatically intersecting component capabilities
would make arbitrary compositions easier to construct, but it could silently change which
evaluation cases are valid and which serving requests an Artifact can handle. A composition
that cannot satisfy a declared capability therefore fails validation; implementation
abilities beyond the declaration remain internal. This adds up-front specification and
composition checks in exchange for stable Experiment semantics, predictable serving
compatibility, and failures that occur before results are interpreted or an Artifact is
adopted.

Evaluation Cohort membership is materialized from the Problem, Dataset, Partitions, and
Protocol before Approach inference and is shared by every Variant in a Direct Comparison.
The Experiment is rejected before its first Run when a Variant's declared capability cannot
cover an assigned Slice or Cohort. If a Variant declares coverage but later fails on an
included case, the outcome is an Inference Failure rather than permission to shrink that
Variant's cohort.

Query evidence requirements belong to the Evaluation Slice's Problem-defined query form. A
case with a qualifying Judgment but without the evidence required to construct that query is
excluded while the shared Cohort is materialized, with
`missing_required_query_evidence` reported as its reason. The evaluator cannot reinterpret it
as cold-start or another query form; evaluating such cases requires a separate Slice and all
compared Variants must declare that capability.

Every Variant is evaluated against the same complete Candidate Catalog. An Approach may be
unable to represent or retrieve some members, but this remains observable Approach behaviour:
a relevant miss affects the declared metric. For current Personalized Top-N Ranking, `N` is
an exact output count: a Protocol or request with `N` larger than the bound Catalog is invalid
before inference, while a shorter output for a valid `N` is an Inference Failure. Internal
top-M pools never redefine the Catalog, and neither caller nor evaluator intersects the
Catalog with Approach-specific coverage.
