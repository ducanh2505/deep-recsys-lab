# Retain resource usage and derive monetary cost

Decision-bearing Runs, Refit Builds, and Serving Trials retain the raw resource dimensions
required by their governing plans and bind them to the actual Execution Environment Snapshot.
External Dependency usage is grouped by dependency canonical key and usage phase; totals are
retained by default, while per-invocation usage requires an explicit predeclared need. Missing
required usage makes the corresponding resource or monetary-cost claim inconclusive without
invalidating independent quality or correctness evidence. Raw usage is durable evidence;
monetary cost is derived from explicit, time-sensitive pricing assumptions rather than made a
property of an Approach Variant, Artifact, Run, Refit Build, Trial, or Result.

Capturing resource evidence during execution adds instrumentation and storage overhead, but
the evidence generally cannot be reconstructed reliably after an execution has finished.
Separating usage from price also prevents provider- and date-specific prices from silently
changing the meaning of historical evidence, at the cost of requiring a later derivation when
a monetary comparison is needed. A reusable Cost Model or Pricing Snapshot is deferred until
a concrete provider, billing, or reporting workflow requires one.
