# Use problem-specific approach inference interfaces

An executable Recommendation Approach exposes the deep inference interface defined by its
target Recommendation Problem rather than a Lab-wide interface that returns a full-catalog
score vector. Evaluation and serving adapt their separate request and policy contracts to
this shared inference seam, which accepts problem-defined query evidence and returns
problem-defined Recommendations against the complete Candidate Catalog bound to the
Artifact. Callers cannot supply request-specific candidate subsets. A universal score vector
would simplify some ranking metrics, fusion strategies, and diagnostics, but it
hard-codes catalog-ranking assumptions and makes external orchestration coordinate internal
pipeline stages. Intermediate candidates and scores therefore remain private unless an
explicit diagnostic contract exposes them. The Lab standardizes only minimal version and
provenance fields for such observations; specialized payload schemas are introduced only
for concrete research or evaluation needs rather than anticipated component taxonomies.
This choice makes generic orchestration Problem-aware and requires adapters for each
Problem in exchange for keeping future Problems and Approach implementations free to use
deeper, domain-appropriate interfaces without creating speculative maintenance burden.
Serving Adapters remain outside this interface and map transport details onto the separate,
transport-neutral Serving Contract. Operational identity, resolved dependency references,
and actual inference mode are carried by a minimal Inference Receipt beside the
Problem-defined output rather than becoming universal Recommendation fields.

The Approach Specification owns final ordering, including resolution of ties; a configurable
ordering rule is an identity-bearing Approach Variant binding. The inference interface
therefore returns the final ordered Recommendations, and evaluation or serving may not apply
a shared score-vector Top-K, tie-breaker, or other reordering policy. Otherwise an external
adapter could silently change measured and served Approach semantics.

For score-based Approaches, tie resolution is deterministic by default. Randomized
tie-breaking is permitted only as an explicitly declared stochastic mechanism whose stable
key, seed semantics, and configurable choices participate in Approach Specification or
Variant identity rather than as runtime entropy introduced by an adapter.

Current score-based Approach Specifications use ascending canonical Candidate identifier as
their transparent deterministic tie-break. They may share an implementation helper, but each
Specification owns and declares the rule; an internal array index is not the semantic key,
and another deterministic or stochastic rule represents different Approach semantics.
