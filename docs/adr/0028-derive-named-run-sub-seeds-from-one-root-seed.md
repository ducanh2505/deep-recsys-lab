# Derive named Run sub-seeds from one root seed

Each Run accepts one root seed and uses a versioned, domain-separated derivation rule to
resolve named sub-seeds for every declared stochastic mechanism, including parameter
initialization, batch ordering, surrogate sampling, and dropout. The Reproducibility Envelope
retains the root, derivation version, mechanism names, and resolved values. Evaluation
Partition and synthetic-data generation seeds remain independent scopes and are never derived
from the Run root.

Requiring every sub-seed in authoring configuration would be explicit but noisy, while
passing one raw seed to every library couples otherwise unrelated random streams: adding a
draw to sampling could silently change initialization or dropout. Named derivation keeps the
common Run interface small and isolates streams at the cost of maintaining stable mechanism
names and derivation versions. Persisting resolved values protects old Runs from future
derivation changes and makes the seed lineage auditable rather than relying on a global
same-seed guarantee.

Mechanism keys are stable semantic identifiers declared by the owning Specification. Adding
or removing a stochastic mechanism changes that Specification or Variant. A source-code
rename that does not change behaviour preserves the canonical key—or provides a compatible
migration—so it cannot silently redirect the random stream of an existing Run.

The normal Run interface exposes only the root seed. A controlled Experiment may freeze a
named sub-seed override as a planned Run dimension—for example, varying initialization while
holding surrogate sampling fixed—and records the complete resolved seed map. An execution may
not introduce an ad-hoc sub-seed override after the Experiment Specification freezes.
