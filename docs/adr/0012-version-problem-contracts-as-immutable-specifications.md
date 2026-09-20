# Version problem contracts as immutable specifications

A Recommendation Problem is a durable conceptual family, while each executable and
evaluable contract is an immutable, content-identified Problem Specification. Approach
Specifications, Problem–Dataset Bindings, Evaluation Protocols, Serving Contracts, and
Experiments reference the exact Specification they implement. Mutating one shared Problem
schema would reduce version management but could reinterpret historical Artifacts and
Results; treating every contract revision as an unrelated Problem would preserve history
but fragment the stable task vocabulary. Semantic contract changes therefore create a new
Specification within the same Problem family unless the core task or output objective also
changes. Human-readable versions aid navigation without being the identity, and descriptive
metadata may change independently. Specifications are exact-match compatible by default;
same-family names and apparent schema compatibility do not authorize Artifact reuse.
Compatibility declarations, adapters, and their contract tests are introduced only for
concrete consumers, with semantics-changing adapters included in Artifact identity. This
contract uses a hybrid representation: serializable manifest data carries semantic identity,
while a separately snapshotted Problem Implementation provides validation, codecs, Binding
checks, and executable boundaries under conformance tests. Making executable class paths the
identity would couple history to package structure, while forcing every rule into data would
produce an increasingly complex schema language. This adds compatibility, migration, and
conformance work in exchange for stable historical meaning and explicit contract evolution
without a speculative compatibility framework.
