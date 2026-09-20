# Keep sensitive source identifiers out of datasets and artifacts

Dataset Adaptation explicitly declares which identifier and metadata fields may enter
canonical Dataset content. Sensitive source identifiers are replaced with Dataset-scoped
opaque identifiers before materialization, and re-identification mappings remain outside
Datasets, Approach Artifacts, and Git. Public benchmark identifiers may be retained only when
an explicit license and privacy declaration permits them. Approach Artifacts inherit this
boundary and may package only permitted canonical fields.

Keeping original identifiers would simplify debugging, source reconciliation, and direct
serving integration, but immutable Datasets and content-addressed Artifacts are designed to be
copied and retained. Once sensitive values enter those copies, later deletion or access
revocation cannot reliably recover them. The Lab accepts an explicit allowlist and an external
mapping boundary in exchange for reducing privacy leakage and accidental redistribution.
Serving integrations that require system-specific identifiers must therefore map at an
authorized boundary rather than relying on sensitive values embedded in the Artifact.

That authorized boundary is the Serving Adapter. It may use an external mapping dependency
to translate incoming Subject identifiers and outgoing Candidate identifiers while leaving
recommendation semantics unchanged. The Serving Qualification Specification owns the
Adapter-phase External Dependency Specification and its missing-mapping policy; each Trial
records the resolved Snapshot, not the sensitive mapping contents.
Translation cannot silently drop, substitute, or reorder outputs. A reusable Identity
Resolver domain object is deferred until multiple adapters or providers require a shared
contract and lifecycle.

The initial missing-Candidate policy fails the complete serving request even when only one
recommended identifier cannot be translated. Returning a shorter list or substituting a
fallback would create a new output behaviour after Approach inference and could hide catalog
drift. Partial mapping is therefore deferred until a concrete Serving Contract specifies its
semantics and the combined Adapter behaviour is qualified independently.

A missing incoming Subject mapping is also an explicit request failure, not permission for
the Adapter to choose popularity, anonymous, or history-only behaviour. Cold-start remains a
separate Problem-defined query form and declared Approach capability selected by the caller.
This keeps identity-integration failure from silently changing recommendation semantics.

Each canonical identifier field has one scalar type throughout a Dataset, although different
fields may choose different types. Dataset Adaptation may explicitly normalize source values
before materialization, but mixed values such as integer `42` and string `"42"` in one field
are rejected rather than treated as distinct entities. This gives canonical ordering and
serving lookup one visible schema meaning; changing normalization or type changes canonical
Dataset content instead of silently reinterpreting an existing Dataset.

Identifier normalization must also be injective over the source identifiers observed during
Adaptation. If two distinct source values map to one canonical value, Adaptation fails rather
than merging their observations; a source that genuinely distinguishes them requires a
collision-free encoding or mapping before Dataset materialization.

The Lab is not a production product and does not add consent, retention, erasure, or
tombstone lifecycles. It assumes public, synthetic, non-sensitive, or otherwise
user-authorized local research inputs and makes no production privacy-compliance claim. This
keeps the current data model small; a concrete private-data use case would have to revisit
the assumption before claiming compliant operation.
