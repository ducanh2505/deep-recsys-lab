# Use immutable content-addressed datasets

Experiments bind immutable Datasets identified from canonical adapted observations, schema
semantics, and declared source release or generation inputs rather than mutable names or
filesystem paths. Dataset Source provenance retains acquisition, license, citation, raw
payload identity, checksums, locators, materialization instructions, and adapter details,
while an adapter code change that yields identical canonical content and semantics does not
alone create a new Dataset identity. Treating a path or human-readable release name as
identity would avoid hashing and manifest maintenance but could silently change historical
Experiments; including Evaluation Partition assignments would make caching simple but
fragment one observation collection across protocols. The Lab accepts canonicalization,
hashing, and provenance costs in exchange for stable references, deduplication, and verifiable
experimental inputs, with Evaluation Partitions remaining independently identified under
ADR-0008.

Canonical Datasets preserve distinct event-level observations. Binary collapse, frequency or
value summation, weighting, and time decay interpret those observations as Preference
Signals and therefore belong to an Approach Variant's Signal Preparation. A shared prepared
matrix may be a derived implementation representation only when its aggregation semantics
are selected by the Variant; it cannot silently become the Dataset contract. Centralizing one
matrix is computationally convenient, but would make otherwise different Approaches inherit
the same hidden modeling assumption and prevent the Dataset from supporting legitimate
alternative interpretations. The current `PreparedDataset.train` construction sums duplicate
subject-candidate coordinates before an Approach selects its policy and is an implementation
conflict to separate from canonical Dataset preparation. The initial correction keeps the
Dataset boundary event-level and has each Approach build its required sparse representation
inside Signal Preparation. It introduces no derived-data domain object or cache; rebuilding
is the simple default, and a cache keyed by Dataset, Partition, and Signal Preparation
identity is deferred until measured cost justifies its lifecycle and maintenance.

Canonical Dataset payloads also keep event observations separate from a Candidate data table
keyed uniquely by canonical Candidate identifier. Both payloads share the Dataset identity
and lifecycle; the table is not another domain object and may contain identifier-only rows.
Candidate fields originate from a Source-owned candidate entity or an explicitly declared
Dataset Adaptation. A denormalized event source may supply a Candidate field only when
Adaptation verifies that it is invariant for that Candidate. Conflicting repeated values fail
Adaptation, and event-specific fields such as individual review text never become Candidate
data through first-row or last-row selection. This requires extra source joins and validation,
but prevents row-order-dependent content, held-out-event leakage, and false Candidate
semantics. Content-based Approaches may therefore reject an otherwise valid Dataset whose
Candidate table lacks their required declared content.

The Candidate table is authoritative for Candidate Catalog membership. Every event Candidate
reference must resolve to exactly one table row, while rows without events remain valid
always-available Catalog members. For an event-only Source, Adaptation explicitly
materializes an identifier-only table from distinct referenced Candidate identifiers;
downstream code never infers a union, intersection, or last-observed catalog. This makes
zero-event Candidates representable and validation unambiguous at the cost of one required
canonical table. Subject membership remains event-derived until a concrete subject-feature or
cold-subject use case justifies a corresponding subject-data payload.

Migration does not legitimize the existing sum of duplicate coordinates as a universal
default. Every fitted Approach must explicitly bind its own Signal Preparation—such as
binary-positive, frequency, raw-value, or time-decayed semantics—and validation fails when it
does not. Evaluation likewise constructs its Protocol-owned Judgments directly from
event-level partition observations, applying its qualifying predicate before any binary or
graded collapse; it never reuses an Approach-derived matrix. This can make existing
underspecified configurations invalid, but avoids preserving accidental behavior as a new
contract.

The reusable seam is event-to-signal preparation, not a new shared matrix. The current
implicit-feedback Problem starts with one named binding, `binary_positive`, whose immutable
semantics filter events by `value > 0` and collapse each Subject–Candidate pair to one. An
Approach Specification declares whether it supports the binding and a Variant records the
selection. A shared in-process implementation may hide filtering, identifier mapping,
collapse, and sparse construction behind a small interface, but the resulting representation
is transient Fit input rather than a Dataset payload. No standalone Signal Preparation
object, registry, persisted representation, or cache is introduced. Additional bindings are
added only for concrete Approach needs. ADR-0029 now binds the researched
`item_cooccurrence` and `cooccurrence_svd` Variants to `binary_positive`; the initial
implicit-feedback `lightgcn` Variant likewise binds the same event predicate and pair collapse
for both bipartite adjacency construction and pairwise-positive sampling. These are explicit
Approach decisions rather than inheritance by implementation convenience.

External raw and prepared payload bytes are materialized outside Git by default. The
repository versions only provenance plus acquisition instructions, except for deliberately
included synthetic data or small fixtures whose license allows redistribution. Committing all
payloads would simplify checkout-time reproduction, but would create repository-size,
licensing, privacy, and secret-leak risks; content checksums preserve verification without
making Git the data store.
