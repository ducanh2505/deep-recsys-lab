# Shape configuration around domain specifications

Top-level configuration represents the domain Specifications that govern an execution rather
than a global sequence of model, retrieval, fusion, reranking, training, evaluation, and
serving stages. Experiment configuration binds a Problem Specification and Dataset, one or
more Approach Variants, its Evaluation Protocol and Partitions, optional Selection Procedure,
and planned Run dimensions; Refit Procedures and Serving Qualification Specifications have
their own lifecycle configurations. Technical-stage sections would make Hydra composition
and one current Top-N pipeline straightforward, but would force unrelated Problems to carry
meaningless keys and prevent generic orchestration from naturally representing multiple
Variants. Approach-specific bindings therefore remain nested and are validated by their
Approach Specification and Problem Implementation rather than interpreted by the generic
runner. Composition tools such as Hydra remain implementation mechanisms, not the domain
shape. This adds delegated validation and less globally introspectable configuration in
exchange for supporting deep Problem modules without a universal pipeline schema. Every
public Approach binding has a strict schema owned by its Approach Specification: defaults
are materialized into the normalized Variant, unknown or invalid fields fail before a Run,
and open dictionaries exist only in explicitly versioned integration namespaces. Requiring
schema changes for new parameters slows ad-hoc extension but prevents accepted-yet-unused
configuration and makes the recorded Variant match executed behaviour. Configuration has a
two-phase lifecycle: authoring accepts an exact Approach Specification plus overrides,
validates them, materializes defaults, canonicalizes semantic values, and commits an
immutable content-derived Approach Variant identity; execution accepts that Variant identity
and rejects semantic overrides, permitting only declared Run-scoped bindings such as a seed
or environment placement. Requiring a separate authoring step adds CLI friction, but avoids
an execution whose identity and effective behaviour diverge. Canonicalization remains an
internal materialization requirement rather than a burden placed on every run command.
Variant authoring emits a portable, canonical manifest, and an Experiment carries a verified
copy instead of depending on a bare identifier resolved through mandatory shared registry
state. Curated manifests may be versioned with the repository and a local content store may
cache them; a remote Variant Registry is deferred until a concrete CI or multi-machine
sharing workflow needs centralized discovery or coordination. This duplicates small
manifests, but keeps experiments portable and avoids operating registry infrastructure before
it has a demonstrated use case.
