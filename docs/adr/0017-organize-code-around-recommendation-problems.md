# Organize code around recommendation problems

Problem-specific code is organized under Recommendation Problem boundaries rather than
Lab-wide technical-stage packages. Each Problem Implementation colocates its Approach
implementations, component roles, data-binding validation, evaluation mechanics, inference
interface, and serving contracts; only artifact storage, experiment lifecycle, provenance,
content identity, and other semantics proven independent of Problems remain in generic core.
Global model, retrieval, fusion, and reranking packages would make current Personalized
Top-N roles easy to discover and share, but would pressure future tasks such as Rating
Prediction to conform to a pipeline they do not have. Provider, dataset-source, and transport
integrations may remain outside a Problem when genuinely neutral. Shared abstractions are
extracted only after at least two Problems need the same semantics, accepting some early
duplication in exchange for deep Problem modules and avoiding premature universal contracts.
