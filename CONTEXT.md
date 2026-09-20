# Recommendation Engineering Lab

An environment for researching, experimenting with, and showcasing approaches across the
recommendation-system lifecycle, together with the academic knowledge behind them. It
applies production-grade engineering standards to evaluate production applicability
without representing a concrete production recommendation product. It is also a deliberate
learning and portfolio environment for developing engineering judgment applicable to current
interviews and professional work. Design choices therefore prefer actively supported,
industry-relevant practices and make their trade-offs teachable; novelty alone is not a
selection criterion, and historically established practices remain appropriate when modern
systems still rely on them.

## Language

**Recommendation Engineering Lab**:
The overall domain: a research-and-engineering environment in which recommendation
approaches are studied, compared, and demonstrated end to end, while the maintainer practices
and can explain contemporary software and ML engineering decisions used in professional
work.
_Avoid_: Recommendation Platform, Production Recommendation System

**Recommendation Approach**:
An end-to-end, evaluable way of building and operating recommendations. It targets exactly
one Problem Specification and therefore one Recommendation Problem, and encompasses Signal
Preparation, a composition of Approach Components, and serving behaviour; it owns neither a
concrete Dataset nor a concrete Evaluation Protocol.
_Avoid_: Method, Model, Pipeline when referring to the end-to-end whole

**Composite Recommendation Approach**:
A first-class Recommendation Approach that combines the outputs of two or more declared
constituent Approach Variants into one Problem-defined recommendation output. The exact
constituents and combination semantics belong to the composite Variant identity, while
assessment and serving address one Approach Artifact for the complete composition rather
than an evaluation-only post-processing step.
Every composite Variant has at least two constituents identified by exact Approach Variant
Identity under unique stable slot keys. Its constituents target the same Problem
Specification and Problem–Dataset Binding, share one Candidate Catalog, and may include
different Variants of the same Approach but never duplicate one exact Variant. The composite
supports only the intersection of their query modes; every request invokes every constituent,
and a constituent failure fails the composite rather than silently skipping or replacing it.
A composite is materialized from committed constituent Approach Artifacts rather than fitting
those constituents again. Its own Artifact Identity binds every stable slot to the exact
constituent Artifact Identity and its combination Variant, and its payload contains the
constituents' inference material so normal inference has no child-artifact lookup. Assessment
or qualification evidence for a constituent does not transfer to the composite.
End-to-end repeated evaluation binds constituent Artifacts by the Experiment's frozen
repetition-pairing rule; assembly itself never refits them.
The initial RRF and standardized weighted-score composites additionally require every
constituent to declare an internal full-Catalog scoring capability in its Approach
Specification, resolved by its Variant and verified by its Artifact during assembly. Shared
public query modes alone do not establish composition compatibility. This internal contract
provides one finite score per Catalog Candidate for each supported composition query; it
does not extend the public Recommendation output or introduce a separate domain object.
Composition invokes that declared contract directly, never consumes captured Diagnostic
Observations as execution inputs, and never fills missing scores or infers them from a
partial ranking. A constituent lacking this capability is rejected before assembly. The
initial Pointwise LLM Reranking Variant scores only its declared pool and does not satisfy
this capability, so it cannot be a constituent of either initial fusion Variant.
The Approach Specification owns a separate parity rule for this declared internal scoring
capability. The initial rule requires exact equality of the full-Catalog score vector under
canonical Candidate mapping on predeclared verification cases or fixtures, both during
export and when a new runtime claims compatibility with a thin Artifact. Exact standalone
Candidate ordering alone is insufficient because changed score magnitudes can alter fusion
output. Public parity still requires exact ordered Candidate identifiers, and a composite
also verifies its own final ordered output. Failure of internal score parity rejects the
claimed export or runtime compatibility; the capability cannot be silently removed and
standalone output parity cannot bypass the failure. These checks introduce neither a public
score field nor another domain object. They deliberately reject numeric differences even
when standalone ranking is unchanged; score-tolerance alternatives are not part of this
initial baseline.
_Avoid_: Hybrid Model, External Fusion Step, Evaluation-only Ensemble

**Reciprocal Rank Fusion Composite Approach**:
The Composite Recommendation Approach with stable key `rrf_composite` that combines each
constituent's full-Catalog ordering through reciprocal-rank contributions. It consumes rank
evidence rather than score magnitude; constituent identities, rank semantics, and the
reciprocal-rank constant remain explicit Variant bindings.
Its initial Variant requires one finite score for every Candidate, assigns one-based average
ranks to tied scores over the full Catalog, and sums equal constituent contributions
`1 / (60 + rank)`. A constant-score constituent therefore contributes the same constant to
every Candidate and cannot create ordering evidence. Weighted RRF is unsupported; a different
rank constant creates another Variant, and final fused ties follow canonical Candidate order.
_Avoid_: Hybrid Approach, Weighted Score Fusion, Learned Fusion

**Standardized Weighted Score Fusion Composite Approach**:
The Composite Recommendation Approach with stable key `zscore_weighted_composite` that
standardizes each constituent's full-Catalog score vector and combines the standardized
values with declared weights. Score normalization and weighting are part of its semantics;
it is not raw-score addition, rank fusion, or learned fusion.
Its initial Variant computes per-request population mean and standard deviation over every
finite full-Catalog score from each constituent, maps a zero-variance vector to all zeros,
and sums the resulting z-scores with strictly positive finite slot weights normalized to one.
Equal weights are the materialized default; it performs no clipping, learned calibration, or
top-k normalization, and final fused ties follow canonical Candidate order.
_Avoid_: Weighted Fusion, Raw Score Sum, Reciprocal Rank Fusion, Learned Fusion

**Reranked Recommendation Approach**:
A first-class Recommendation Approach that obtains a Candidate pool from one exact base
Approach Artifact and applies a declared reranking component to produce the final
Problem-defined output. Its Variant declares the compatible base Approach Variant,
reranking mechanism, pool semantics, and dependency slots; its Artifact Identity additionally
binds the exact base Artifact and resolved inference requirements. Assessment and serving run
the same reranking path, and a Serving Adapter cannot enable, disable, or replace it.
A Reranked Artifact is assembled from a committed base Artifact without refitting it and
packages the base inference payload plus any reranking context payload needed for atomic
inference. The source base remains independent, its evidence does not transfer, and a model
available only through a provider or local cache remains an inference-phase External
Dependency rather than packaged payload.
End-to-end repeated evaluation binds its base Artifact by the same Experiment-owned
repetition-pairing rule used for Composite Approaches; assembly itself never refits the base.
_Avoid_: Serving Reranker, Presentation Ordering, Composite Recommendation Approach

**Pointwise LLM Reranking Approach**:
The Reranked Recommendation Approach with stable key `pointwise_llm_reranking` that asks a
declared external language model to score each Candidate in a base pool independently from
the same query context and declared Candidate content. The provider is an External Dependency
binding rather than the Approach name; model revision, prompt semantics, and output contract
remain identity-bearing, and its scores carry no default calibration claim.
Its initial Variant binds `rerank_depth = 100`: the base Artifact supplies its exact Top-100,
or the complete Catalog when smaller, every pool Candidate is reranked, and the first requested
`N` are returned. A request with `N` greater than the depth or Catalog size is rejected before
dependency invocation. The pool is not silently enlarged from request input; changing its
depth creates another Variant and Candidates outside it are outside the reranker's evidence.
This Variant does not conform to the initial Serving Contract whose permitted `N` extends
to Catalog size without a lower fixed ceiling. Serving it requires a separately identified
Serving Contract with `N <= min(100, catalog_size)` and its own Serving Qualification; the
Adapter cannot silently narrow the initial Contract according to the loaded Artifact.
An Evaluation Slice requesting `N = 10` may still evaluate it when its other compatibility
requirements hold, because evaluation compatibility is distinct from serving conformance.
It prepares query context as `positive_occurrence_sequence`, preserves repeated positive
occurrences and valid domain chronology, and retains the most recent
`context_window = 50`. A known Subject uses its Artifact-owned Fit-time suffix; supplied
history preserves declared array order and uses timestamps only for validation. Supported
query modes are the intersection with the base Artifact, while an unknown Subject or empty
prepared sequence fails. Prompts contain only the ordered `candidate_text` values for the
context and the Candidate being scored—not identifiers, interaction values, or arbitrary
metadata—and compatibility requires non-empty text for every Catalog Candidate.
The initial external-model contract binds the exact prompt, model revision or digest, and
generation settings, requests greedy decoding with `temperature = 0`, and binds a fixed seed
when the dependency supports one. Temperature alone supplies no repeatability guarantee; the
resolved Dependency Specification does. Every pool Candidate must return the closed JSON
shape `{"score": number}` with a finite value in `[0, 1]`; any missing, malformed, out-of-range,
or failed response fails the entire inference without clamping, retry, omission, or backoff.
Candidates sort by descending LLM score and preserve base-Artifact order on equal scores. The
internal score is uncalibrated pointwise ranking evidence, not a probability. Only exact
Run-local memoization under the shared dependency rules may reuse a response.
_Avoid_: Ollama Reranker, Listwise LLM Reranking, Calibrated Relevance Model

**Item Co-occurrence Approach**:
The Approach with stable key `item_cooccurrence` that projects declared
Subject–Candidate evidence into a Candidate–Candidate graph and uses its edge strengths
directly for recommendation. Its projection context, weighting, pruning, and query semantics
remain explicit Approach bindings rather than generic meanings of “graph.”
_Avoid_: Graph Co-occurrence, Generic Graph Recommender

**Co-occurrence SVD Approach**:
The Approach with stable key `cooccurrence_svd` that factorizes a declared item
co-occurrence graph through truncated singular-value decomposition and recommends from the
resulting low-rank similarities. It is distinct from Item Co-occurrence and does not denote
node2vec, random-walk embeddings, or graph neural recommendation.
_Avoid_: Graph Embeddings, Generic Graph Embedding Approach

**LightGCN Approach**:
The Approach with stable key `lightgcn` that learns Subject and Candidate representations by
propagating collaborative signals over a normalized Subject–Candidate bipartite graph. It is
not an item–item projection, Co-occurrence SVD, node2vec, or a generic graph-embedding label;
its graph construction, learning objective, and inference semantics remain explicit Approach
bindings.
_Avoid_: Graph Embeddings, Co-occurrence SVD, Generic GNN Recommender

**BPR Matrix Factorization Approach**:
The Approach with stable key `bpr_matrix_factorization` that learns free Subject and Candidate
embeddings, scores their compatibility by dot product, and fits their relative order with a
pairwise Bayesian Personalized Ranking objective. It has no graph propagation, side-feature
encoder, bias term, or explicit-rating reconstruction semantics. BPR names its objective, not
the Approach family: other representations such as LightGCN may use the same objective without
becoming matrix factorization.
_Avoid_: BPR Approach, Generic Matrix Factorization, Explicit-rating SVD

**Multinomial VAE for Collaborative Filtering Approach**:
The Approach with stable key `multivae` that encodes an unordered non-negative
Subject–Candidate profile into a diagonal-Gaussian latent distribution and reconstructs
full-Catalog ranking evidence through a multinomial likelihood with variational
regularization. Fit samples the latent posterior, while inference deterministically decodes
its mean. It is the collaborative-filtering Mult-VAE family rather than a generic,
multimodal, deterministic, or explicit-rating autoencoder.
_Avoid_: Generic VAE, Multimodal VAE, Mult-DAE, Explicit-rating Autoencoder

**ID-only Neural Two-Tower Approach**:
The Approach with stable key `id_two_tower` that independently maps learned Subject and
Candidate identifier embeddings through nonlinear towers into a shared normalized vector
space for retrieval. It is transductive: identifiers absent from its fitted Catalog or Subject
set have no representation. It does not denote a generic two-tower family with content,
metadata, context, or other feature encoders, and such an Approach must declare its own input
and cold-start semantics. Its current purpose is a research and engineering ablation against
BPR Matrix Factorization for studying nonlinear tower parameterization, unit-normalized
scores, dual-encoder export, and precomputed-Candidate retrieval; it does not support a claim
of production feature-based two-tower capability.
_Avoid_: Two-Tower Approach, Feature-based Two-Tower, Cold-start Retriever

**Self-Attentive Sequential Recommendation Approach**:
The Approach with stable key `sasrec` that maps a meaningfully ordered, bounded suffix of
Candidate-event occurrences through position-aware causal self-attention to next-Candidate
ranking evidence. It learns no required Subject-identity representation and can encode either
a stored known-Subject sequence or a request-supplied sequence. Causality, event order,
position information, and next-event semantics define its boundary; an unordered profile,
bidirectional masked-item encoder, or Subject-ID-only scorer is not SASRec.
_Avoid_: Generic Transformer Recommender, BERT-style Recommender, Attentive Set Recommender

**Global Popularity Approach**:
The Approach with stable key `global_popularity` that assigns one fitted, query-independent
score to every Candidate from aggregate qualifying evidence. Different contribution units,
weighting, or time-decay rules define Variants; personalization or request-context dependence
leaves this boundary.
_Avoid_: Popularity Fallback, Personalized Popularity, Trending without a time-decay binding

**Item k-Nearest-Neighbour Collaborative Filtering Approach**:
The Approach with stable key `item_knn` that derives Candidate–Candidate neighbourhood
strengths from declared Subject–Candidate preference evidence and aggregates them for a query
profile. It is collaborative rather than content similarity, and its matrix normalization,
neighbour selection, and aggregation remain explicit Variant bindings.
_Avoid_: Generic k-NN, Content k-NN, Item Co-occurrence Approach

**TF-IDF Content Similarity Approach**:
The Approach with stable key `tfidf_content_similarity` that represents declared Candidate
text content with term-frequency/inverse-document-frequency features and recommends by
content similarity to query Candidates. Identifier strings are not content unless a Problem–
Dataset Binding explicitly gives them that meaning.
_Avoid_: Semantic Retrieval, Collaborative Filtering, Identifier Similarity

**Feature-Hashed Token Similarity Approach**:
The Approach with stable key `hashed_token_similarity` that maps declared Candidate text tokens
into a bounded-dimensional signed feature-hash representation and retrieves by vector
similarity. It is a lexical approximation with possible hash collisions, not a learned or
pretrained semantic embedding method.
_Avoid_: Semantic Embedding, Dense Language Model Retrieval, TF-IDF

**First-Order Candidate Transition Approach**:
The Approach with stable key `first_order_transition` that estimates next-Candidate ranking
evidence solely from adjacent ordered Candidate occurrences and the query's final occurrence.
It does not denote a generic higher-order Markov model, sequence model, or unordered
co-occurrence method.
_Avoid_: Generic Markov Recommender, SASRec, Item Co-occurrence Approach

**Approach Specification**:
The semantic contract of a Recommendation Approach, declaring its target Problem
Specification, Signal Preparation and training objective, invariant component structure,
inference semantics, including any Algorithmic Backoffs and supported Degradation Modes,
together with final-order and tie-breaking semantics and configurable parameters, component
slots, runtime/export slots, and External Dependency slots with their
permitted usage phases. It supplies strict schemas, types, ranges, required fields, and
defaults for every public binding. For fitted Approaches these
bindings include optimizer semantics, learning-rate policy, batch size, fixed training budget,
any early-stopping rule, patience, and observation metric, and the export-parity comparison
rule. An Approach-owned model-context window that limits how much supplied query evidence is
consumed is likewise a semantic binding and part of Variant identity. The current ranking
rule requires exact ordered Problem-defined Candidate outputs;
tolerance is permitted only for numeric values that are themselves part of another Problem's
output contract. A declared internal full-Catalog scoring capability additionally requires
its exact score-vector parity rule; public ordering parity does not replace it.
Tie resolution is deterministic unless the Specification explicitly
declares a stochastic tie-breaking mechanism, and it declares stable canonical keys for
every stochastic mechanism.
Adding or removing a mechanism changes Approach semantics; a code-only rename preserves the
canonical key or supplies a compatible migration. Unknown fields fail unless an explicitly versioned
integration namespace permits them. Bindings allowed by the Specification create Approach
Variants; changes outside it require a new Specification.
_Avoid_: Approach Variant, Experiment Specification, Run Configuration

**Approach Variant**:
A fully specified, immutable parameterization of a Recommendation Approach selected for
evaluation. Authoring materializes it from an exact Approach Specification and proposed
bindings by validating fields, resolving defaults, and canonicalizing semantic values; its
content-derived identity covers that Specification and the resulting normalized bindings.
Its canonical representation is portable and self-contained rather than dependent on a
shared Variant Registry. Execution resolves this identity and rejects semantic overrides.
Its bindings include the runtime/export strategy and exact External Dependency
Specifications used during Fit, export, or Artifact inference, while the concrete Dataset,
Evaluation Protocol, Execution Environment Specification, and
workload remain conditions of the Experiment rather than part of the Variant. A Run seed selects a stochastic realization
and does not create a new Variant; changing the Approach's stochastic algorithm or
distribution does. Generic
Experiment orchestration preserves its normalized bindings but delegates their component
structure and validation to the target Approach Specification and Problem Implementation.
Unknown bindings fail closed, and config syntax, paths, aliases, labels, or notes remain
outside identity.
_Avoid_: Recommendation Approach, Run Configuration, Approach Artifact

**Approach Capability Contract**:
The explicit set of Problem-Specification-defined query forms or behaviours that an
Approach Variant promises before a Run begins. An Approach Artifact verifies and exposes
this declared public contract; component composition must fail validation when it cannot
satisfy the contract rather than silently narrowing it. An implementation may have
additional internal abilities, but they are not public capabilities until a Variant
declares them. An Approach may internally ignore or fail to retrieve Catalog members, but it
cannot redefine the Candidate Catalog or make membership request-specific.
_Avoid_: Runtime-discovered Capability List, Component Capability Intersection

**Recommendation Problem**:
A durable family of recommendation tasks sharing a core task and output objective, such as
Personalized Top-N Ranking. Its concrete input, feedback, Subject, Candidate, output,
constraint, component-role, and evaluation contracts live in immutable Problem
Specifications. Feedback type alone does not determine the Problem family, and the Lab does
not impose a universal pipeline or user-item schema on all families.
_Avoid_: Dataset, Model, Recommendation Approach

**Problem Specification**:
An immutable, content-identified version of a Recommendation Problem's semantic contract,
including its Subject and Candidate types, accepted inputs and feedback meaning, output
contract, capability vocabulary, Problem–Dataset Binding requirements, component roles, and
Evaluation Judgment contract. Approach Specifications, Bindings, Evaluation Protocols, and
Serving Contracts reference an exact Problem Specification. Semantic changes create a new
Specification within the same Problem family unless the core task or output objective
changes enough to constitute a new Problem; descriptive metadata may evolve independently.
Specifications use exact-match compatibility by default. Any cross-Specification reuse must
be declared and contract-tested for a concrete need, and an adapter that changes inference
semantics participates in Approach Artifact identity.
_Avoid_: Recommendation Problem Family, Mutable Problem Schema, Lab-wide Contract

**Problem Implementation**:
Code that realizes one exact Problem Specification by providing its validation, codecs,
Problem–Dataset Binding checks, and Problem-specific inference and evaluation boundaries.
The Specification remains serializable data rather than a Python class identity, while rules
that need executable behaviour stay in code. It owns the Problem-specific package boundary
that colocates its Approach implementations, component roles, evaluation mechanics, and
serving contracts instead of presenting those roles as Lab-wide technical stages. A Run
binds the implementation through its Code Snapshot, and conformance tests verify it against
the Specification. A semantics-preserving bug fix or package reorganization need not create
a new Specification; a domain-observable behaviour change does.
_Avoid_: Problem Specification, Python Class Path as Identity, Recommendation Approach

**Dataset**:
An immutable, content-addressed, problem-compatible collection of canonical observations
and related subject or candidate data selected as an experimental condition. Its identity
captures canonical content, schema semantics, and the declared source release or generation
inputs, while excluding filesystem location and experiment-specific partitioning. It retains
distinct event-level observations rather than embedding one Approach's binary, frequency,
weighted, or time-decayed representation. Changing the Dataset does not by itself create a
different Recommendation Approach.
Its canonical payload separates Interaction Events from a Candidate data table keyed uniquely
by canonical Candidate identifier; these are logical payloads under one Dataset lifecycle,
not additional domain objects. Candidate rows may contain only identifiers when the Source
has no declared content, leaving the Dataset valid but incompatible with Approaches that
require Candidate content.
The Candidate table is authoritative for Catalog membership: every event reference must
resolve to exactly one row, while a row with no events remains a valid always-available
Candidate. A Source containing only events requires Adaptation to materialize and declare an
identifier-only table from their distinct Candidate identifiers; downstream consumers never
silently union or intersect the two payloads. Subject membership remains event-derived until
a concrete subject-data use case requires another payload.
_Avoid_: Recommendation Approach, Prepared Dataset

**Dataset Source**:
The external release, data-producing system, or generator from which a Dataset originates,
together with acquisition, licensing, and citation information. A mutable path or URL names
a Source but cannot identify the Dataset used by an Experiment. Source release and
generation inputs participate in Dataset identity, while acquisition and adapter execution
details remain provenance. Provenance retains raw-payload identities, checksums, locators,
and materialization instructions, not a requirement to commit the payload bytes. External raw
and prepared datasets remain outside Git by default; intentionally versioned exceptions are
limited to synthetic data or small fixtures whose license permits redistribution. Because the
Lab is not a production product, it does not model consent, retention, erasure, or tombstone
workflows. It assumes public, synthetic, non-sensitive, or otherwise user-authorized local
research inputs; the user remains responsible for external data rights, and the Lab makes no
production privacy-compliance claim.
_Avoid_: Dataset, Research Source, Local File Path

**MovieLens 20M 2016-10 Dataset Source**:
The first external benchmark Source selected for full Lab support, with stable key
`movielens_20m_2016_10`. It binds the official `ml-20m.zip` payload whose README was generated
on 2016-10-17; provenance verifies the publisher's MD5, separately records a Lab-computed
SHA-256, citation, and usage terms, and keeps the archive outside Git. MovieLens 1M, 25M, the
April 2015 20M payload, and other editions are separate Sources rather than aliases or
automatic “latest” resolutions.
Its initial Adaptation maps every valid `ratings.csv` row to an Interaction Event and every
`movies.csv` row to the authoritative Candidate table, including Candidates having no rating.
It losslessly preserves the integer source identifiers, explicit rating value, UTC occurrence
time, title, and genres; physical row order has no chronological meaning. It does not yet
materialize `candidate_text`, and leaves tags, links, and Tag Genome outside canonical content.
A separately declared content Adaptation produces a new Dataset with `candidate_text` for
the initial TF-IDF and Feature-Hashed Token Similarity baselines. It deterministically
combines the original title, including any embedded year, with the source's actual genre
labels, omitting the `(no genres listed)` absence marker from derived text. It adds no tags,
Candidate identifiers, plot summaries, or unrelated metadata. Original title and genre
fields remain preserved; tokenization and lexical normalization remain Approach semantics.
The derived text is Dataset-owned content under the existing Dataset identity and lifecycle,
not a separate object, and the initial Dataset without text remains unchanged. This baseline
supports lexical evidence from titles, years, and genres rather than a claim of plot-content
understanding; it does not resolve the deferred MovieLens sequence baselines.
Invalid headers, values, Candidate uniqueness, or event-to-Candidate references fail
Adaptation rather than causing implicit repair or row removal. Rating meaning remains a
Problem and Approach concern: Adaptation never silently classifies a low rating as positive.
MovieLens baseline Variants that consume occurrence sequences are deferred pending a review
of data-processing choices in the original papers and a subsequent explicit modeling
decision. This includes SASRec, First-Order Candidate Transition, and the sequence context
used by Pointwise LLM Reranking. Neither the generic `positive_occurrence_sequence` binding
nor the binary-pair `rating_at_least_4` binding implicitly establishes their MovieLens
baseline. No MovieLens-specific sequence predicate, repetition policy, or simultaneous-group
handling is selected yet; the existing Source Adaptation and generic Approach semantics remain
unchanged.
_Avoid_: Generic MovieLens Dataset, Implicit-feedback Dataset, MovieLens Latest

**Dataset Adaptation**:
Source-specific translation into canonical observations while preserving their original
meaning and provenance. Its implementation version is provenance rather than Dataset
identity when it produces exactly the same canonical content and schema semantics. It does
not bind the Dataset to a Recommendation Problem, assign approach-specific preference
meaning, or create experimental partitions. It explicitly declares the identifier and
metadata fields permitted in canonical content. Each canonical identifier field uses one
declared scalar type throughout a Dataset; fields may differ, and Adaptation may explicitly
normalize source representations before materialization, but mixed types within one field
are invalid. Normalization must be injective across the observed source identifiers; a
collision fails Adaptation rather than merging entities, and preserving distinct source
identities requires a collision-free canonical mapping. Sensitive source identifiers are
replaced with Dataset-scoped opaque identifiers before materialization, and re-identification
mappings remain outside the Dataset, Approach Artifacts, and Git. Public benchmark
identifiers may be preserved only when an explicit license and privacy declaration permits
them. Repeated or identical-looking source rows remain distinct Interaction Events by
default; Adaptation may deduplicate them only when the source contract supplies a reliable
event identity or uniqueness rule, and doing so changes canonical Dataset content and
identity. When a source provides temporal information, Adaptation maps and preserves the
distinct semantics of Occurrence Time and Availability Time rather than collapsing them into
one ambiguous `timestamp`. Naive temporal values are interpreted as UTC unless a
source-specific Adaptation explicitly declares another timezone, after which canonical
temporal values are normalized to UTC; changing that interpretation changes canonical
Dataset content. For an observed Interaction Event, Availability Time cannot precede
Occurrence Time; violating that invariant fails Adaptation. This constraint does not compare
Availability Time with unrelated business dates in metadata. Adaptation may preserve a
source-provided sequence only when the source contract gives that sequence chronological
meaning. Such a sequence may provide chronology when Occurrence Time is absent or refine a
group whose Occurrence Times are equal, but it cannot contradict the order of distinct
Occurrence Times; that inconsistency fails Adaptation rather than silently selecting an
authority. File row order, identifier sorting, and implementation tie-breakers are not
temporal evidence. An external system or resource used during Adaptation is declared through
an External Dependency Specification, and the resolved Snapshot remains Dataset provenance.
When two executions produce exactly the same canonical content and schema semantics, they
retain one Dataset identity even if that provenance differs.
Candidate data comes only from Source-owned Candidate fields or an explicitly declared
Adaptation. Event-specific fields never become Candidate data by taking an arbitrary first or
last event. When a denormalized event source repeats a declared Candidate field, Adaptation
must verify one invariant value per Candidate; a conflict fails rather than choosing by row
order. `candidate_text` is then materialized deterministically from the declared Candidate
fields. MovieLens title or genres therefore require its movie catalog, and Yelp business
content requires business data rather than one review.
_Avoid_: Problem–Dataset Binding, Signal Preparation, Data Partitioning

**Problem–Dataset Binding**:
An immutable declaration that maps a Dataset's entities, fields, and preserved observation
semantics onto exactly one Problem Specification's data contract and validates their
compatibility without changing Dataset content. An Experiment binds one Problem
Specification and one Dataset through this declaration; the same Dataset may have Bindings
to multiple Problems or Specifications. It validates each mapped identifier field's declared
type against the Problem contract. A Binding used by an ordered-history query also validates
that the Dataset preserves the required chronology through Occurrence Time or a
source-provided sequence with declared temporal meaning; file row order, identifiers, and
implementation tie-breakers cannot supply missing chronology. Failure makes that query form
incompatible while leaving unordered query forms available. If compatibility requires
content transformation or loss of source meaning, Dataset Adaptation must produce a new
Dataset instead.
_Avoid_: Dataset Adaptation, Signal Preparation, Evaluation Judgment

**Signal Preparation**:
The Recommendation Approach's interpretation, filtering, transformation, or aggregation
of problem-compatible observations into Preference Signals. Changing this strategy changes
the Approach even when the Dataset remains the same. Collapsing repeated real events into a
binary value, summing their values or frequency, and applying weighting or time decay are
therefore identity-bearing Variant semantics rather than Dataset preparation. Every fitted
Approach Specification explicitly declares the Signal Preparation it supports; a Variant
missing that binding is invalid rather than receiving a generic aggregation fallback. An
implicit-feedback Popularity Variant defaults to one qualifying positive contribution per
distinct Subject–Candidate pair and ranks by distinct positive Subject count; counting every
qualifying event is a separate frequency-oriented Variant. An implicit-feedback ItemKNN
Variant likewise defaults to a binary positive Subject–Candidate incidence matrix before
computing similarity; frequency- or value-weighted matrices define separate Variants. For
the current implicit-feedback Problem, the reusable named binding `binary_positive` means
filtering individual events by `value > 0` and then collapsing each Subject–Candidate pair to
one. The MovieLens 20M 2016-10 implicit-preference binding instead uses the named
`rating_at_least_4` preparation: it filters individual rating events by `value >= 4.0` and
then performs the same binary pair collapse. Ratings below 4 remain Dataset observations but
provide neither positive nor negative Preference Signals under this preparation;
`rating_at_least_4` and `binary_positive` therefore identify different Approach Variants
rather than two names for one rule. The initial `global_popularity` Variant uses
`binary_positive` and scores every Candidate by its number of distinct positive Subjects. It
declares both `known_subject` and
`supplied_history` so it can participate in shared Evaluation Slices, but intentionally ignores
the content of every conforming query and ranks using one immutable internal full-Catalog
score vector, returning only the requested ordered Candidate identifiers.
This does not relax the Serving Contract: an unknown Subject in a subject-identity query still
fails and popularity is not an implicit cold-start fallback. Its Artifact stores only that
vector and Catalog identifier bindings, not Subject profiles or a generic train matrix. A
Candidate with no positive Fit evidence receives zero; an entirely zero vector remains a
valid degenerate research Artifact ranked by canonical tie order, with serving acceptability
left to Serving Qualification. The score is distinct-Subject support within that Artifact,
not a probability, and inference applies no normalization, time decay, query weighting, or
seen-Candidate filtering. A named binding is embedded in Variant identity rather than having
an independent lifecycle, and its derived representation is transient Fit input rather than
Dataset content. The initial `item_knn` Variant applies `binary_positive` consistently to Fit,
stored known-Subject profiles, and supplied histories. It computes Candidate-column cosine
similarity from the resulting Subject–Candidate incidence matrix, removes self-similarity,
and retains at most `max_neighbors = 100` strictly positive neighbours per source Candidate;
ties at the cutoff use canonical Candidate order. This row-wise maximum may make the stored
neighbour matrix asymmetric, while symmetric union or intersection semantics define other
Variants. Query scoring sums the retained similarities contributed by every Candidate in the
prepared profile without dividing by profile length. The Variant declares both
`known_subject` and `supplied_history`, and its Artifact stores the sparse neighbour matrix and
immutable prepared known-Subject profiles. Unknown Subjects and empty prepared profiles fail.
A non-empty profile with no retained neighbour signal remains a valid degenerate query with
zero/tied scores and canonical tie ordering rather than a fallback. It scores the complete
Catalog without seen-Candidate filtering: self-evidence is absent, but a Candidate in the
profile may receive evidence from another profile Candidate. `max_neighbors` is an upper bound,
so a smaller Catalog or fewer positive similarities requires neither clamping nor failure.
The score is uncalibrated summed cosine-neighbour evidence within one query.
The initial `tfidf_content_similarity` Variant consumes only the bound `candidate_text` field.
It applies Unicode NFKC normalization and case folding, then tokenizes Unicode word unigrams
with optional internal hyphens while retaining digits; it applies no stop-word removal,
stemming, or n-grams. Raw token counts supply term frequency, and
`idf(term) = log((1 + catalog_size) / (1 + document_frequency(term))) + 1` is fitted across
the complete Fit-time Candidate Catalog. Each Candidate vector is L2-normalized, similarity is
its cosine dot product with another Candidate, and self-similarity is removed. The Artifact
stores the complete nonzero sparse similarity matrix without neighbour truncation, together
with immutable `binary_positive` known-Subject profiles. Both `known_subject` and
`supplied_history` use `binary_positive` and sum similarities contributed by every prepared
profile Candidate. Unknown Subjects and empty profiles fail; a non-empty profile containing
only zero-content Candidates remains valid and produces zero/tied scores. Candidates already
in the profile remain output-eligible and may receive evidence from other profile Candidates.
The score is non-negative lexical-similarity evidence within one query, not a probability.
An absent `candidate_text` binding is incompatible, while an empty field produces a zero
vector and an all-empty Catalog may produce a degenerate all-zero research Artifact without
identifier or metadata fallback.
The initial `hashed_token_similarity` Variant consumes the same `candidate_text`, Unicode
normalization, case folding, and unigram tokenization as the initial TF-IDF Variant. For each
UTF-8 token it computes SHA-256, derives a bucket from the first 32 digest bits modulo
`hash_dimension = 128`, and derives a stable positive or negative sign from the next digest
byte. Signed raw token counts accumulate in that fixed-size vector; the Variant fits no
vocabulary, IDF, or learned embedding. Candidate vectors are L2-normalized and compared by
cosine similarity. Self-similarity is removed, and each source Candidate retains at most
`max_neighbors = 100` strictly positive similarities with canonical cutoff tie ordering;
negative similarities caused by collisions are discarded. Hash collisions are accepted
approximation semantics of this bounded-memory lexical baseline, not semantic understanding.
Both `known_subject` and `supplied_history` use `binary_positive` profiles and sum retained
similarities. Its Artifact stores the sparse neighbour matrix and immutable prepared known
profiles. Unknown Subjects and empty profiles fail, while non-empty zero-content profiles may
produce valid zero/tied scores. It applies no identifier fallback or seen-Candidate filtering.
The initial implicit-feedback `lightgcn` Variant uses the same `binary_positive`
Subject–Candidate set for both its normalized bipartite adjacency and the positive examples of
its pairwise training objective; repeated qualifying events contribute once, while frequency-
or value-weighted graph construction defines another Variant. The initial
`item_cooccurrence` and `cooccurrence_svd` Variants apply
`binary_positive` consistently to Fit observations, stored known-Subject profiles, and
request-supplied history; repeated qualifying events contribute once, while zero-only and
negative-only evidence contributes nothing. Both Variants declare the `known_subject` and
`supplied_history` query forms; equivalent binary-positive profiles produce the same ranking,
and an unknown Subject fails rather than falling back to history or popularity. Their Approach
Artifacts store immutable `binary_subject_profiles` derived from the Fit Partition as fitted
state for `known_subject`; runtimes do not reinterpret a generic raw or summed train matrix.
This payload is owned by and shares the lifecycle of the existing Approach Artifact rather
than introducing another domain object or cache. A known-Subject or supplied-history query
whose `binary_positive` profile is empty lacks required query evidence: inference fails,
evaluation excludes the case before inference under `missing_required_query_evidence`, and no
popularity or canonical-order fallback is permitted. Frequency- or value-weighted query
evidence requires another Variant. In contrast, a non-empty prepared profile is valid even
when its learned associations yield an all-zero or fully tied score vector; deterministic
canonical tie resolution then produces the required ordering, the case remains in evaluation,
and this is a degenerate model result rather than a backoff. A rank-deficient or all-zero
fitted graph may therefore produce a research Artifact, while Serving Qualification separately
determines whether it is acceptable for a serving use case. Their initial projection-context
binding is `whole_subject_fit_history`: after `binary_positive`, every unordered Candidate
pair in one Subject's complete Fit history contributes to the projected graph regardless of
temporal distance. The initial edge-weight binding is `distinct_subject_count`: an edge's
weight is the number of distinct Subjects whose Fit histories contain both Candidates, and
each Subject contributes at most once to a given pair. Normalizing a Subject's contribution by
history length changes graph semantics and therefore defines another Variant. The initial graph
Variants bind `min_distinct_subject_count = 1`, retaining every eligible observed edge.
Raising this pruning threshold changes fitted graph state and recommendation behavior, so the
resolved value is part of Variant identity rather than an execution-time override. They also
bind `zero_self_edges`: the projected diagonal is removed before direct scoring or
factorization, and any diagonal reintroduced by low-rank reconstruction is suppressed during
scoring, so one Candidate cannot support itself. This is not seen-Candidate filtering; a
Candidate already present in query history remains eligible and may receive evidence from
other history Candidates. The initial `item_cooccurrence` Variant binds
`sum_neighbor_weights`: for each eligible output Candidate, its score is the sum of graph-edge
weights from every Candidate in the `binary_positive` query history. A maximum, mean with
score-scale semantics, or other aggregation changes query behavior and defines another
Variant. Its runtime/export binding is `sparse_full_graph`: the Approach Artifact stores the
complete sparse graph after edge pruning and self-edge removal, and inference computes `q C`
without per-Candidate neighbor truncation. Its internal ranking score is the summed
distinct-Subject support contributed by the query profile. The initial
`cooccurrence_svd` Variant factorizes the same raw, pruned, zero-diagonal
distinct-Subject-count matrix used by `item_cooccurrence`, without popularity, marginal, PMI,
or cosine normalization. This isolates low-rank factorization as the material difference
between the initial Variants; factorizing a normalized association matrix defines another
Variant. Its factorization semantics are `rank_k_reconstruction`: for the declared rank `k`,
scores derive from the truncated reconstruction `C_k = U_k Σ_k V_kᵀ`, not from a
positive-semidefinite embedding Gram matrix. Constructing `E = V_k sqrt(Σ_k)` and scoring
with `E Eᵀ` adds different embedding semantics and is not this Variant. Its
`signed_reconstruction_scores` binding preserves signed off-diagonal values during scoring;
they are latent approximation contributions, not Explicit Negative Feedback. Clipping them to
zero or retaining only positive similarities changes the reconstruction behavior and defines
another Variant. Query scoring binds `sum_reconstructed_weights`: each eligible output
Candidate receives the sum of its signed reconstructed relationships from every Candidate in
the `binary_positive` query history, without cosine or latent-vector normalization. Its
runtime/export binding is `factorized_full_catalog`: the Approach Artifact stores `U_k`, the
singular-value vector, and `V_kᵀ`, and inference computes `q U_k Σ_k V_kᵀ` while suppressing
self-contributions. It neither materializes dense `C_k` nor applies output-changing neighbor
truncation. Its internal ranking score is signed reconstructed support. Scores from both graph
Approaches are uncalibrated ranking evidence meaningful only within one query; they are not
probabilities or confidence values and cannot be compared across queries or Approaches. Both
Variants return only the final ordered Candidate identifiers in evaluation and serving.
Their scores remain internal ranking or composition evidence and may be retained only as
explicitly declared Diagnostic Observations. Fit uses a
sparse truncated-SVD solver fixed by the Approach Specification rather than exposed as
a public binding. Any solver randomness uses a named sub-seed derived from the Run seed;
non-convergence fails the Run without switching solver or densifying the graph. A public solver
choice is deferred until a concrete solver-comparison Experiment exists. Its declared `rank`
is an exact identity-bearing binding and must satisfy
`1 <= rank < candidate_count`; an incompatible Experiment fails preflight rather than silently
clamping the value. A data matrix whose numerical rank is lower remains valid and simply
produces zero trailing singular values. The initial `cooccurrence_svd` Variant binds
`rank = 32`; a Dataset with `candidate_count <= 32` is incompatible with that Variant and must
use a separately identified lower-rank Variant. Session-, basket-, or time-window projection
contexts likewise define separate Variants. An Approach does not inherit a binding merely
because its implementation can consume the same representation. An Approach may construct
sampled surrogate negatives for a training objective, but that modeling assumption is distinct
from Explicit Feedback and from Protocol-owned Sampled
Distractor Candidates. The surrogate sampling distribution, exclusion rule, samples per
positive, and refresh schedule are semantic bindings of the Approach Variant; changing one
creates another Variant, while a Random Seed selects a Run realization of the same policy.
The initial implicit-feedback `lightgcn` Variant samples uniformly from Subjects having at
least one `binary_positive` Candidate and at least one unobserved Catalog Candidate, then
samples one of that Subject's positives uniformly and one Candidate uniformly from the
complement of the Subject's full positive set. Unobserved Candidates remain surrogate training
examples rather than Explicit Negative Feedback. Sampling uses a named sub-seed derived from
the Run seed, and Fit fails when no Subject satisfies the sampler's eligibility rule. Its
training-objective binding is `bpr_pairwise_logistic`: Fit scores a Subject–Candidate pair by
embedding dot product and minimizes `softplus(score_negative - score_positive)` over sampled
triples. This learns relative order rather than ratings or calibrated probabilities; another
objective defines another Variant. Its propagation contract constructs an undirected
Subject–Candidate bipartite adjacency without self-loops and assigns each binary edge the
symmetric-normalized weight `1 / sqrt(subject_degree * candidate_degree)`. Each layer performs
only linear propagation through that adjacency—without activation, feature transformation, or
learnable convolution weights—and final Subject and Candidate embeddings are the uniform mean
of layers `0..L`, including the learned layer-zero embeddings. Changing normalization, layer
aggregation, or adding nonlinear transformations changes Variant semantics or leaves the
LightGCN Approach boundary. The initial capacity bindings are `embedding_dimension = 32` and
`propagation_layers = 2`; changing either creates another Variant and any data-driven choice
between such Variants uses a Selection Partition. LightGCN expresses its sampled-training
budget in `sampled_passes`, not in Dataset `epochs`. One sampled pass consists of
`ceil(trainable_positive_pair_count / batch_size)` optimizer updates, where trainable positive
pairs belong to Subjects satisfying the sampler's eligibility rule; every update independently
draws a full batch of triples with replacement under the declared surrogate-sampling policy.
A sampled pass is therefore a Dataset-scaled workload unit, not an exhaustive traversal, and
does not guarantee that every trainable pair is drawn. The pass count and batch size are
identity-bearing Variant bindings; calling this unit an `epoch` is avoided because it would
incorrectly imply deterministic coverage. The initial LightGCN Variant binds
`sampled_passes = 100`, `batch_size = 128`, the Adam optimizer, and
`learning_rate = 1e-3`. It initializes layer-zero embeddings with Xavier-uniform using a
dedicated named Run sub-seed distinct from surrogate sampling. Its explicit regularization is
the batch-averaged squared L2 norm of the sampled Subject, positive-Candidate, and
negative-Candidate layer-zero embeddings with `l2 = 1e-4`; it is not weight decay over
propagated embeddings. The baseline uses the fixed sampled-pass budget without early stopping
and makes no claim of being optimal. Alternative optimization, initialization,
regularization, batch-size, or stopping bindings define other Variants, and data-driven choice
among them uses a Selection Partition. The initial LightGCN Variant declares only the
`known_subject` query form and scores with the fitted final embedding of a Subject represented
in its Approach Artifact. An unknown Subject fails explicitly; inference cannot derive a
Subject representation from supplied history, substitute popularity, or invoke another query
form. Supporting `supplied_history` requires a separately identified Variant with an explicit
and evaluated out-of-sample fold-in rule. A Protocol requiring that query form is incompatible
with the initial Variant; no generic embedding aggregation may be inferred as a fallback.
Its Approach Artifact stores the fitted final Subject and Candidate embedding matrices together
with their identifier and Candidate Catalog bindings. For a supported Subject it scores every
Candidate by the dot product of their final embeddings, without cosine normalization, sigmoid
transformation, or clipping negative values. The resulting internal ranking score is
uncalibrated ranking evidence comparable only among Candidates within the same query; it is not
a probability, confidence value, Explicit Negative Feedback, or cross-query quantity. The
Artifact does not retain the propagation graph, layer-zero embeddings, or optimizer state and
cannot resume Fit. Resumable Fit state, when captured, belongs to the existing Run-local
Training Checkpoint lifecycle and never to the Artifact; the LightGCN binding introduces no
additional domain object. A Candidate in the Fit-time Candidate Catalog remains valid when it
has no `binary_positive` Fit edge. It is unobserved for every Subject and may therefore receive
gradient as a sampled surrogate negative: BPR Matrix Factorization updates its Candidate
embedding directly, while LightGCN updates its layer-zero embedding even though its propagated
components remain zero. This is declared surrogate-negative behavior rather than a fallback or
a claim of content-based cold-Candidate support. A finite stochastic Run need not actually draw
every such Candidate; guaranteeing negative-sample coverage requires another sampler Variant.
A Candidate absent from the Artifact-bound Catalog remains unsupported because the Artifact
has no identifier binding or fitted representation for it. A known LightGCN Subject is
supported only when it has at least one prepared Fit edge. A zero-degree Subject does not by
itself invalidate Fit for other Subjects, but its serving query fails and
its evaluation case is excluded before inference under `missing_required_query_evidence`.
A supported Subject need not itself be eligible as a BPR anchor provided it has graph evidence;
Fit fails globally only when the previously declared sampler has no eligible Subject. With its
fixed sampled-pass budget, the initial Variant exports the fitted state after the final
successful optimizer update; it does not select a lower-training-loss state or otherwise treat
intermediate Training Checkpoints as candidates. A Training Checkpoint may resume an
interrupted Run while preserving its stochastic state, but it does not alter the stopping
point or selected fitted state. A non-finite loss, gradient, parameter, or final embedding
fails Fit without rollback, partial Artifact creation, or substitution of an earlier
Checkpoint. Choosing a Checkpoint from held-out Measurements instead belongs to an explicit
Selection Procedure.
The initial `bpr_matrix_factorization` Variant independently binds the same
`binary_positive` Signal Preparation, `bpr_pairwise_logistic` objective, and surrogate
sampling policy as the initial LightGCN Variant: uniformly select an eligible Subject, then
one of its prepared positives, then an unobserved Candidate from the complement of its full
prepared positive set, with replacement. Its sampler uses its own named Run sub-seed and Fit
fails when no Subject is eligible. These are explicit bindings of both Variants rather than a
Lab-wide default or semantics inherited from a shared implementation. Holding signals,
sampling, and objective constant permits an Experiment to isolate graph propagation from
plain factorization; changing any of them defines another Variant instead of silently changing
that comparison. To serve as a controlled baseline for the initial LightGCN Variant, the
initial BPR Matrix Factorization Variant also binds `embedding_dimension = 32`,
Xavier-uniform initialization with a dedicated named sub-seed, `sampled_passes = 100`,
`batch_size = 128`, the Adam optimizer, `learning_rate = 1e-3`, and `l2 = 1e-4` over the
batch-averaged squared norms of sampled Subject, positive-Candidate, and negative-Candidate
embeddings. It uses no early stopping, exports the state after the final successful optimizer
update, and fails on a non-finite loss, gradient, parameter, or final embedding without
rollback. These shared initial values support a controlled comparison and do not assert that
the Approaches share optimal hyperparameters; changing them defines independently selectable
Variants. The initial BPR Matrix Factorization Variant declares only `known_subject`; it does
not infer a Subject representation by averaging Candidate embeddings or otherwise support
`supplied_history`. A supported Subject must have at least one prepared positive and at least
one unobserved Candidate in the Fit-time Catalog, because only such Subjects can occur as BPR
anchors and receive a fitted Subject embedding. Its Artifact stores final Subject and Candidate
embedding matrices plus the supported-Subject set within its fitted state, without introducing
another domain object. It scores the complete bound Catalog by raw Subject–Candidate dot
product without normalization, sigmoid transformation, or negative-score clipping, producing
uncalibrated within-query ranking evidence. Unknown and unsupported Subjects fail explicitly.
A Direct Comparison uses one predeclared Evaluation Cohort that every compared Variant can
cover; it cannot remove BPR-MF-specific unsupported cases after execution. A Cohort containing
such a Subject makes the Experiment incompatible with this Variant before execution.
The initial `id_two_tower` Variant independently binds the same `binary_positive` Signal
Preparation, uniform eligible-Subject/positive/unobserved surrogate sampling, and
`bpr_pairwise_logistic` objective as the initial BPR Matrix Factorization Variant. Each tower
maps a learned 64-dimensional identifier embedding through an independent
`64 -> 64 -> 32` ReLU network and L2-normalizes its output. The raw dot product is therefore a
cosine score bounded to `[-1, 1]`; the initial Variant has no temperature or learned score
scale. Weights use Xavier-uniform initialization and biases are zero. Training binds
`sampled_passes = 100`, `batch_size = 128`, Adam with `learning_rate = 1e-3`, and
`l2 = 1e-4` over the sampled raw identifier embeddings only, with no tower weight decay,
dropout, or early stopping. Initialization and surrogate sampling use distinct named Run
sub-seeds. It exports the final successful update and fails on a non-finite loss, gradient,
parameter, or final vector without rollback. The Variant declares only `known_subject`; a
supported Subject requires at least one prepared positive and one unobserved Catalog Candidate,
and unknown or unsupported Subjects fail. Its Artifact stores only final unit Subject and
Candidate vectors and the supported-Subject set, not raw embeddings, tower parameters, or
training state. It scores the complete Catalog without seen-Candidate filtering and uses
those scores only as uncalibrated within-query evidence. These bindings make it a controlled nonlinear and
normalized ablation against BPR Matrix Factorization rather than a feature-based two-tower
claim.
The initial implicit-feedback `multivae` Variant binds `binary_positive` for both Fit and query
profiles: it retains event-level observations with `value > 0` and collapses each
Subject–Candidate pair to one. A non-negative count profile remains within the Multinomial VAE
Approach only through another explicitly identified Variant. Signed evidence, untransformed
explicit ratings, and the generic summed interaction matrix are incompatible with its
multinomial-likelihood semantics rather than receiving an implicit conversion. The initial
Variant declares both `known_subject` and `supplied_history` as native query forms. It learns
no Subject-identity parameter: `known_subject` runs the Artifact's immutable prepared Fit
profile through the same encoder used for a request-supplied profile. Equivalent
`binary_positive` profiles therefore produce identical scores and ranking regardless of
Subject identifier, event order, repetition, or query form. An unknown Subject identifier
fails instead of falling back to another query form; fold-in requires the caller to request
`supplied_history` explicitly. A profile that is empty after Signal Preparation lacks required
query evidence: serving fails and evaluation excludes the case before inference under
`missing_required_query_evidence`, rather than decoding encoder bias as a cold-start fallback.
Candidates represented in the non-empty profile remain eligible members of the complete
output Catalog. Fit includes only Subjects whose prepared profile is non-empty and fails
preflight when none remain; an empty profile is not used as a zero-vector training example.
Each input profile is L2-normalized before encoding. Fit-only input dropout corrupts the
encoder input while leaving the original prepared profile as the multinomial reconstruction
target, so it is denoising rather than removal of Dataset evidence. One Mult-VAE training
`epoch` means one shuffled, exhaustive pass over eligible Subject profiles: a single
permutation is partitioned into batches and every eligible profile occurs exactly once. A
Candidate with no positive Fit target remains in the full-Catalog decoder and receives
multinomial-normalization gradients; this does not imply feature-based support for a Candidate
absent from the Artifact-bound Catalog. Its initial compact architecture maps the Catalog-sized
profile through a `hidden_dimension = 64` tanh layer into diagonal-Gaussian
`latent_dimension = 32` mean and log-variance vectors, and symmetrically decodes the latent
representation through a 64-dimensional tanh layer into one logit per Catalog Candidate. It
binds Fit-only `input_dropout = 0.2`, Xavier-uniform weights, and zero biases. Fit uses a
reparameterized posterior sample, while inference deterministically decodes the posterior
mean. Its Approach Artifact uses the dense ONNX runtime, contains the immutable prepared
known-Subject profiles needed by `known_subject`, and excludes optimizer or other resumable
training state. It uses raw full-Catalog decoder logits internally without softmax, normalization,
clipping, stochastic sampling, or history masking. These logits are uncalibrated ranking
evidence comparable only among Candidates in the same query; Candidate membership in the
query profile does not remove it from the output Catalog. Architecture size, dropout, and
export runtime are identity-bearing Variant bindings, and this compact Variant makes no claim
of reproducing a paper-specific configuration. Its optimization profile binds `epochs = 100`,
`batch_size = 128`, the Adam optimizer, `learning_rate = 1e-3`, and no weight decay. The loss
is full-Catalog multinomial negative log-likelihood plus
`beta * KL(q(z|profile) || N(0, I))`. Beta follows a per-optimizer-update linear schedule from
zero to `beta_cap = 0.2` over the first 20 percent of total planned updates, then remains at
the cap; total updates derive from the fixed epoch budget and eligible-profile batches. It uses
no early stopping, exports the state after the final successful optimizer update, and fails on
a non-finite loss, gradient, parameter, or exported output without rollback. Parameter
initialization, profile ordering, input dropout, and latent reparameterization use separate
stable named sub-seeds derived from the Run root seed. Paper reproduction or another training
profile requires a separately identified Variant rather than interpreting these compact
bindings as canonical research settings.
The initial implicit-feedback `sasrec` Variant binds `positive_occurrence_sequence`: it keeps
every event-level observation with `value > 0`, preserves repeated qualifying
Subject–Candidate occurrences as distinct sequence elements, and orders them only by complete
domain-valid chronology. It does not inherit pair-collapsing `binary_positive` semantics.
A missing order or unresolved simultaneous group in any sequence required by Fit makes the
Experiment fail preflight; file row position, identifiers, and implementation sorting cannot
supply or break chronology. Changing the event predicate, deduplicating repeated occurrences,
or defining explicit simultaneous-group handling creates another Variant. Its initial
training-construction binding is `all_prefix_full_catalog_cross_entropy`: every consecutive
prefix-to-next-occurrence transition becomes one example, and a prefix longer than the
declared context window contributes only its most recent suffix. One epoch applies one shuffle
and exhaustively visits every materialized example exactly once. The objective is exact
full-Catalog categorical cross-entropy, so every non-target Candidate competes in the softmax,
including Candidates already present in the prefix; it performs no negative sampling. Fit
fails preflight if the prepared Fit sequences produce no transition. A shifted
multi-position-target construction or sampled binary objective, including paper-reproduction
semantics, defines another Variant and cannot replace this objective as an implementation
optimization. The initial Variant declares both `known_subject` and `supplied_history`, using
the same sequence encoder without a Subject-identity parameter. A supplied-history request
declares its array order as a domain-meaningful source sequence; optional timestamps validate
that order but never cause the runtime to reorder it. Timestamp order contradicting the
declared sequence makes the request invalid, while equal timestamps are unambiguous when this
declared source sequence gives them order. Query preparation applies the same event predicate,
preserves repetitions, and then retains the most recent suffix within the context window. A
known Subject uses the Artifact's immutable prepared Fit suffix. Equivalent prepared ordered
suffixes produce identical scores regardless of Subject identifier or query form. An unknown
Subject identifier fails rather than falling back to history, and fold-in requires an explicit
`supplied_history` query. An empty prepared sequence lacks required query evidence and fails;
one event is sufficient for inference. Candidates present in the input sequence remain in the
complete output Catalog. Its initial compact architecture binds `context_window = 50`, learned
absolute Candidate and position embeddings of `embedding_dimension = 32`, and two single-head
causal self-attention blocks. Each block uses scaled dot-product attention with causal and
padding masks, post-normalized residual branches, and a `32 -> 64 -> 32` point-wise GELU
feed-forward network. Dropout `0.1` applies to the projected attention output and feed-forward
output before their residual normalization. The Candidate embedding table is shared between
sequence input and output scoring. Inference uses the final non-padding sequence state to
produce raw dot-product logits for the complete Catalog. Its Approach Artifact contains the
dense ONNX scorer and immutable prepared known-Subject Fit suffixes, but no resumable training
state. The logits are uncalibrated within-query ranking evidence and are not softmaxed,
normalized, clipped, or comparable across queries. The context window is an identity-bearing
Approach Variant binding rather than a transport limit. Window length, head or block count,
normalization placement, feed-forward structure, and dropout semantics may define other
SASRec Variants only while preserving causal ordered next-event behavior. The initial
optimization profile binds `epochs = 100`, `batch_size = 128`, the Adam optimizer, constant
`learning_rate = 1e-3`, no weight decay, and no label smoothing. Candidate and position
embeddings use `Normal(0, 0.02)` initialization with the padding embedding fixed at zero;
linear weights use Xavier-uniform with zero biases, and LayerNorm uses unit scale with zero
bias. It uses no early stopping, exports the state after the final successful optimizer
update, and fails on a non-finite loss, gradient, parameter, or exported logit without
rollback. Parameter initialization, example ordering, and dropout use separate stable named
sub-seeds derived from the Run root seed. Alternative optimization, training-budget, or
checkpoint-selection semantics define another Variant or explicit Selection Procedure.
The initial `first_order_transition` Variant binds the same `positive_occurrence_sequence`
event preparation and supplied-history order contract as the initial SASRec Variant. It counts
every adjacent Fit occurrence `source -> target`, including repeated and self-transitions, then
normalizes each source row by its total outgoing count. Inference uses only the final prepared
query occurrence and ignores all earlier context. Both `known_subject` and `supplied_history`
are supported; the Artifact stores the sparse row-normalized transition matrix and immutable
last prepared Fit Candidate for each supported known Subject. Unknown Subjects and empty
prepared sequences fail. A source with no observed outgoing transition yields an internal
all-zero score vector and returns Candidate identifiers in canonical tie order, not a
popularity or other Algorithmic Backoff. A Fit
set with no transitions may likewise produce a degenerate all-zero research Artifact whose
serving acceptability belongs to Serving Qualification. The complete Catalog is scored,
self-transitions remain eligible, and seen Candidates are not filtered. A score is empirical
one-step transition frequency within the fitted evidence, not a calibrated future-event
probability.
Signal Preparation that requires a total event order is compatible only with a Fit Partition
that preserves that order. An unresolved simultaneous group makes the Experiment fail
preflight unless the Variant explicitly declares identity-bearing group-handling or exclusion
semantics; identifiers, row positions, implicit dropping, and implementation tie-breakers
cannot supply those semantics.
_Avoid_: Dataset Adaptation, Data Partitioning

**Recommendation Subject**:
The entity or request-scoped context for which a Recommendation Problem produces
recommendations. Each Problem defines its eligible subject forms, such as a user, session,
or group.
_Avoid_: User as a Lab-wide assumption

**Recommendation Candidate**:
An entity or action eligible to appear in the output of a Recommendation Problem. A
catalog item is one problem-specific candidate type rather than a Lab-wide assumption.
_Avoid_: Item as a Lab-wide assumption

**Candidate Text Content**:
The logical `candidate_text` field containing Dataset-owned textual evidence that a
Problem–Dataset Binding permits content-based Approaches to consume. Source-specific Dataset
Adaptation may materialize it deterministically from declared source fields such as title,
description, or categories; adding an unrelated metadata field does not implicitly enter its
content. It is not a separate domain object. Missing required `candidate_text` makes a
content-based Experiment incompatible, while an explicitly empty value remains empty content;
a Candidate identifier is never substituted as text.
_Avoid_: All Metadata Concatenation, Identifier Text Fallback, Content Specification Object

**Candidate Catalog**:
The identified universe of Recommendation Candidates for a Problem–Dataset Binding. The Lab
assumes every Catalog member is available for every serving invocation and evaluation case;
it does not model inventory, active windows, request-specific eligibility, or caller-supplied
subsets. An Approach may retrieve or rank only an internal subset, but that is Approach
behaviour and does not change Catalog membership.
For the current Dataset contract, membership comes from the authoritative Candidate data
table rather than observed Interaction Events; Candidates with no events remain members, and
an event referencing a missing member is invalid Dataset content.
_Avoid_: Eligible Candidate Set, Inventory, Request-specific Candidate Universe

**Candidate Generator**:
An Approach Component role for Personalized Top-N Ranking that reduces the Candidate Catalog
to a size-controlled internal subset without promising the final order. An Approach may omit
this role when it can rank the full Catalog directly. Its generated subset or top-M pool does
not redefine availability, and missing a relevant Candidate remains attributable to the
Approach.
_Avoid_: Ranker, Retriever as a universal role name

**Ranker**:
An Approach Component role for Personalized Top-N Ranking that orders candidates according
to the Approach's objective. It may use an internal scoring signal without exposing that
signal as a Recommendation Score.
_Avoid_: Candidate Generator, Model as a universal role name

**Recommendation**:
A Recommendation Candidate selected for a Recommendation Subject by a Recommendation
Approach under the target Problem's contract. The Problem defines the output fields and
their semantics; operational provenance belongs to a separate Inference Receipt.
_Avoid_: Recommendation Candidate, Prediction

**Recommendation Score**:
An optional value produced for a Recommendation whose meaning and scale must be declared
by its Problem and Approach. It is not universally a probability, calibrated relevance,
confidence, or a value comparable across Approaches.
The initial Personalized Top-N Ranking output does not expose one; Approach scores and
component contributions remain internal or explicitly captured Diagnostic Observations.
This rule applies to every Approach, including graph, Composite, and Reranked Approaches.
Internal score use for composition does not add scores to the Problem-defined output or its
Evaluation Outcome Set; retaining scores requires predeclared diagnostic capture.
_Avoid_: Probability, Confidence unless explicitly defined

**Evaluation Protocol**:
The Problem-Specification-compatible rules by which an Experiment constructs evaluation
inputs and interprets held-out observations as Evaluation Judgments and outputs as
comparable evidence. It is owned by the Experiment and contains evaluation-only choices
such as data partitioning, full-Catalog or sampled-candidate measurement, required Approach
capabilities, query forms, cutoffs, and metrics. A history query uses all eligible evidence
before its cutoff by default; a deliberate query-history window changes Protocol and Slice
identity, applies equally to every Variant, and is distinct from an Approach-owned
model-context window. An external judge or other evaluation-only dependency is an External
Dependency Specification owned by the Protocol; changing its output-affecting semantics
changes Protocol identity, and each Run records its resolved Snapshot. A sampled-candidate
Protocol freezes its sampler semantics, sample size, randomness inputs, and exact
materialized per-case samples, which are shared across Variants; sampling is evaluation
methodology and does not make unsampled Catalog members unavailable. It materializes
immutable Evaluation Partitions rather than changing Dataset identity. Sampled non-targets
are Sampled Distractor Candidates, not negative preference labels unless the Problem and
Protocol explicitly construct a negative Evaluation Judgment. It binds each metric's
predeclared
Inference Failure treatment, always exposes attempted, successful, failed, and unattempted
case counts, and may declare resource-protection conditions that terminate a Run early.
A Protocol making a point-in-time or production-applicability claim admits observations and
features according to Availability Time. If the Dataset lacks it, the Protocol must declare
the assumption that Availability Time equals Occurrence Time and scope the claim to that
assumption; non-temporal benchmarks need neither field. Within each observation family the
Protocol consumes, Availability Time must either be complete or be covered entirely by that
equality assumption. Partially populated availability cannot trigger a record-by-record
fallback and is incompatible with a point-in-time claim until Dataset Adaptation resolves its
semantics. A point-in-time claim also requires every observation or feature influencing the
queried Artifact, including evidence from other Subjects, to have been available before the
query time. Per-Subject chronological partitioning alone does not establish this condition;
assuming Availability Time equals Occurrence Time does not repair cross-Subject future-data
access. For an ordered behavioral-history query, Availability Time first determines which
observations are accessible at the cutoff, then Occurrence Time and any valid source sequence
order only those admitted observations. Ordering by arrival instead is a distinct
Problem-defined query form, not an evaluator fallback. Events tied at the available temporal
resolution and lacking a meaningful source sequence form one simultaneous group: a temporal
Protocol may not split the group across an access boundary or claim an order within it. When
a per-case terminal holdout boundary falls inside such a group, the whole group stays on the
held-out side. If the Problem cannot form a valid multi-target Judgment or the remaining
history violates a declared minimum, the Protocol excludes that evaluation case through a
predeclared Evaluation Cohort rule and reports the reason and count rather than choosing an
arbitrary event.
_Avoid_: Serving Contract, Run

**Terminal Chronological Holdout Protocol**:
The initial per-Subject chronological benchmark Evaluation Protocol that assigns each Subject's earliest
approximately 70% of events to Fit, the next 10% to Selection, and the latest 20% to
Assessment under complete domain-valid chronology. It never uses row order or identifiers to
break temporal ties, never splits a simultaneous event group, and never falls back to random
partitioning; cases left without required Fit evidence or Assessment Judgments are excluded
by declared Cohort rules, while an empty resulting Cohort fails preflight.
For a Subject with `n >= 3` events, its target counts are
`max(1, floor(0.10 * n + 0.5))` for Selection and
`max(1, floor(0.20 * n + 0.5))` for Assessment, with the remainder assigned to Fit. Assessment
takes the latest suffix and Selection the preceding suffix; a boundary crossing a simultaneous
group expands the held-out side earlier without shrinking the group. A resulting case without
Fit evidence or an Assessment Judgment is excluded with its declared reason. When `n < 3`, all
events remain in Fit and the Subject is excluded from held-out Cohorts as
`insufficient_partition_events`. The MovieLens 20M implicit-preference instance partitions
all rating events before applying `rating_at_least_4` within each role. A held-out role with
no qualifying event therefore causes the predeclared Cohort exclusion rather than pulling an
earlier positive rating across the temporal boundary; filtering positives before partitioning
defines a separate positive-event Protocol.
This Protocol preserves chronology within each Subject but does not establish system-wide
point-in-time correctness: a shared Artifact may learn from one Subject's Fit events that
occur after another Subject's held-out query time. Its evidence therefore supports this
per-Subject chronological benchmark, not by itself a temporal serving simulation or a
production-applicability claim based on point-in-time-correct quality. The 70/10/20 assignment
remains unchanged. A Study needing that claim must declare a separate Protocol ensuring that
all data influencing each queried Artifact was available before that query; the current
Protocol does not implicitly acquire a global cutoff or per-query refitting workflow.
_Avoid_: Temporal Split with Row-order Ties, Random Holdout, Automatic Split Fallback

**Random Per-Subject Holdout Protocol**:
An explicitly selected non-temporal benchmark Protocol that assigns each Subject's events to
Fit, Selection, and Assessment using a declared deterministic partition seed. It carries no
temporal or production-applicability claim and its results are not directly comparable with a
Terminal Chronological Holdout Protocol.
It uses the same round-half-up target-count formula and `n < 3` exclusion as the Terminal
Chronological Holdout Protocol, but assigns distinct events from a deterministic per-Subject
permutation; chronology and simultaneous groups have no role.
_Avoid_: Default Holdout, Temporal Evaluation, Random Fallback

**Evaluation Partition**:
An immutable, independently identified assignment of a Dataset's observations or cases to
the Fit, Selection, or Assessment roles declared by an Evaluation Protocol. Its identity
captures the source Dataset, partition rules, randomness inputs, and concrete assignments.
Every Approach Variant in a Direct Comparison uses the same Partition assignments;
protocols such as cross-validation represent their folds explicitly. An Approach may
perform Signal Preparation within the data made available to it but may not repartition
observations or cross the Protocol's data-access boundaries.
_Avoid_: Dataset, Prepared Dataset, Signal Preparation

**Fit Partition**:
The Evaluation Partition role whose observations may directly influence learned parameters
or other fitted state of an Approach Artifact. Signal Preparation may transform or filter
the supplied observations but cannot expand this access boundary.
_Avoid_: Selection Partition, Assessment Partition

**Selection Partition**:
The Evaluation Partition role whose observations may influence the choice of an Approach
Variant, hyperparameters, or stopping point. Measurements obtained from it are development
evidence and cannot also serve as unbiased final evidence for the choice they influenced.
An Evaluation Protocol may omit this role when no data-driven selection occurs.
_Avoid_: Fit Partition, Assessment Partition, Final Test Set

**Assessment Partition**:
The Evaluation Partition role reserved for evaluating Approach Variants after the relevant
selection decisions are fixed by a declared Selection Procedure. Its observations cannot
influence fitting or selection. A claim about generalization requires evidence from this
role; a Protocol may omit it only when its outputs are explicitly treated as development
evidence rather than final evidence. Independence is relative to the Approach and its design
history, not guaranteed by a partition's `test` name: once observations from a Partition
influence a later Approach, Variant, or Selection Procedure, reusing it for that work may
still provide benchmark-comparison evidence but not independent assessment evidence. A final
generalization claim requires a Partition whose observations did not influence the fitting,
design, or selection decisions being assessed. Exposure includes any derived signal that
actually informs such a decision—for example aggregate or slice metrics, failed-case lists,
individual outputs, or even a pass/fail response—whether consumed by a human or an automated
agent; direct access to raw observations is not required. Prior knowledge of only the Problem
schema and Evaluation Protocol is not exposure to a subsequently materialized Partition.
_Avoid_: Fit Partition, Selection Partition, Validation Set

**Selection Procedure**:
The predeclared procedure by which an Experiment uses Selection Partition measurements to
choose one or more Approach Variants or stopping points for assessment. Its candidate set,
Evaluation Metric Specifications, aggregation, decision rule, and tie-break rules are part
of the immutable Experiment Specification. Selection and assessment may occur in one
Experiment when this Procedure is frozen before execution; changing it after observing
selection evidence creates a new linked Experiment. When quality and resource evidence are
both relevant, the Procedure may predeclare a lexicographic priority or retain the Pareto set.
The Lab defines neither a universal combined score nor a hard resource-budget selection rule;
resource evidence is comparable only when its measurement semantics and required Execution
Environment dimensions align, and it is not automatically reduced to monetary cost. An
Experiment that reports several objectives without a predeclared decision rule may retain
valid comparative evidence but cannot name a winning Variant. When the Experiment intends to
select a subset for assessment, absence of that rule makes its Specification invalid before
the first Run rather than permitting a post-hoc choice. Every Variant retained by the
Procedure, including every member of a retained Pareto set, may be measured on the Assessment
Partition, but those measurements cannot break a tie or narrow that set. Without a
pre-assessment rule that already determines a single winner, the Experiment's conclusion
retains multiple assessed candidates rather than selecting one from assessment evidence.
An Approach-owned stopping rule that observes only Fit Partition signals, such as training
loss, remains part of Variant training semantics and is not model selection. A rule that
observes a held-out metric or chooses among checkpoints must use the Selection Partition and
is also governed by this frozen Procedure. Neither form may observe the Assessment Partition.
The Run retains an immutable selection trace for every checkpoint actually considered,
including its stopping point, content digest, Measurement references, and the final decision.
The initial single-winner template maximizes case-macro Selection NDCG@10, then case-macro
Selection Recall@10, then the canonical Approach Variant Identity in ascending byte order.
The last rule supplies determinism rather than a quality claim. The candidate set and rule
freeze before execution, and Assessment cannot revise the winner. A comparison-only
Experiment need not name a winner, while an Experiment containing one fixed Variant omits the
Procedure; any reserved Selection observations nevertheless remain outside its Fit and
Assessment roles.
When the Experiment uses the initial five-Run stochastic repetition template, both metric
criteria are arithmetic means over all five valid planned Runs; an invalid or incomplete Run
cannot simply be omitted from either mean.
A semantics-preserving correction to a Selection metric requires replaying the frozen
Selection Procedure over the corrected Measurements derived from retained Evaluation Outcome
Sets. The corrected decision is retained as a superseding selection trace, preserving the
prior trace and its Measurement references, and dependent Experiment Results are superseded
accordingly. If the corrected winner changes, Assessment evidence for the former winner
remains evidence only for that former Variant and exact evaluated target; it cannot transfer
to the newly selected Variant or checkpoint. Missing required Artifact or checkpoint material
or Assessment evidence leaves the corrected final conclusion incomplete rather than keeping
the former winner merely because it already has Assessment evidence. This correction uses
the existing selection-trace and evidence-supersession lifecycles without creating another
domain object or changing the frozen selection semantics.
Replay is sufficient only when retained evidence covers every decision required by the
corrected Procedure. If a metric implementation defect caused early stopping and the corrected
rule would require further training or checkpoint evaluations that never occurred, replaying
only the observed checkpoints cannot establish the corrected outcome. The affected conclusion
remains incomplete and recovery requires a new Run, not continuation that rewrites the old
execution history. That Run may replace the invalid attempt only after a Run Failure
Adjudication and only when the frozen technical replacement policy permits it; otherwise the
work requires a new linked Experiment with Assessment exposure classified under the existing
rules. Fixed-budget training without held-out-metric stopping is not subject to this missing
trajectory merely because a metric is recomputed.
_Avoid_: Ad-hoc Tuning, Assessment Result, Approach Specification

**Evaluation Judgment**:
A problem-compatible relevance, utility, label, or target constructed by an Evaluation
Protocol from held-out observations for scoring an Approach's output. Its semantics are
shared by all Approach Variants in a Direct Comparison and remain independent of how each
Approach performs Signal Preparation. The same observations may support binary, graded, or
other Judgments under different explicit Protocols; a Dataset does not make that choice. A
held-out interaction may make a Candidate relevant even when that Candidate also appears in
the query evidence; novelty-only relevance requires a different Problem Specification rather
than an implicit evaluation filter. The current implicit-feedback Protocol uses binary
Candidate relevance: one or more qualifying held-out events for the same Candidate count
once, while frequency- or value-graded relevance requires another explicit Protocol. Its
qualifying predicate is applied to individual events before that binary collapse and is
`value > 0`; changing the predicate or threshold changes Evaluation Protocol identity. The
Protocol materializes Judgments from event-level partition observations independently of any
Approach-derived matrix or Preference Signal representation. The initial MovieLens 20M
implicit-preference Protocol is a distinct Protocol whose predicate is `value >= 4.0`; lower
ratings are neither relevant targets nor negative Judgments. This matches the study's
`rating_at_least_4` Signal Preparation without making an Approach-derived representation the
source of evaluation truth.
_Avoid_: Preference Signal, Raw Interaction, Approach-specific Ground Truth

**Sampled Distractor Candidate**:
A non-target Candidate materialized by a sampled-candidate Evaluation Protocol so the target
can be ranked against a bounded comparison set. In an implicit-feedback Problem it is not a
negative preference merely because no positive event was observed. The Protocol excludes the
case's relevant targets, declares how other interactions are treated, and shares the exact
materialized distractors across all compared Variants. A point-in-time Protocol constructs
the pool using only information available at the query time and does not exclude a Candidate
merely because a later interaction will occur. A Reported-result Reproduction may follow a
source's all-time exclusion rule, but that Slice is benchmark evidence without a
point-in-time-correctness claim.
_Avoid_: Negative Feedback, Training Negative Sample, Unavailable Candidate

**Evaluation Metric Specification**:
The complete semantic identity of an evaluation measure, including its metric and version,
accepted Evaluation Judgment semantics, parameters such as cutoff, aggregation or weighting
rules, and treatment of undefined cases and Inference Failures, including whether a failure
receives a meaningful worst value, is excluded with explicit coverage, or invalidates the
Slice. A short label such as `ndcg@10` is display metadata rather than sufficient identity.
Specifications are introduced as concrete metrics require them; the Lab does not need a
speculative universal metric taxonomy.
The initial full-Catalog binary implicit-feedback Slice binds `K = 10` and reports equally
weighted case-macro Recall@10 and NDCG@10. Per-case recall divides Top-10 binary hits by all
relevant Candidates; NDCG uses binary gain, discount `1 / log2(rank + 1)`, and an ideal list
of length `min(10, relevant_count)`. Any Inference Failure invalidates the Slice's quality
result rather than receiving zero or exclusion, an empty Cohort fails preflight, and attempted,
successful, failed, and preflight-excluded case counts are always reported. Precision, MAP,
MRR, coverage, and interval estimates remain absent until a concrete Study requires them.
_Avoid_: Metric Name, Unqualified Score

**Evaluation Cohort**:
The problem-compatible set of Recommendation Subjects or evaluation cases whose outcomes
contribute to reported metrics. Its inclusion and exclusion rules, counts, and reasons are
declared by the owning Evaluation Protocol rather than applied silently. Membership is
materialized from the Problem, Dataset, Partitions, and Protocol before Approach inference;
every Variant in a Direct Comparison receives the same Cohort. A Variant whose declared
capability cannot cover it makes the Experiment Specification invalid before execution,
while a runtime failure after declaring coverage remains an Inference Failure rather than a
Variant-specific exclusion. A post-inference grouping such as applied Algorithmic Backoff is
a Variant-local outcome breakdown, not another Evaluation Cohort. For the current binary
implicit Protocol, a case without any qualifying Judgment is excluded before inference under
the predeclared reason `no_qualifying_judgment`, with its count reported rather than assigned
a metric value or Inference Failure. If these exclusions leave the entire Cohort empty, the
Experiment fails preflight with its exclusion summary retained; no Run, Measurement, or
Evaluation Integrity Failure is created because execution never began and materialization
remained faithful to the Protocol. A case that has a qualifying Judgment but cannot supply
the Evaluation Slice's required query evidence is likewise excluded before inference under
`missing_required_query_evidence`; it may participate only in another Slice whose explicit
query form, such as cold-start, accepts its evidence. For an ordered-history Slice, a case
whose required query evidence contains a tied event group with no meaningful source ordering
is excluded before inference under `ambiguous_required_query_order`; identifiers, file row
positions, and implementation tie-breakers cannot resolve it. The same case may participate
in an unordered Slice whose query form accepts the evidence.
_Avoid_: Dataset, Training Population

**Evaluation Slice**:
A named portion of an Evaluation Protocol that fixes one Problem-defined query form and its
evaluation conditions, including an Evaluation Cohort, exact Candidate Catalog, and whether
measurement uses that full Catalog or one exact materialized candidate-sampling scheme. Every
Approach Variant measured in the Slice must declare the required capability; the evaluator
neither falls back to another query form nor silently intersects Variant capabilities or
Approach-specific candidate coverage. Results from Slices with different query forms,
Catalogs, or candidate-sampling schemes do not support a Direct Comparison.
The initial full-Catalog implicit-feedback Slice fixes `known_subject`. A Variant lacking that
capability is incompatible before execution; the evaluator never substitutes
`supplied_history`. Fold-in evaluation uses a separate Slice fixed to `supplied_history`, and
its measurements are not treated as the same comparison condition.
_Avoid_: Evaluation Cohort, Runtime Capability Fallback

**Serving Contract**:
The versioned, transport-neutral operational input and output boundary for invoking a
Recommendation Approach under one exact Problem Specification. It is independent of any
particular Approach and declares the Approach capabilities required for interchangeable
Variants to conform. It uses the complete Candidate Catalog fixed by the Problem–Dataset
Binding, excludes evaluation-only and wire-protocol policy, and uses the same Approach
inference semantics as offline evaluation.
Semantic input or output changes create another Contract version, while transport changes
use another Serving Adapter without changing the Contract. The Contract explicitly permits
or rejects named Degradation Modes, requires the actual mode to be observable through an
Inference Receipt, and declares which Receipt fields a caller may receive.
The initial Personalized Top-N Ranking Contract accepts exactly one of `known_subject` with
`subject_id`, or `supplied_history` with a non-empty ordered `events` array, plus exact `N`.
Each supplied event contains a Catalog `candidate_id`, a finite `value` defaulted and
materialized as `1.0`, and optional `occurred_at`; timestamps must be present for every event
or none, and when present validate nondecreasing declared array order without reordering it.
Equal times retain array order, and a timezone-less value is interpreted as UTC. Repeated
events remain distinct. Unknown identifiers fail without query-form or Algorithmic Backoff,
extra fields fail closed, and the initial Contract has no request context, Candidate subset,
seen-Candidate exclusion, history-count limit, or fixed `N` ceiling below Catalog size.
A separate bounded-count Contract permits only `N <= min(100, catalog_size)` for serving
the initial Pointwise LLM Reranking Variant. This bound is explicit Contract semantics rather
than an Artifact-dependent transport guard; exact count, full Catalog membership, and ordered
Candidate-ID-only output remain unchanged. Qualification binds this exact bounded Contract
and cannot be inherited from the initial unrestricted-count Contract. The bounded Contract
is another instance of the existing Serving Contract concept, with its existing versioning
and Qualification lifecycle, rather than a new domain object type.
Every Serving Contract fixes both its permitted query-mode set and its permitted exact-`N`
domain before an Artifact is selected. Conformance requires the Artifact to support every
declared mode throughout that domain; supporting only part of the Contract is insufficient
for Serving Qualification. Replacing an Artifact never narrows either dimension implicitly.
A separately identified `known_subject`-only Contract supports the initial LightGCN, BPR
Matrix Factorization, and ID-only Neural Two-Tower Variants, which do not conform to the
initial Contract accepting both query modes. A bounded-count Contract for a Reranked Variant
likewise fixes a query-mode set that the declared composition can cover; its modes are not
discovered or reduced when an Artifact is loaded. Contracts are introduced for concrete
serving use cases rather than generating every combination of modes and count bounds.
These remain instances of the existing Serving Contract lifecycle, and each Qualification
binds the exact Contract being satisfied.
_Avoid_: Evaluation Protocol, HTTP API Schema, Evaluation Request

**Serving Adapter**:
A replaceable boundary that maps a concrete transport or framework, such as HTTP, gRPC,
batch execution, or BentoML, onto a Serving Contract. It owns wire serialization,
authentication integration, transport status, request correlation, and transport resource
guards rather than recommendation semantics. Evaluation invokes the Approach Inference
Interface directly instead of using an Adapter. Operational Serving Qualifications identify
the Adapter implementation because it can affect latency, reliability, and security
evidence. It may reject a request under an explicitly declared transport guard but may not
truncate its history or cause evaluation to inherit that guard; the current scope has no
history-count limit. It may not silently substitute recommendation behaviour. Retry, timeout, and
circuit-breaker policy are outside the current stable-dependency scope and deferred until a
concrete reliability use case exists. It may co-serialize permitted Inference
Receipt fields without making them part of the Problem-defined output. It may translate
authorized transport- or system-facing Subject and Candidate identifiers to and from the
canonical identifiers used by the Serving Contract through an external mapping dependency.
The mapping contents remain outside the Artifact, repository, and Qualification evidence;
the Serving Qualification Specification owns its External Dependency Specification and
missing-mapping policy, and the Trial records the resolved Snapshot. Translation may not
silently drop, substitute, or reorder
Problem-defined outputs. Under the basic policy, any missing Candidate mapping fails the
whole serving request explicitly. Partial lists and mapping fallbacks are unsupported until a
concrete Serving Contract defines their semantics and receives separate Qualification
evidence. A missing Subject mapping likewise fails a subject-identity query explicitly; the
Adapter cannot reinterpret it as anonymous, popularity-based, or history-only inference.
Cold-start is a distinct Problem-defined query form and Approach capability that the caller
must request explicitly through a compatible Serving Contract. Output-affecting transport
caches are outside the current Lab scope.
_Avoid_: Serving Contract, Approach Inference Interface, Recommendation Approach

**Degradation Mode**:
A named, output-affecting alternative inference behaviour declared by an Approach
Specification for resource pressure or another explicitly researched condition. External
dependency failure is not currently modeled. The
Approach Variant binds its implementation, the Serving Contract decides whether it is
permitted, and the Inference Receipt identifies the mode actually used. Each Mode requires
its own Evaluation and Serving Qualification evidence; if no qualified permitted Mode can
run, the request fails explicitly rather than silently changing semantics. A normal data
condition handled by an Algorithmic Backoff is part of primary inference semantics, not a
Degradation Mode.
_Avoid_: Algorithmic Backoff, Silent Fallback, Transport Retry, Best-effort Success

**Algorithmic Backoff**:
A predeclared primary inference rule used when a conforming query supplies valid evidence
but that evidence yields no usable signal or Candidates. The Approach Specification defines
its trigger and choices; the Approach Variant binds either explicit failure or a named
backoff, making the choice part of Variant identity and requiring separate evaluation of the
affected invocations, and it cannot reinterpret missing required evidence as cold-start or
replace an unsupported query form.
_Avoid_: Silent Fallback, Degradation Mode, Cold-start Substitution

**Approach Inference Interface**:
The Problem-Specification-specific external seam through which evaluation and serving
invoke an executable Recommendation Approach. It accepts the Specification's query and
evidence, plus an output limit when the contract requires one, and returns the
Specification-defined Recommendations against the complete Candidate Catalog bound to the
Artifact; callers cannot supply a request-specific subset. Candidate generation, ranking,
reranking, and internal scores remain implementation details unless a separate diagnostic
contract deliberately exposes them. The returned order is final: evaluation, serving, and
their adapters may neither resolve ties nor otherwise reorder it. For current Personalized
Top-N Ranking, the limit requests an exact count; an impossible count is rejected before
inference and a conforming invocation cannot return a partial list. Its invocation is
associated with an Inference Receipt that remains outside the Problem-defined output.
_Avoid_: Universal Score Vector, Serving Contract, Evaluation Protocol

**Inference Receipt**:
A minimal provenance sidecar associated with an inference invocation, separate from its
Problem-defined Recommendation output. It identifies the request or evaluation-case
correlation, exact Approach Artifact and contract Specifications, actual primary or
Degradation Mode, stable key of any applied Algorithmic Backoff, final outcome, and resolved
inference-phase External Dependency Snapshot references, plus
whether exact Run-local memoization supplied the outcome and the memoized result's
provenance. The memoization key includes the Run identity as well as every semantic input,
so an outcome produced by one Run cannot complete another Run even when their other hashes
match. The enclosing Run or Serving Trial binds the exact inference-runtime Code Snapshot
or immutable external runtime/container version; an individual Receipt may reference that
execution-level binding instead of duplicating it. A Serving Adapter decides which safe
fields to expose on the wire, while evaluation may store common bindings once at Run level
instead of duplicating them per case. Diagnostics and detailed telemetry
are excluded until a concrete use case requires an extension.
_Avoid_: Recommendation Output, Diagnostic Observation, Telemetry Payload

**Diagnostic Observation**:
An opt-in intermediate output deliberately captured for reproducible investigation of an
Approach, such as a generated candidate set or component contribution. When an Experiment
Result or Research Study relies on it, the observation has a named, versioned schema and
Run provenance; it is not exposed through the Serving Contract by default. The Lab defines
only a minimal shared envelope and adds Problem- or Approach-specific payload schemas when
a concrete use case justifies their maintenance. An Approach Specification owns its
diagnostic schemas by default, an Evaluation Protocol selects which observations to
capture, and the owning Recommendation Problem defines a shared schema only when a real
cross-Approach comparison requires one. A Run records observations but does not define
their meaning. Raw External Dependency requests or responses are captured only when the
owning Specification and Evaluation Protocol predeclare them as Diagnostic Observations,
including schema, request correlation, redaction, and retention scope. Replaying a retained
response supports investigation but is not fresh inference or Repeatability Evidence. Ad-hoc
logs and traces may aid debugging but cannot support reproducible claims.
_Avoid_: Debug Trace, Recommendation, Run Measurement

**Approach Component**:
A replaceable unit that fulfils a problem-defined role within a Recommendation Approach.
An Approach may use one monolithic component or several composed components, while
component categories and ordering remain local to the target Recommendation Problem.
_Avoid_: Model as a name for every component, Universal Pipeline Stage, Plugin as a domain term

**Model**:
An Approach Component whose behaviour includes parameters learned from data. A Model is
not a synonym for a Recommendation Approach or its executable artifact.
_Avoid_: Recommendation Approach, Approach Artifact

**Approach Artifact**:
An executable materialization produced by applying an Approach Variant to one Dataset under
a Problem–Dataset Binding during a Run or Refit Build, and immutable after it is committed.
It contains no state whose purpose is to resume training and does not imply that the
Approach contains a Model. An Artifact used for assessment remains distinct from any later
Artifact produced by a Refit Procedure. Executable does not necessarily mean self-contained:
the Artifact declares every inference-phase External Dependency Specification it requires,
or explicitly has none. Fit- or export-only dependencies do not become serving requirements.
An external resource becomes Artifact payload rather than an inference dependency only when
all immutable bytes required from that resource are contained in the Artifact and covered by
Artifact Identity. Retaining only a URI, model name, or machine-local cache entry does not
make the Artifact self-contained; the original source revision remains provenance after
packaging.
Runtime and package bytes bundled by an export strategy into an immutable container or image
are likewise Artifact payload when that image digest participates in Artifact Identity.
Every Artifact declares an artifact-format version and inference-runtime contract version
that state how its payload is interpreted and which public inference behaviour a runtime
must implement. The artifact-format major applies to the shared manifest and container,
whereas the runtime-contract major is interpreted only within its stable logical runtime;
compatibility is therefore keyed by `(artifact format, logical runtime, runtime contract)`.
A logical runtime name is a persisted canonical machine identity, not a class, module, plugin
display label, or mutable alias. Refactoring implementation names preserves it; loaders never
silently rewrite it, and choosing a genuinely new logical runtime changes Artifact Identity.
It identifies a representation and execution family rather than a Model or Recommendation
Approach: an incompatible representation change that preserves that family increments its
contract major, while a separately selectable execution family receives a new name. A change
to Problem-observable recommendation semantics remains an Approach Specification or Variant
change regardless of runtime naming.
Loading an Artifact is read-only and never changes its compatibility key or payload. Any
explicit representation conversion produces a distinct Artifact Identity and lineage rather
than upgrading, overwriting, or reinterpreting the source Artifact in place.
Normal inference resolves the compatibility key from the Artifact and cannot override its
logical runtime or contract version through CLI or serving configuration. A diagnostic may
invoke another loader directly, but its output is not Artifact inference, has no Inference
Receipt, and cannot support evaluation or qualification evidence.
A bundled Artifact includes the required Approach-specific inference code, compiled bytes, or
container in its identity-bearing payload. A thin Artifact instead names a stable logical
runtime or loader contract without embedding or identifying one source-code revision. A
compatible runtime implementation may execute the same thin Artifact after it declares
support for that compatibility key and passes structural, integrity, and behavioural
compatibility gates; matching the manifest schema alone is insufficient. The exact runtime
implementation used remains a clean Code Snapshot or immutable external runtime/container
version on the Run or Serving Trial, so evidence obtained with one implementation does not
silently transfer to another. Code used only for training or export remains Artifact
Provenance rather than Artifact Identity.
Within one Code Snapshot, each compatibility key currently resolves to exactly one runtime
implementation; automatic backend selection and hardware-dependent fallback are unsupported.
Multiple simultaneous implementations and an execution-owned selector are deferred until a
concrete CPU/GPU or equivalent comparison requires them, and such a selector would remain
outside Artifact Identity.
Its shared manifest and each logical runtime's inference-affecting metadata use closed,
versioned schemas; unknown semantic fields fail rather than being ignored. Every semantic
manifest value is covered by Artifact Identity or deterministically validated from
identity-covered content, while descriptive metadata and creation lineage belong to Artifact
Provenance outside the manifest. A generic extension namespace is absent until a concrete
integration justifies one.
It may contain only canonical identifiers and metadata permitted by Dataset Adaptation
and the Problem–Dataset Binding; it never embeds a sensitive source identifier or a
re-identification mapping. A fitted Artifact is eligible for Assessment only after its
exported inference outputs pass the Variant's predeclared parity rule against the selected
Training Checkpoint; Assessment invokes the Artifact, never the Checkpoint. The current Lab
supports only behaviour-preserving export for selected checkpoints. An intentionally
output-changing strategy such as ranking-altering quantization requires a distinct Approach
Variant whose selection and evaluation measure the exported representation itself; that
workflow is deferred until a concrete need exists.
_Avoid_: Model, Model Artifact, Recommendation Approach

**Artifact Identity**:
A content-derived identifier over an Approach Artifact's Dataset identity, Problem–Dataset
Binding, Approach Variant and executable inference semantics including inference-phase
External Dependency Specifications, artifact-format and inference-runtime contract versions,
logical runtime selection, immutable payloads, and any embedded computation graph. These
values form one closed, canonical identity document within the manifest, and Artifact
Identity is derived from that whole document rather than a separately maintained field
allowlist; the identifier itself remains outside the hashed document to avoid self-reference.
Small semantic documents are represented there by their content IDs rather than embedded
copies; the governing Run, Experiment, or Artifact Provenance retains their canonical values.
The manifest embeds only the strict fields required to validate and execute inference, while
large Dataset and payload content is referenced by digest. Full self-describing closure is
deferred until distribution or workspace-independent audit becomes a concrete requirement.
Missing or hash-mismatched semantic references do not corrupt an otherwise executable
Artifact, so local development inference may continue from its strict runtime fields. They do
make it ineligible for decision-bearing assessment or Serving Qualification: pre-execution
validation stops before creating a Run or Serving Trial, rather than recording an Inference
Failure.
For a bundled Artifact, implementation or container bytes participate directly as payload.
For a thin Artifact, the non-bundled runtime implementation's Code Snapshot does not
participate; several compatible runtime Snapshots may therefore execute the same Artifact
identity, while each Run or Serving Trial retains evidence for its exact Artifact–runtime pair. A
semantics-preserving runtime fix need not create another Artifact, but an intentional change
to Problem-observable inference behaviour requires a new Approach Specification or Variant
and Artifact, and any payload or embedded-graph change necessarily creates a new identity.
Creation lineage is excluded and identity is independent of Run identity, so multiple Runs
may produce or reference the same Artifact.
_Avoid_: Run ID, Mutable Version, Human-readable Alias

**Artifact Provenance**:
Lineage relating an Approach Artifact to the Runs or Refit Builds and non-inference
conditions that produced it. It is stored outside Artifact Identity and may associate
several executions with the same Artifact or link a refit Artifact to the Experiment Result
that selected its Variant. Canonical semantic documents referenced by content ID are small,
immutable, and retained indefinitely once materialized; the Lab currently has no automatic
garbage collection, reference counting, or deletion workflow for them. This retention rule
does not extend to Dataset content, Artifact payloads, Training Checkpoints, source bundles,
or containers. Descriptive labels, creation time, source Code Snapshot, build
details, and notes belong here rather than in the identity-bearing manifest. It records the
realized stopping point or checkpoint selected by a Variant's frozen training and Selection
Procedures without turning that outcome into a new Variant. It is stored as a separate sidecar
or evidence record rather than as unhashed fields inside the Artifact manifest. It also
retains export-parity evidence, including the checkpoint
digest, Artifact identity, cases or fixture identity, comparison rule, any Problem-permitted
numeric tolerance, Code Snapshot, and outcome. Verification cases are predeclared contract
fixtures or come from Fit or Selection Partitions and are bound by the Experiment or Run.
For a declared internal full-Catalog scoring capability, parity evidence also identifies its
exact score-vector comparison and canonical Candidate mapping on those verification inputs;
public output parity alone does not establish the internal contract's preservation.
Assessment inputs remain unopened until Artifact verification succeeds. Resolved External
Dependency Snapshots used during Fit, export, or verification are provenance rather than
mutable fields on the Artifact.
_Avoid_: Artifact Identity, Artifact Content

**External Dependency Specification**:
An immutable value object embedded by exactly one domain owner for an external system or
resource whose output can affect Dataset materialization, Fit or export, Artifact inference,
evaluation, or a Serving Adapter. It is not a global registry entry. It declares a stable
canonical key, exactly one usage phase, provider or protocol, required capability or
operation, exact stable semantic model or resource revision, output-affecting configuration, and any
stochastic behaviour and seed guarantee. Accepting a seed parameter without a reproducibility
guarantee is not treated as seed-controllable. When the same provider or model revision is
used in multiple phases, each phase has a separately owned declaration and canonical key;
multi-phase Specifications and implicit cross-phase sharing are unsupported.

A resource is external when the owning immutable Dataset or Artifact does not contain all
bytes required from it for the relevant phase. Packaging those exact bytes moves them into
payload identity and leaves their original source revision as provenance. A URI, model name,
or populated local cache is only a locator or execution convenience and does not change this
classification. A language package, framework, system library, driver, or hardware resource
is not an External Dependency merely because it is installed from elsewhere; unbundled
runtime requirements belong to the Execution Environment.

Ownership determines identity and lifecycle. Dataset Adaptation owns preparation-phase
Specifications and retains Snapshots as Dataset provenance. An Approach Specification
declares dependency slots and an Approach Variant binds the Fit, export, and inference
Specifications; Fit or export changes create another Variant, while inference Specifications
also enter Artifact Identity. An Evaluation Protocol owns evaluation-only Specifications.
A Serving Qualification Specification owns Adapter-only Specifications. Concrete endpoints,
credentials, and secrets are execution inputs rather than semantic identity. The current Lab
assumes every declared dependency remains reachable and behaviourally stable during an
execution; outage, drift, retry, timeout, circuit breaking, and dependency-failure
degradation are deferred. An Artifact with no inference-phase Specifications is explicitly
self-contained with respect to external inference dependencies.
_Avoid_: Package Dependency, Endpoint Configuration, Global Dependency Registry, Unversioned Latest Alias

**External Dependency Snapshot**:
The execution-owned record of the external dependency actually resolved for one declared
Specification and usage phase. Exactly one logical Snapshot exists for each combination of
owning execution identity, Specification identity, and usage phase; every invocation in that
scope references it rather than creating an invocation-level Snapshot. It records the
available immutable revision or digest, service/runtime version, requested seed or sampling
settings when applicable, and relevant observed configuration. It is retained by Dataset
provenance, a Run, a Refit Build, or a Serving Trial according to the owner and phase, and
validates conformance without changing the owner's identity. Resolving a different revision
or configuration within the same scope violates the stable-dependency assumption and rejects
the relevant phase rather than creating a second Snapshot and mixing evidence. Failure to
resolve the declared stable revision or configuration is a configuration or integrity
mismatch, not an availability failure. Raw requests and responses are not Snapshot fields.
External call counts, tokens, latency, and other consumption are also excluded: they are
execution Measurements or raw resource evidence, grouped by dependency canonical key and
usage phase. Totals are retained by default; per-invocation usage is retained only when the
governing execution plan predeclares that requirement.
_Avoid_: External Dependency Specification, Artifact Payload, Secret Capture

**Refit Procedure**:
A declared post-assessment procedure that applies a selected Approach Variant to an
explicitly broader fitting-data scope and produces a new Approach Artifact. It runs only
after the source Experiment is complete, so former Assessment observations may be admitted
without changing the original Experiment or Artifact. Its lineage references the Selection
Procedure and Experiment Result, but that prior assessment evidence does not become evidence
about the exact refit Artifact; the Procedure declares the parity and validation checks
needed before further use.
_Avoid_: Selection Procedure, Continued Training of an Assessment Artifact, Experiment Rewrite

**Refit Build**:
One concrete execution of a Refit Procedure that produces or reproduces an Approach
Artifact and records its own Code Snapshot, data scope, Random Seeds, Execution Environment
Snapshot, required raw resource evidence, and lineage to the source Experiment Result. It is
not a Run, does not belong to
the source Experiment, and cannot contribute Run Measurements to that Experiment. Run and
Refit Build may share internal execution infrastructure without sharing domain meaning.
_Avoid_: Run, Experiment Stage, In-place Artifact Update

**Serving Candidate**:
An explicit, scoped role held by an Approach Artifact only within a Serving Qualification
whose checks it passed. It is not a separate Artifact type, global boolean, or change to
Artifact identity, and does not mean the Artifact is deployed or universally
production-ready.
_Avoid_: Approach Artifact Type, Deployed Artifact, Production-ready Model

**Serving Qualification Specification**:
An immutable pre-execution plan binding one exact Approach Artifact, the clean Code Snapshot
or immutable external runtime/container version expected to execute it, Serving Contract,
Serving Adapter implementation, Serving Workload, Execution Environment Specification, the
Artifact's inference-phase and Adapter-owned External Dependency Specifications, Trial
eligibility and minimum evidence, estimators,
aggregation and uncertainty rules, permitted Trial violations, acceptance criteria, and the
raw resource dimensions required to support efficiency claims. It freezes before its first
Serving Trial; changing any semantic field creates a new Specification.
_Avoid_: Serving Trial, Serving Qualification, Mutable Load-test Configuration

**Serving Trial**:
One immutable concrete execution belonging to exactly one Serving Qualification
Specification, binding the actual Execution Environment and External Dependency Snapshots,
Code Snapshots for the inference runtime, Adapter, load generator, and
measurement-collection logic, raw measurements, and completion and integrity status. It
validates the Artifact's declared format and runtime contract against that exact runtime
implementation. It preserves one observation of serving behaviour and cannot alone declare
an Artifact qualified; incomplete or invalid Trials remain in the
evidence history, while dirty-code Trials are development evidence only. Decision-bearing
raw measurements retain non-sensitive request correlation and Workload class, scheduled,
admission, and completion timing, outcome, mode, relevant
request/response shape, and the Specification-required resource time series and cumulative
usage. Relevant resource evidence may include wall time, CPU or accelerator use, peak memory,
storage and network use, and external calls or tokens; dependency-specific usage is grouped
by canonical key and usage phase. It is interpreted only with the Trial's
Execution Environment Snapshot. Payloads, subject identifiers, and recommendation content are
excluded by default, and percentile summaries remain derived.
_Avoid_: Experiment Run, Serving Qualification, Aggregate Benchmark Result

**Serving Qualification**:
An immutable conclusion that applies one Serving Qualification Specification's predeclared
rules to all eligible Serving Trials, retaining their variability, incomplete evidence, and
result against every acceptance criterion as pass, fail, or inconclusive, together with the
Code Snapshot for aggregation and decision logic. Every eligible Trial is included; by
default each Trial's conservative estimate must meet a threshold unless
the Specification predeclares an allowed violation rule, and insufficient valid evidence is
inconclusive rather than pass. Its conclusion applies to the exact Artifact, inference
runtime implementation, Adapter, Workload, Environment, and dependency conditions bound by
the Specification and Trials. Reusing the same Artifact with another compatible runtime
implementation requires new Trials and a new Qualification rather than inheriting the old
one. One Artifact may have several Qualifications and may be a Serving Candidate for one
target but not another; Qualification remains outside Artifact identity and never rewrites
prior evidence. Artifact integrity is a prerequisite
but is not itself a Qualification, and Run-local memoized outcomes are not fresh latency or
repeatability evidence. Cold-start and steady-state evidence remain separate unless the
Workload declares a mix; End-to-End Serving Latency is required, and narrower measurements
cannot replace it.
Capacity evidence reports offered rate, achieved throughput, latency distribution, failures,
queueing, and resource saturation together rather than in isolation.
Missing a required resource measurement makes the corresponding resource or derived monetary-
cost criterion inconclusive, but does not invalidate independent quality or correctness
evidence. Monetary cost remains a time-sensitive derivation from raw usage and explicit
pricing assumptions rather than a property of the Artifact or Trial.
A semantics-preserving aggregation bug fix creates a superseding Qualification from the same
Trials; corrupted collection evidence invalidates the affected Trial, while changed
estimator or acceptance semantics require a new Specification and fresh Trials.
_Avoid_: Global Production-ready Flag, Artifact Verification, Deployment Approval

**Training Checkpoint**:
Run-local state captured so interrupted Model training can continue, potentially including
framework-specific parameters, optimizer state, and random state. It may also be a candidate
stopping point considered by a Selection Procedure. It is not an Approach Artifact and is
never a source of truth for serving. The selected Checkpoint is exported and verified as an
Approach Artifact; non-selected Checkpoint bytes may be deleted after the Run unless a Study
declares a retention need, while their digests and immutable selection trace remain.
Parity failure rejects the exported Artifact and prevents Assessment; it never makes the
Checkpoint an alternative assessment or serving boundary.
_Avoid_: Approach Artifact, Deployable Model

**Personalized Top-N Ranking**:
A Recommendation Problem that ranks catalog items for a known user or a supplied
history of Interaction Events and evaluates the ranked results against declared Evaluation
Judgments. In its current Specification, `N` is exact and cannot exceed the bound Candidate
Catalog; an impossible `N` is invalid before inference, while a shorter output for a valid
`N` is an Inference Failure.
Its initial output is exactly an ordered list of `N` canonical Candidate identifiers. Position
is rank; Recommendation Score, Candidate metadata, explanation, and explicit rank fields are
absent. A Serving Adapter may co-serialize permitted request correlation and Artifact
provenance from the Inference Receipt, but those fields remain outside the Recommendation
output, and external metadata enrichment cannot change its order.
_Avoid_: Generic Recommendation, Universal Recommendation Contract

**Interaction Event**:
An occurrence observed at event level that may provide evidence for a Recommendation
Problem. It does not inherently imply preference; repeated occurrences remain distinct
until source-specific Dataset Adaptation can prove they are duplicate records or the target
Problem gives them feedback semantics and an Approach interprets them. Identical field values
alone do not prove duplicate identity. An event may carry both when the occurrence happened
and when it became knowable to the system; those times are not interchangeable, and an
observed event cannot become available before it occurs.
_Avoid_: Preference Signal, Aggregated Interaction

**Occurrence Time**:
The time at which an Interaction Event or other observation happened in the source domain.
It supports event ordering but does not prove that the recommendation system could already
observe the event. Equal values do not imply an order; only a source sequence with declared
chronological semantics may break a tie.
_Avoid_: Availability Time, Ingestion Order

**Availability Time**:
The earliest time at which an Interaction Event, feature, or metadata value was available to
the system being simulated. Point-in-time evaluation uses this boundary to prevent future
information from entering inference or fitting, but it does not reorder admitted behavioral
history unless the Problem explicitly defines an arrival-ordered query form. It is optional
for non-temporal benchmarks; when absent for an entire consumed observation family, equality
with Occurrence Time is an explicit Protocol assumption rather than an implicit
interpretation of one `timestamp` field. Partial absence within that family is unknown
provenance, not permission for a per-record equality fallback.
_Avoid_: Occurrence Time, Evaluation Cutoff

**Preference Signal**:
Evidence consumed by a Recommendation Approach after problem-specific interpretation of
Interaction Events or other inputs. A signal may be filtered, weighted, transformed, or
aggregated without changing the source events.
_Avoid_: Interaction Event, Raw Event

**Implicit Feedback**:
Preference evidence inferred from observed behaviour rather than directly stated by the
subject. Its meaning is defined by the target Recommendation Problem, and absence of an
event is not inherently negative feedback.
_Avoid_: Positive Feedback, Explicit Feedback

**Explicit Feedback**:
A subject's directly expressed evaluation or preference under a scale or choice whose
meaning is defined by the target Recommendation Problem. It may express positive,
negative, or neutral preference.
_Avoid_: Implicit Feedback

**Experiment**:
A planned, reproducible evaluation of one or more Approach Variants under an owned
Evaluation Protocol for exactly one Problem Specification and exactly one Dataset joined by
one Problem–Dataset Binding, and binding the Evaluation Partitions needed by that Protocol.
It therefore belongs to exactly one Recommendation Problem and may contain multiple Runs to
measure repeatability or variability.
_Avoid_: Run, Job

**Experiment Specification**:
The semantic definition of an Experiment, including its Problem–Dataset Binding, Approach
Variant set, Baseline, Evaluation Protocol and Partitions, any Selection Procedure, and
planned Run dimensions, decision-bearing repetition count, and any sequential stopping or
completion rule. It also predeclares a technical replacement policy identifying which
Evaluation Integrity Failures allow a new full Run with the same planned dimensions to
supersede an invalid attempt. Before execution it verifies that every Variant's declared
capabilities cover every assigned Evaluation Slice and its materialized Evaluation Cohort.
The default stochastic Run dimension is one root seed. A controlled study may predeclare
named sub-seed overrides as additional planned dimensions, but ad-hoc overrides are rejected
after the Specification freezes.
It declares the raw resource dimensions required for any efficiency claim.
For every bound Assessment Partition it records an `independent`, `exposed`, or `unknown`
declaration relative to the complete Variant set and Selection Procedure, together with
concise provenance supporting that declaration. The Experiment author owns this declaration
and freezes it with the Specification before the first Run. `exposed` and `unknown` Partitions
may produce benchmark-comparison evidence but not final generalization evidence; this
conservative classification applies to the whole Direct Comparison even when only one
Variant was influenced. The Specification also carries and verifies each Variant's canonical
representation so the Experiment remains interpretable without a machine-local registry; it
becomes immutable when the first Run starts, and semantic changes create a new linked
Experiment.
The initial repetition template uses one decision-bearing Run per Variant when every Variant
and dependency declares deterministic behaviour. If any compared Variant or dependency is
stochastic or lacks that guarantee, every Variant uses the frozen root-seed set
`[0, 1, 2, 3, 4]` against identical Partition assignments; deterministic baselines are also
repeated. All five Runs are planned before execution with no outcome-driven stopping. Per-Run
Assessment metrics, their arithmetic mean, and sample standard deviation are retained; no
confidence interval or significance claim is implied. An uncontrollable dependency seed is
reported as dependency-influenced variability rather than exact reproduction, and an invalid
or incomplete Run requires the frozen replacement policy or leaves the Result incomplete.
For end-to-end evaluation of Composite or Reranked Approaches, repetition root seed `s`
binds each constituent or base to the committed Artifact produced by its corresponding
source Run with root seed `s`, using the Experiment's exact Dataset, Problem–Dataset Binding,
and Fit Partition. The pairing rule freezes before execution, and each assembled Run retains
the source Run and exact Artifact lineage; source Artifact identities may resolve as their
planned Runs finish. Assembly consumes those Artifacts without refitting. Stochasticity in
constituent or base fitting counts when choosing the repetition template, even when assembly
and final inference are deterministic. A best-seed choice or an Artifact fitted with extra
Selection or Assessment observations cannot substitute for the planned input. Distinct source
Runs may legitimately produce identical content-addressed Artifacts without losing their
separate execution lineage. A separately predeclared Study may instead hold exact input
Artifacts fixed across repetitions, but its conclusions are conditional on those fitted
states and do not establish end-to-end fitting variability. These bindings belong to the
existing Experiment and Run lifecycle rather than a new pairing object.
Resource evidence for a Composite or Reranked Approach distinguishes assembly from creation
of the complete Approach from scratch. Assembly evidence measures only the actual resources
used to assemble already available input Artifacts. Creation-from-scratch evidence also
includes the Fit and export executions producing the constituent or base Artifacts, linked
through their exact execution lineage. The Experiment declares the claimed scope before
execution; reusing an Artifact does not make its upstream creation free for an end-to-end
claim. Missing required source-execution evidence makes that claim inconclusive while leaving
valid assembly evidence available. Wall time and peak memory from separate executions cannot
be mechanically summed into end-to-end elapsed time or peak memory; each claim must be
supported by its declared measurement and aggregation semantics. These scopes use existing
Run resource evidence and lineage without introducing another domain object or resource limit.
_Avoid_: Run Configuration, Mutable Experiment Record

**Run**:
One concrete execution instance belonging to exactly one Experiment. Multiple Runs can
execute the same Experiment without changing the Experiment's identity; post-assessment
Refit Builds are not Runs. A Run resolves already-materialized Approach Variant identities;
it may bind declared Run dimensions such as a root seed and environment placement but cannot
override Variant semantics. A versioned, domain-separated derivation rule resolves the Run
root seed into named sub-seeds for each declared mechanism, such as parameter initialization,
batch ordering, surrogate sampling, and dropout. It may apply only named sub-seed overrides
already frozen as planned Experiment dimensions and records the complete resolved map; CLI
execution cannot introduce another override. A Run stopped before completing a required Evaluation Cohort is
Incomplete: it may preserve partial development or operational evidence but cannot contribute
decision-bearing quality evidence. It may resume only unattempted cases while every
identity-bearing Reproducibility Envelope binding remains unchanged; changing such a binding
or replacing a recorded outcome creates a new Run. A Run retains the Specification-required
raw resource usage, such as wall time, CPU or accelerator use, peak memory, storage and
network use, and external calls or tokens, bound to its Execution Environment Snapshot.
Dependency-specific usage is grouped by canonical key and usage phase; per-invocation usage
is retained only when the Experiment Specification predeclares it.
Viewing an Assessment result does not disqualify later repetitions already frozen in the
Experiment Specification. A repetition added, removed, or stopped in response to observed
Assessment evidence cannot enter that Experiment's decision-bearing Result; it requires a
new linked Experiment, for which the reused Partition is `exposed`. When the frozen technical
replacement policy applies, a new full Run with the same planned dimensions may supersede an
invalid attempt; the failed Run and lineage remain in the evidence history, and only the
eligible replacement contributes to the Result. A completed valid Run cannot be replaced
because its metric is unfavorable. When available evidence cannot attribute an execution
failure to either the Approach or evaluation integrity, the Run records `unresolved`; it is
retained, cannot be replaced automatically, and cannot contribute a conclusive claim unless
later diagnostic evidence resolves the attribution.
_Avoid_: Experiment

**Run Failure Adjudication**:
An immutable evaluation-owned decision about one failed or `unresolved` Run, recording the
classification as Approach-attributed failure, Evaluation Integrity Failure, or still
`unresolved`, together with the rationale, diagnostic-evidence references, adjudicator
identity, and decision time. It does not rewrite the Run or its outcomes. It is required
before an ambiguous failure can be resolved or a technical replacement can be admitted; an
integrity classification permits replacement only when the frozen Experiment policy also
allows it. A correction creates a new Adjudication that explicitly supersedes the prior one,
which remains in the evidence history. Diagnostic executions referenced by an Adjudication
do not thereby become decision-bearing Experiment Runs. The Approach may emit errors and
diagnostic facts but cannot classify its own failure as an integrity defect; classification
is owned by the evaluation boundary. In a single-person Lab the same person may author the
Approach and act as adjudicator, but that role, rationale, and evidence remain explicit.
An automated Adjudication binds the Code Snapshot of its classification logic and the input
evidence; a manual Adjudication binds the adjudicator identity, rationale, and evidence. Both
forms use the same immutable supersession lifecycle without introducing a separate
Adjudication Policy object.
_Avoid_: Mutable Run Status, Inference Failure, Diagnostic Run

**Stochastic Mechanism**:
A declared algorithm, distribution, or policy that governs random behaviour. It belongs to
the semantic specification of the object that owns the behaviour—for example an Approach
Variant's surrogate-sampling distribution and refresh schedule or an Evaluation Protocol's
partition procedure—and changing it changes that owning specification. Each mechanism has a
stable canonical key used for seed derivation. Adding or removing a mechanism is semantic;
renaming an implementation symbol alone must preserve that key so it does not change a Run's
random realization.
_Avoid_: Random Seed, Random Library Implementation

**Random Seed**:
A scope-specific input selecting a reproducible realization of a Stochastic Mechanism. A
Run root seed creates another Run of the same Approach Variant; a versioned,
domain-separated derivation rule creates named mechanism sub-seeds so adding or consuming
randomness in one mechanism does not silently shift another. The root seed, derivation
version, mechanism names, any predeclared controlled overrides, and resolved sub-seeds are
retained. Direct sub-seed override is not a general authoring or CLI mechanism. A partition seed contributes
to Evaluation Partition identity, and a synthetic-data generation seed contributes to
Dataset identity; these scopes remain independent and are not derived from the Run root.
_Avoid_: Approach Hyperparameter, Stochastic Mechanism, Global Seed

**Run Measurement**:
An immutable metric observation derived for exactly one Run and bound to its Approach
Variant and exact evaluated target—either an Approach Artifact or Training Checkpoint digest—
Dataset, Evaluation Partition and Slice, Evaluation Metric Specification, evaluator Code
Snapshot, Execution Environment Snapshot, Evaluation Cohort, and actual inference or
Degradation Mode. Decision-bearing Measurements also reference the retained Evaluation
Outcome Set. A correction that preserves the declared metric semantics creates a superseding
Measurement within the same Experiment, while a semantic metric change requires a new Metric
Specification and linked Experiment. Primary Direct Comparison Measurements cover the full
predeclared Cohort; conditional metrics derived from an applied Algorithmic Backoff are
Variant-local outcome breakdowns and cannot serve as Direct Comparison evidence.
_Avoid_: Experiment Result, Research Finding

**Evaluation Outcome Set**:
An immutable record of case outcomes for exactly one Run, Evaluation Slice, and evaluated
Approach Artifact or Training Checkpoint digest, together with evaluation-case correlation
and associated Inference Receipt information; a complete set has one explicit outcome for
every included case, while an Incomplete Run may retain a clearly partial, append-only set
whose existing outcomes cannot be replaced on resume. Each recorded outcome is either the
exact final Problem-defined output or an Inference Failure. Complete sets are retained for
every checkpoint Measurement that informs Selection and for every Artifact Measurement that
informs Assessment, while exploratory Runs may omit them and internal candidates or scores
remain Diagnostic Observations.
_Avoid_: Evaluation Output Set, Run Measurement, Diagnostic Observation, Serving Response

**Inference Failure**:
An evaluation-case outcome in which the Approach does not produce a conforming
Problem-defined output under the declared Run conditions, after any permitted Degradation
Mode is accounted for. It remains attributable to the evaluated system rather than being
silently excluded; the Evaluation Protocol governs its measurement treatment. Exhausting a
declared resource limit, including an Approach-caused out-of-memory condition, is attributable
to the evaluated system and is not eligible for technical Run replacement. Failure on a
Cohort case that the Variant declared it could handle cannot retroactively remove that case
from the Variant's metrics.
_Avoid_: Evaluation Integrity Failure, Empty Valid Output, Silent Exclusion

**Evaluation Integrity Failure**:
A condition showing that the evaluation harness cannot faithfully execute or evidence the
frozen Experiment Specification because of an external infrastructure, data, or harness
defect, such as host loss, a corrupted Partition, or an evaluator contract defect. An empty
Judgment reaching metric aggregation after the Cohort was materialized is such a contract
defect: the metric layer fails closed instead of silently changing membership. It invalidates
the Run and cannot be converted into an Approach-attributed Inference Failure.
_Avoid_: Inference Failure, Poor Recommendation, Evaluation-case Exclusion

**Code Snapshot**:
An immutable, recoverable representation of executable source used by a Run, Refit Build,
Serving Trial, or Serving Qualification. The default form is a clean Git commit reachable
from declared repository provenance; a future alternative may be a safely scoped,
content-addressed source bundle,
while external qualification tools use an immutable version or container digest in execution
provenance. A dirty flag or patch alone is not a Snapshot, and the Lab does not blindly
capture untracked files; dirty executions produce only development evidence, while final
Experiment Results and Serving Qualifications require clean Snapshots for all
decision-affecting code. For a thin Approach Artifact, the inference runtime's Code Snapshot
is execution evidence rather than Artifact Identity. The same Artifact may run under a later
Snapshot only when that runtime declares support for the Artifact's format and runtime
contract and passes structural, integrity, and Problem-appropriate behavioural compatibility
checks; schema parsing alone is insufficient. Each decision-bearing Run, Serving Trial, and
Qualification remains scoped to its exact Artifact–runtime pair, so earlier evidence does not
transfer automatically. Reproducing earlier evidence exactly resolves its original Snapshot,
whereas evaluating or serving the Artifact with a later compatible Snapshot creates fresh
evidence without retraining, re-exporting, or minting another Artifact. The whole-repository
clean Git commit remains the initial coarse Snapshot: unrelated committed changes therefore
create a new execution Snapshot but do not churn thin-Artifact identity. A scoped
content-addressed source bundle is deferred until a concrete distribution or provenance need
justifies maintaining dependency discovery. Training- or export-only Snapshots remain
provenance.
_Avoid_: Branch Name, Dirty-worktree Flag, Untracked Working Tree

**Reproducibility Envelope**:
The declared conditions within which a Run can be meaningfully reproduced, including its
Specifications, exact Approach Artifact and separately bound runtime Code Snapshot,
dependencies, Execution Environment Specification and
Snapshot, External Dependency Snapshots, Run root seed, seed-derivation version, named resolved
sub-seeds, independently scoped Partition and synthetic-data seeds, determinism settings, and
numerical tolerances. It explicitly marks declared stochastic mechanisms that cannot be
seed-controlled. Persisted Dataset, Evaluation Partition, and Approach Artifact content uses
exact integrity hashes, while numerical reproduction may use declared tolerances when a
backend cannot guarantee bitwise identity. No Run implies bitwise reproducibility across
undeclared hardware, software, or external-service environments.
_Avoid_: Same Seed Guarantee, Universal Determinism

**Repeatability Evidence**:
Evidence about variability obtained from multiple planned Runs of the same Experiment under
an aligned Reproducibility Envelope while varying declared realization dimensions such as a
Run seed. One deterministic, replayed, or memoized execution does not establish
repeatability. For a non-seed-controllable dependency, only multiple fresh planned invocations
under the same declared stable External Dependency Specification and aligned Snapshots
support a repeatability claim; an exact
memoized replay is useful for resuming the same Run or recomputation but is not a new
realization. Recomputing Measurements from a retained Outcome Set performs no new inference
and therefore is not cross-Run memoization.
_Avoid_: Run Reproduction, Single-run Result

**Serving Workload**:
The protocol-defined distribution and arrival pattern of requests used to evaluate serving
behaviour, including relevant payload mix, concurrency, initial model state, warm-up
procedure and measurement inclusion, and measurement duration.
It treats cold-start and steady-state as distinct populations unless it explicitly declares
a representative mixture, and describes a benchmark population rather than the context of an
individual recommendation. It declares whether arrivals are open-loop and independent of
response completion or closed-loop and response-paced, together with offered load,
backpressure, and any overload threshold; capacity claims require the former, while the
latter remains valid for explicitly modeled interactive clients or component benchmarks.
_Avoid_: Recommendation Context, Production Traffic without qualification

**Approach Inference Latency**:
Elapsed time measured across the Approach Inference Interface, including its internal logic
and dependency calls but excluding the Serving Adapter and
external network. Component scoring time alone is not Approach Inference Latency.
_Avoid_: Core Scoring Latency, End-to-End Serving Latency, Unqualified Latency

**End-to-End Serving Latency**:
Elapsed time across explicitly declared serving observation points, including Adapter
queueing and processing plus Approach invocation; external network time is included only when
the declared client-side observation point covers it. It is the latency boundary required by
a Serving Qualification.
_Avoid_: Approach Inference Latency, Core Runtime Benchmark, Unqualified Latency

**Execution Environment Specification**:
The Experiment- or Serving Qualification Specification-owned declaration of required or
controlled execution conditions, including relevant runtime constraints, accelerator class,
topology, resource limits, and which dimensions must align for comparison. It defines
validity and comparability requirements without claiming what environment an execution
actually received; it does not require a container. An Approach Artifact may declare runtime
compatibility constraints, but exact language, package, library, driver, and hardware
versions remain environment conditions unless their bytes are bundled into Artifact Identity
by its export strategy.
_Avoid_: Execution Environment Snapshot, Approach Variant, Host Description

**Execution Environment Snapshot**:
The execution-owned record of the actual resolved environment for a Run, Refit Build, or
Serving Trial, including relevant hardware, operating system, language and dependency
versions, drivers, topology, resource limits, and container digest when applicable. It is
validated against the governing Execution Environment Specification; a mismatch excludes
the affected evidence rather than allowing a silent comparison. Device placement belongs
here rather than to Approach Variant identity, even though different environments may produce
different content-addressed Artifacts from the same Variant and Random Seed. A container used
only as an execution harness remains an environment condition; when the immutable container
itself is the Approach Artifact's export format, its digest also participates in Artifact
Identity.
_Avoid_: Execution Environment Specification, Deployment Requirement

**Production-grade Engineering**:
The software and evidence discipline applied throughout the Lab, including reproducibility,
verification, observability, and explicit failure handling. It favors practices with current
industry use or active ecosystem support and records why they fit this Lab, rather than
adopting a mechanism because it appears novel or enterprise-like. It does not make an
Approach Artifact universally ready for production.
_Avoid_: Production-ready Artifact, Production Certification

**Production-readiness Assessment**:
A conditional evaluation of an Approach Variant and Artifact against explicit target
requirements and acceptance criteria under a declared Serving Workload and Execution
Environment Specification. For an exact Artifact, its output may form a Serving
Qualification; it produces scoped evidence rather than a universal production-readiness
claim.
_Avoid_: Unqualified Production-ready Claim, Production Certification

**Experiment Result**:
An immutable, protocol-governed synthesis of Run Measurements for an Experiment, including
relevant variability, failures, and exclusions. An invalidated Result remains part of the
evidence history, while a correction creates a new Result that explicitly supersedes it. A
Result based on one Run makes that evidence limit explicit; assessment or final claims
include only Measurements bound to a Code Snapshot, while dirty-worktree measurements remain
development evidence. Missing a resource dimension required by the Experiment Specification
makes only the corresponding efficiency or derived monetary-cost claim inconclusive; it does
not invalidate independent quality evidence. Monetary cost is derived from retained raw usage
and explicit time-sensitive pricing assumptions rather than stored as an intrinsic property
of an Approach Variant, Artifact, or Result. An unresolved execution-failure attribution
makes the affected repetition and any claim that requires it inconclusive; it is neither
silently excluded nor counted against the Approach. A Result resolves failure attribution
through the latest non-superseded Run Failure Adjudication and retains that decision's
identity in its evidence lineage. Its conclusion cannot attribute an observed difference to
a narrower component than the factors the Experiment actually held constant or controlled.
A corrected Selection decision propagates to a superseding Result. The Result identifies
the corrected winner and any missing evidence needed to assess it, while preserving prior
Results and valid target-specific Measurements; evidence for a former winner cannot support
the new winner's assessment conclusion.
_Avoid_: Run Measurement, Research Finding

**Baseline**:
An Approach Variant designated by an Experiment as the reference for a comparison. A
Baseline has no Lab-wide status outside the Dataset, Evaluation Protocol, Evaluation
Cohort, and other conditions of that Experiment.
_Avoid_: Default Model, Universally Best Approach

**Direct Comparison**:
A comparative claim supported by Experiment Results produced under aligned Dataset and
Problem–Dataset Binding, Evaluation Protocol and Partition assignments, Evaluation Slice
and query form, Evaluation Cohort, Evaluation Judgment and metric semantics, and, for
operational claims, Serving Workload, Execution Environment Specification, and compatible
Execution Environment Snapshots. Variants may differ in Signal Preparation and other
Approach semantics, so this evidence supports an end-to-end Variant comparison by default;
attributing the difference to one model or component additionally requires the other
identity-bearing factors to be held constant or varied through a predeclared ablation.
Results from unrelated Experiments are not directly comparable by default.
_Avoid_: Cross-run Metric Juxtaposition

**Research Study**:
A structured investigation of one or more research questions or hypotheses that may span
multiple Recommendation Problems. It separates claims drawn from cited academic sources
from findings supported by linked Experiments, and records traceable Research Claims,
conclusions, and limitations.
_Avoid_: Paper Summary, Experiment

**Research Claim**:
An assertion promoted into a Research Study conclusion or used as rationale for an Approach
Specification. It is explicitly classified as a source-backed claim linked to a Research
Source Record and precise locator when available, an Experiment Finding linked to its
Experiment Result and Evaluation Slice, a Lab synthesis that identifies its input evidence,
or an unverified hypothesis. Informal Research Notes need not structure every sentence this
way. Conflicting Claims remain visible with their scopes and limitations rather than being
silently rewritten.
_Avoid_: Unattributed Conclusion, Research Note Fragment, Metric without Scope

**Approach Reimplementation**:
An independently written implementation intended to realize a Recommendation Approach
described by one or more Research Sources. Its Study declares the intended fidelity and any
known deviations; successful implementation alone does not claim that published results
were reproduced.
_Avoid_: Reported-result Reproduction, Reference Code Copy

**Reported-result Reproduction**:
A deliberate attempt to test a specific published result under materially aligned Dataset,
partition, preprocessing, metric, and execution conditions. It identifies the exact source
claim and declares whether it uses reference code or an Approach Reimplementation, together
with every known deviation and tolerance.
_Avoid_: Approach Reimplementation, Extension Study, Similar Metric Value

**Extension Study**:
A Research Study that deliberately evaluates an existing Recommendation Approach or claim
under a different Dataset, Recommendation Problem, Evaluation Protocol, or other material
condition. It contributes new scoped evidence rather than claiming to reproduce the source
result. The unqualified word `replication` is avoided unless the Study defines its intended
meaning explicitly.
_Avoid_: Reported-result Reproduction, Direct Comparison to an Unaligned Paper Result

**Research Source**:
An externally authored academic work or authoritative reference cited by the Lab. Its
inclusion preserves provenance and does not make its claims findings of the Lab.
The work is represented in the repository by a Research Source Record rather than by an
assumption that its full text can be redistributed.
_Avoid_: Research Source Record, Research Note, Research Study

**Research Source Record**:
The version-controlled metadata identifying the exact Research Source consulted, including
its citation, persistent identifier and version, access location and date, content checksum
when obtainable, and license or redistribution status. Git stores this Record by default,
not the source binary. A PDF or other full-text copy enters version control only when its
redistribution rights and repository size policy permit; otherwise it remains in an ignored
local cache or authorized external store while the Record preserves retrieval and identity
information.
_Avoid_: PDF Filename as Identity, Bare URL, Research Note

**Research Note**:
A Lab-authored summary, critique, or synthesis of one or more Research Sources. It may
inform a Research Study but is not evidence produced by a Lab Experiment. It references
Research Source Records rather than relying on an untracked local paper copy.
_Avoid_: Research Source, Research Study, Experiment
