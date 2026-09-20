# Separate item co-occurrence from co-occurrence SVD

The current direct projected-item graph and its SVD-derived recommender are separate
Recommendation Approaches with stable keys `item_cooccurrence` and `cooccurrence_svd`.
Although they may share Signal Preparation and graph-construction implementation, SVD adds an
invariant representation-learning mechanism and changes recommendation behavior; placing a
`raw | svd` switch inside one umbrella Approach would obscure that distinction and weaken
their identities. Experiments can still align all other bindings to compare the two.

The broad implementation names `graph_cooccurrence` and `graph_embeddings` are conflicts to
replace before these Approach identities become durable: the former does not distinguish an
item projection from a user–item graph, while the latter incorrectly suggests node2vec,
random-walk, or graph-neural families. Those future methods receive their own Approach
Specifications rather than inheriting the SVD identity. The evidence and alternatives are
summarized in [the graph Signal Preparation research note](../research/graph-signal-preparation.md).

The first implicit-feedback Variant of each Approach binds `binary_positive` for both fitting
and inference: qualifying events satisfy `value > 0`, each Subject–Candidate pair contributes
once, known-Subject profiles preserve that same representation, and supplied history is
collapsed identically before scoring. Frequency or value weighting is a different Variant.
Both Variants declare `known_subject` and `supplied_history` query capabilities. Equivalent
binary-positive profiles must produce the same ranking, while an unknown Subject fails rather
than being reinterpreted as history, popularity, or another query form.
Their Approach Artifacts store immutable `binary_subject_profiles` derived from the Fit
Partition and use them directly for `known_subject`. This fitted payload is owned by and shares
the lifecycle of the Approach Artifact; it is neither a separate domain object nor a cache.
The runtime must not reconstruct these profiles from a generic raw or summed train matrix.
A known-Subject or supplied-history query whose `binary_positive` profile is empty lacks the
required query evidence. Inference fails; evaluation excludes the case before inference under
`missing_required_query_evidence`; and the Approach cannot return popularity or canonical-order
fallback results.
In contrast, a non-empty prepared profile remains valid when learned associations produce an
all-zero or fully tied score vector. Canonical tie resolution yields the required ordering and
the case remains in evaluation; this is a degenerate model result, not a backoff. A
rank-deficient or all-zero graph may produce a research Artifact, with serving acceptability
decided later by Serving Qualification rather than hidden Fit or evaluation exclusion.
Its initial projection-context binding is `whole_subject_fit_history`: construct the incidence
matrix with one row per Subject over all of that Subject's Fit observations, then project it as
`C = XᵀX`. This intentionally ignores within-history order and temporal distance. Session-,
basket-, or time-window graph construction changes the graph semantics and therefore requires
a separate identity-bearing Variant. The initial edge-weight binding is
`distinct_subject_count`: `C[i,j]` is the number of distinct Subjects whose Fit histories
contain both Candidates, so one Subject contributes at most one unit to a pair. Per-Subject
normalization based on history length changes the edge meaning and belongs to another Variant.
Both initial Variants bind `min_distinct_subject_count = 1`, retaining every eligible
observed edge. A higher pruning threshold changes the fitted graph and output behavior;
its resolved value therefore participates in Variant identity and cannot be supplied as a
semantic execution-time override. Both Variants also bind `zero_self_edges`, removing the
projection diagonal before direct scoring or SVD and suppressing any diagonal that low-rank
reconstruction reintroduces during scoring. This prevents self-evidence and popularity on
`C[i,i]` from dominating the graph, but does not exclude a Candidate already present in a
query: it remains output-eligible and may receive evidence from other history Candidates.
The initial `item_cooccurrence` Variant binds `sum_neighbor_weights`: an output Candidate's
score is the sum of its edge weights from every Candidate in the `binary_positive` query
history. Maximum or other aggregation semantics require another Variant.
Its runtime/export binding is `sparse_full_graph`: the Approach Artifact stores the complete
sparse graph after pruning and self-edge removal, and inference computes `q C` without
per-Candidate neighbor truncation. This payload shares the existing Approach Artifact
lifecycle and introduces no new domain object.
Its internal ranking score is summed distinct-Subject support from the query profile.
The initial `cooccurrence_svd` Variant factorizes this same raw, pruned, zero-diagonal
distinct-Subject-count matrix, without popularity, marginal, PMI, or cosine normalization.
This makes low-rank factorization the material difference in an aligned comparison with the
initial `item_cooccurrence` Variant. A normalized association-matrix factorization is another
Variant.
Its factorization semantics are `rank_k_reconstruction`: given declared rank `k`, recommendation
scores derive from `C_k = U_k Σ_k V_kᵀ`, the truncated SVD reconstruction of the graph.
The current implementation instead forms `E = V_k sqrt(Σ_k)` and serves `E Eᵀ`. A
zero-diagonal co-occurrence graph need not be positive semidefinite, so that Gram matrix is not
generally its truncated SVD reconstruction; it introduces distinct embedding semantics and is
an implementation conflict with this Variant.
The initial Variant also binds `signed_reconstruction_scores`: all signed off-diagonal
reconstruction values participate in scoring. A negative value is a latent contribution of
the approximation rather than an Explicit Negative Feedback observation. Clipping negative
values or retaining only positive similarities changes the reconstruction semantics and
requires another Variant. The current positive-only neighbor materialization is therefore an
additional implementation conflict.
Query scoring binds `sum_reconstructed_weights`: an eligible output Candidate's score is the
sum of its signed reconstructed relationships from every Candidate in the `binary_positive`
query history, without cosine or latent-vector normalization. This aligns query aggregation
with `item_cooccurrence` while isolating low-rank reconstruction as the material difference.
The runtime/export binding is `factorized_full_catalog`: the Approach Artifact stores `U_k`,
the singular-value vector, and `V_kᵀ`, then computes `q U_k Σ_k V_kᵀ` without materializing
dense `C_k` or truncating Candidate neighborhoods. This preserves exact signed full-Catalog
scores for the fitted rank; any reconstructed diagonal is suppressed to honor
`zero_self_edges`. These arrays are Artifact payloads under the existing Artifact lifecycle,
not a new domain object.
Its internal ranking score is signed reconstructed support. Scores from both Approaches are
uncalibrated ranking evidence meaningful only within one query; they are neither probability
nor confidence and cannot be compared across queries or Approaches. Both Variants return only
the final ordered Candidate identifiers in evaluation and serving, following the shared
Personalized Top-N Ranking output contract. Scores remain internal ranking or composition
evidence; retention requires explicitly declared Diagnostic Observations. This keeps the
output and export-parity contract uniform across Approaches, at the cost of requiring opt-in
diagnostic capture for score analysis rather than exposing scores to every caller.
Fit uses a sparse truncated-SVD solver fixed by the Approach Specification, not a public
binding. Solver randomness consumes a named sub-seed derived from the Run seed. Failure to
converge fails the Run; the implementation cannot silently switch solver or densify the graph.
A public solver binding is deferred until a concrete Experiment compares solver families.
The declared `rank` is exact, participates in Variant identity, and must satisfy
`1 <= rank < candidate_count` for the Experiment's Dataset. An incompatible Experiment fails
preflight instead of clamping the binding. Numerical rank deficiency in a dimensionally
compatible matrix remains valid and yields zero trailing singular values. The current
`min(requested_rank, matrix_dimension)` behavior is an implementation conflict.
The initial `cooccurrence_svd` Variant binds `rank = 32`. A Dataset with
`candidate_count <= 32` is incompatible with that Variant and requires a separately identified
lower-rank Variant rather than a data-dependent effective rank.

## Initial binding summary

| Scope | Binding | Initial value |
| --- | --- | --- |
| Shared | Signal Preparation | `binary_positive` for Fit and query evidence |
| Shared | Query capabilities | `known_subject`, `supplied_history` |
| Shared | Known-Subject fitted state | `binary_subject_profiles` Artifact payload |
| Shared | Empty prepared query | Fail; evaluation reason `missing_required_query_evidence` |
| Shared | Non-empty query with tied or zero scores | Valid result; canonical tie resolution |
| Shared | Projection context | `whole_subject_fit_history` |
| Shared | Edge weight | `distinct_subject_count` |
| Shared | Edge pruning | `min_distinct_subject_count = 1` |
| Shared | Self-edge policy | `zero_self_edges` |
| Shared | Score exposure | Internal ranking/composition evidence; opt-in Diagnostic Observations only |
| `item_cooccurrence` | Query scoring | `sum_neighbor_weights` |
| `item_cooccurrence` | Runtime/export | `sparse_full_graph` |
| `item_cooccurrence` | Score meaning | Summed distinct-Subject support |
| `cooccurrence_svd` | Factorization input | `raw_pruned_cooccurrence` |
| `cooccurrence_svd` | Factorization | `rank_k_reconstruction` |
| `cooccurrence_svd` | Rank | exact `32` |
| `cooccurrence_svd` | Reconstructed values | `signed_reconstruction_scores` |
| `cooccurrence_svd` | Query scoring | `sum_reconstructed_weights` |
| `cooccurrence_svd` | Runtime/export | `factorized_full_catalog` |
| `cooccurrence_svd` | Score meaning | Signed reconstructed support |
| `cooccurrence_svd` | Fit solver | Specification-fixed sparse truncated SVD; not a public binding |

These bindings inherit the Problem and Approach contracts already decided elsewhere: the
complete Candidate Catalog remains eligible, seen Candidates are not filtered, exact Top-N is
returned, and ties use ascending canonical Candidate identity. They introduce no separate
Signal Preparation object or derived-data lifecycle.

## Current implementation mapping

| Current code | Relation to the initial bindings |
| --- | --- |
| `graph_cooccurrence` and `graph_embeddings` plugin keys | Conflict with the canonical Approach keys |
| Shared summed `PreparedDataset.train`, followed by structural binarization | Conflict with event-level `binary_positive`; negative or cancelled values can become presence |
| `C = XᵀX`, zero diagonal, and `min_count = 1` | Structurally aligned with the projection, self-edge, and default pruning bindings once `X` is prepared correctly |
| `min_count` exists only for `graph_cooccurrence`; SVD hard-codes the default | Missing one shared strict binding schema and normalized names |
| `SparseRuntime` computes `query.vector @ similarity` | Aligned with sum aggregation only after query evidence and the stored representation conform |
| Generic raw `train` is always bundled and known-Subject vectors reuse its summed rows | Conflict with `binary_subject_profiles` and inference-side `binary_positive` |
| Request validation requires raw non-empty history but permits a post-filter empty profile | Missing prepared-query validation; zero evidence can become arbitrary tie-ordered output |
| `embedding_dim`, silently clamped to matrix size | Conflict with exact `rank` and preflight validation |
| Dense full-matrix `numpy.linalg.svd` | Conflict with the sparse truncated-SVD Fit invariant |
| `E = V_k sqrt(Σ_k)` followed by `E Eᵀ` | Conflict with `rank_k_reconstruction` |
| Positive-only top-`neighbours` materialization | Conflict with signed full-Catalog reconstruction scoring |
| Retriever capability advertises history queries only | Conflict with the two declared query capabilities |
| Open parameter dictionaries and code-local defaults | Conflict with strict Approach-owned schemas and fully materialized Variant identity |
| Generic serving engine applies shared score-vector Top-K and tie resolution | Conflict with Approach-owned final ordering, although its current canonical-index order happens to align with canonical identifiers |

The current implementation conflicts in both directions—Fit binarizes structural coordinates
only after shared values have been summed, while inference weights raw summed histories—and
must replace both paths together rather than creating a Fit/runtime semantic mismatch.
