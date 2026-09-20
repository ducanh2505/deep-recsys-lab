# Graph co-occurrence and graph-embedding signal preparation

Status: research note, not an architectural decision
Reviewed: 2026-09-17
Decision: [ADR-0029](../adr/0029-separate-item-cooccurrence-from-cooccurrence-svd.md)

## Question

What do the current `graph_cooccurrence` and `graph_embeddings` Approaches actually
compute, and which Signal Preparation semantics should they use for the current
implicit-feedback Problem? In particular, should repeated positive events be collapsed,
counted, or weighted, and how do item-item graphs differ from user-item bipartite graphs?

## Conclusion

The two current built-ins are not two different graph-construction families. Both first
construct the same item-item projection of a subject-item matrix. `graph_cooccurrence`
serves that projection directly, while `graph_embeddings` derives a low-rank similarity
matrix from its singular vectors. The latter is a **co-occurrence SVD**, not node2vec, a
random-walk embedding, or a graph neural recommender.

For their first explicit implicit-feedback Variant, both Approaches should bind the existing
`binary_positive` Signal Preparation:

```text
X[u, i] = 1  iff at least one Fit event for (u, i) has value > 0
C[i, j] = sum_u X[u, i] * X[u, j], for i != j
```

Thus `C[i, j]` is the number of distinct positive Subjects shared by two Candidates. One
Subject contributes at most one unit to a Candidate pair, regardless of how many qualifying
events that Subject generated. This is a sound, simple baseline supported by common implicit
item-item formulations; it is not a universal rule for all graph recommenders.

Event frequency and other weights remain legitimate, but must be separate identity-bearing
Variants. A weighted Variant must declare the exact event-to-edge transform rather than just
say “use counts”: using a frequency matrix `F` in `F.T @ F` makes one Subject's contribution
to edge `(i, j)` equal to `F[u, i] * F[u, j]`, which is materially different from counting one
co-occurring pair or applying a confidence transform.

No new domain object or cache is needed. The Approach Specification and Variant can own the
binding and the graph-construction details. A final domain decision should also cover query
preparation: the current runtime weights neighbors by raw, summed history values, so changing
Fit to `binary_positive` alone would leave a hidden Fit/inference semantic mismatch.

## What the repository computes now

### Shared input matrix

[`prepare.py`](../../src/recsys/datasets/prepare.py) constructs the shared
`PreparedDataset.train` CSR matrix from raw event values and calls `sum_duplicates()`. It does
not apply the current implicit qualifying predicate (`value > 0`) and does not remove stored
zeros.

[`cooccurrence()`](../../src/recsys/retrieval/algorithms.py) then copies that CSR matrix and
sets **every stored value** to `1`. Consequently, its actual input is the sparse matrix's
structural support, not positive preference:

```text
B[u, i] = 1 iff the shared CSR has a stored coordinate (u, i)
C = B.T @ B
diag(C) = 0
```

This admits negative-only observations. It can also admit duplicate values that cancel to a
stored zero, because SciPy defines CSR `nnz`/`getnnz` in terms of stored entries, including
explicit zeros, and provides `eliminate_zeros()` as a separate operation
([SciPy CSR documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html),
[SciPy `eliminate_zeros`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.eliminate_zeros.html)).
The result can therefore change with sparse storage representation even when the dense values
are identical. This is an implementation conflict, not a useful recommendation semantic.

Repeated rows for the same Subject-Candidate coordinate are already collapsed by the shared
matrix. Once the coordinate exists, `cooccurrence()` ignores its numeric value. Repetition
therefore does not increase a pair edge today, but this happens after an incorrect raw-value
sum rather than after filtering qualifying events.

### `graph_cooccurrence`

[`GraphCooccurrenceRetriever`](../../src/recsys/retrieval/builtins.py) stores `C` as its sparse
item-item similarity matrix after removing self-loops and dropping edges below `min_count`.
The sparse runtime scores a history vector `q` as `q @ C`; it does not normalize for Candidate
popularity, Subject degree, or history length.

### `graph_embeddings`

[`GraphEmbeddingRetriever`](../../src/recsys/retrieval/builtins.py) starts from the same `C`,
densifies it, and computes a full SVD. For requested dimension `k`, it forms
`E = V_k * sqrt(S_k)`, computes `E @ E.T`, and retains positive top neighbors. The Artifact
stores that final sparse neighbor matrix and uses the same sparse runtime; it does not store or
serve the embeddings themselves.

This is only one possible co-occurrence-SVD design. In particular, it is not the Item2Vec
paper's SVD baseline: that baseline normalizes the item-pair count matrix by row/column mass,
uses `U * sqrt(S)`, and compares embeddings by cosine similarity
([Item2Vec, Section 4.2](https://arxiv.org/pdf/1603.04259)). The current code neither performs
that input normalization nor cosine-normalizes its derived embeddings.

### Query weighting is a separate unresolved semantic

[`engine.py`](../../src/recsys/serving/engine.py) sums incoming history values per Candidate;
known-user requests reuse the shared summed train row. [`SparseRuntime`](../../src/recsys/artifacts/runtimes.py)
then multiplies that numeric vector by the similarity matrix. Hence a Subject with 100 repeated
events may contribute 100 times as much at inference even if fitting is changed to one binary
presence. That behavior could be a deliberate Variant, but it must not remain an incidental
consequence of the shared matrix/request adapter. The simplest consistent initial Variant is
to apply `binary_positive` to both Fit observations and history-query observations.

### LightGCN is a different graph boundary

The repository's [`LightGCN`](../../src/recsys/models/lightgcn.py) is separate from
`graph_embeddings`. It propagates embeddings on a graph containing both user and item nodes.
Its current code derives degrees from stored CSR coordinates and therefore inherits the same
positive-filtering/storage-support problem, but its topology and learning objective are not
the same as factorizing the projected item graph.

## Findings from primary sources

### Item-item projection commonly starts from a subject-item representation

The early item-based collaborative-filtering formulation treats items as columns of a
user-item matrix and derives item relationships from those vectors
([Sarwar et al., 2001](https://files.grouplens.org/papers/www10_sarwar.pdf)). That formulation
also illustrates why there is no universal binary rule: the original setting uses ratings and
compares cosine, correlation, and adjusted-cosine similarities.

For implicit feedback, EASE states that `X` is typically binary, with a positive entry meaning
that a user interacted with an item. It explicitly identifies `X.T @ X` as an item-item
co-occurrence matrix when `X` is binary
([EASE, Sections 2 and 3.2](https://arxiv.org/abs/1905.03375)). Under that construction, an
off-diagonal entry counts Subjects shared by the two items; repeats have already been collapsed
at the Subject-Candidate boundary.

The scope of “co-occurrence” is itself semantic. Item2Vec treats input as sets or baskets,
deliberately discards order/time, and uses every pair in the same set as a positive example
([Item2Vec, Section 3](https://arxiv.org/pdf/1603.04259)). A whole Subject history, a shopping
basket, a session, and a temporal window therefore induce different graphs even when they all
use binary-positive events. The current repo uses the entire Subject Fit row; a future
session/window graph must be a different declared construction, not an implementation switch.

### “Graph embedding” does not determine event semantics

Item2Vec learns item vectors directly from item-pair examples, while its SVD baseline embeds a
normalized pair-count matrix. DeepWalk learns node representations from truncated random
walks over an already supplied graph
([DeepWalk](https://arxiv.org/abs/1403.6652)). Node2vec likewise receives a graph; for a
weighted graph its transition probability incorporates the supplied edge weight, and for an
unweighted graph that weight is one
([node2vec, Section 3.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5108654/)). None of these
algorithms decides whether ten clicks should become one edge, ten units of weight, a logarithmic
confidence, or ten session-local pairs. That decision belongs to graph/signal construction
before embedding.

LightGCN demonstrates the other major topology. It defines a binary user-item interaction
matrix `R`, with `R[u, i] = 1` when the user interacted with the item, and builds the bipartite
block adjacency `[[0, R], [R.T, 0]]`; embeddings propagate along those user-item edges
([LightGCN, Section 3.1.3](https://arxiv.org/abs/2002.02126)). In contrast, `C = X.T @ X`
projects away user nodes and connects items directly. Projection creates a clique among the
items in each context and loses the identity of the intermediate Subject. These graphs may
encode related evidence, but they are not interchangeable Approach inputs.

### Binary presence and repeated-event strength are distinct models

Hu, Koren, and Volinsky's implicit-feedback formulation makes the distinction explicit. Raw
consumption `r[u, i]` is converted into binary preference `p[u, i]`, while repeated consumption
controls a separate confidence `c[u, i]`; the paper gives both linear and logarithmic confidence
transforms ([Hu et al., Sections 4 and 7](https://yifanhu.net/PUB/cf.pdf)). This is strong
evidence against treating raw event summation as a harmless implementation detail. Binary
presence, frequency, and transformed confidence answer different modeling questions.

## Topology and weighting choices

| Family | Nodes and edges | What one repeated event can mean |
| --- | --- | --- |
| Binary item projection | Candidate nodes; edge weight is distinct Subjects shared by a pair | Nothing after the first qualifying Subject-Candidate event |
| Frequency/value item projection | Candidate nodes; edge derived from weighted Subject-Candidate values | Potentially multiplicative contribution under `F.T @ F`; transform must be declared |
| Basket/session/window graph | Candidate nodes; pair exists within a declared context unit | Another pair occurrence only if it belongs to a qualifying context under the declared rule |
| User-item bipartite graph | Subject and Candidate nodes; direct interaction edges | Either no new edge after binary collapse or a changed edge weight in an explicitly weighted model |
| Random-walk graph embedding | Whatever nodes/weighted edges the graph constructor supplies | Changes walk probabilities only if construction changes the edge weight |

## Recommendation for this repository

1. Bind both current `graph_cooccurrence` and `graph_embeddings` implicit-feedback Variants to
   `binary_positive`. Build `X` from event-level Fit rows by filtering `value > 0` first and
   then collapsing each Subject-Candidate pair to one.
2. Define the current graph construction explicitly as an unordered whole-Subject Fit-history
   projection: `C = X.T @ X`, zero diagonal, raw distinct-Subject edge count, and a declared
   `min_count`. `binary_positive` alone does not specify this context/projection.
3. Apply the same binary-positive collapse to the history profile used by these initial
   Variants at inference. If frequency-weighted query evidence is worth studying, make it an
   explicit Variant rather than mixing it silently with binary Fit.
4. Keep `graph_embeddings` on the same graph for a controlled raw-graph versus compressed-graph
   comparison, but describe it as `cooccurrence_svd`. Consider renaming the Approach before its
   identifier becomes a durable public contract; “graph embeddings” is too broad to distinguish
   it from Item2Vec, node2vec, or LightGCN.
5. Introduce weighted Variants only with an exact formula: qualifying predicate, per-pair event
   aggregation, optional confidence/time transform, context unit, projection, normalization,
   and query weighting. Do not reuse the shared summed matrix as a generic default.
6. Treat LightGCN separately. Its original binary bipartite construction also supports
   `binary_positive` as a plausible default, but that should be confirmed as a LightGCN
   Approach decision rather than inferred from the two projected-item Approaches.

This recommendation preserves a small reusable event-to-signal seam without pretending that
all graph methods share one graph. It also creates a clean future ablation: keep graph
construction and algorithm fixed while changing only binary presence versus a declared
frequency/confidence transform.

## Minimum tests implied by the recommendation

These are acceptance examples for a later implementation, not changes made by this note:

- `+1` and `-1` events for the same Subject-Candidate still yield binary presence because the
  qualifying `+1` event is filtered before collapse; they must not cancel through summation.
- Negative-only and zero-only Subject-Candidate observations create no graph edge.
- One versus 100 qualifying repeats from the same Subject produce the same binary graph.
- Two distinct positive Subjects shared by Candidates `A` and `B` produce `C[A, B] = 2`.
- Sparse explicit zeros cannot change the graph.
- A history query with repeated positive events matches its binary-collapsed form for the
  initial Variant.
- Item-item projection and bipartite LightGCN fixtures remain separate and cannot accidentally
  consume each other's graph representation.
