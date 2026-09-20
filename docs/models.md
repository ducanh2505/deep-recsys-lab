# Built-in models and retrievers

Built-ins are implementations behind the same contracts; none changes the data schema or
artifact identity rules.

## Model plugins

- **MultiVAE** learns a variational autoencoder over weighted user–item rows and exports a
  dense ONNX scorer. It supports known-user and interaction-history queries.
- **LightGCN** propagates trainable embeddings over a normalized bipartite graph and exports
  final user/item embeddings. It supports known users and history queries through a mean
  item-embedding profile.
- **BPR** trains pairwise user/item embeddings and supports known users plus mean history
  profiles.
- **SASRec** learns causal self-attention over ordered interaction sequences and exports an
  ONNX scorer plus known-user sequence state. It supports known-user and history queries.
- **Two-tower** exports aligned user and item representations usable by inner-product
  retrieval.

## Retriever plugins

The registry also includes popularity, item-nearest-neighbour, TF-IDF, deterministic
semantic feature embeddings, Markov transitions, graph co-occurrence, and graph embeddings.
Model plugins can be selected as retrievers inside a hybrid pipeline.

## Fusion

Reciprocal-rank fusion and weighted score fusion require no learned state. Learned LightGBM
fusion uses the native LightGBM model format; Python object serialization is not part of a
served artifact.

Hybrid component lists and parameters come entirely from configuration. No dataset-specific
ensemble is implied by the package defaults.
