# Use Learned Hybrid Fusion as the final MVP order

The MVP adapts the paper's Learned Hybrid Fusion as a validation-trained classifier over the
union of Popularity, ItemKNN, Mult-VAE, and eligible LightGCN Candidate Pools. Separate fusion
artifacts serve Known-User and History-Only Queries, and the LHF order supplies the final Top-N
directly; the paper's additional downstream ranker remains out of scope so the project can
demonstrate the retrieval bottleneck and a complete serving path without a second learned
ranking stage.
