# LightGCN on MovieLens 20M

## Source and data

The implementation follows He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* (SIGIR 2020), in [`paper/lightGCN.pdf`](../paper/lightGCN.pdf), especially equations (3)-(8), (15), and Section 4.1.2. The prepared MovieLens data and its project-specific split are described in [`data_preparation.md`](data_preparation.md). Those data preparation rules are not claimed to be the paper's experimental protocol.

The default training graph uses **only** `train.parquet`: 7,238,638 positive interactions, 129,757 users, and 11,508 movies. Original MovieLens IDs are mapped to contiguous embedding indices and saved in the checkpoint. Ratings and timestamps do not enter the model. In this default architecture-search protocol, validation and test rows are never graph edges or positive training examples. The separate [train + validation retraining protocol](lightgcn_retraining.md) merges validation into training after architecture selection and keeps test held out.

## Architecture

Let `A = [[0,R],[Rᵀ,0]]`, with `R[u,i] = 1` exactly when `(u,i)` is in train, and let `D` contain the node degrees. The sparse propagation is

```text
E^(k+1) = D^(-1/2) A D^(-1/2) E^(k)
E        = α_0 E^(0) + ... + α_K E^(K),  α_k >= 0,  sum(α_k) = 1
score(u,i) = <E_u, E_i>
```

There are no self-loops, feature transformations, nonlinear activations, or dropout. Only user and item embeddings at layer zero are learned. They use Xavier uniform initialization. The run uses `K=3` layers and 64 dimensions, the satisfactory/default choices in paper Section 4.1.2. Uniform layer weights match the paper's experiments; `--layer-weights` can set the nonnegative `α_0,...,α_K` of its equation (4), summing to 1. The normalized graph has 14,477,276 directed entries and is stored as a PyTorch sparse COO tensor. Device selection prefers CUDA, then MPS, then CPU.

The training loss for each sampled `(u, positive i, negative j)` is `softplus(score(u,j) - score(u,i))`, plus `λ/2` times the squared norms of the sampled layer-zero user, positive item, and negative item embeddings, averaged over the mini-batch. Negatives are sampled uniformly from movies absent from that user's **train** history. Adam updates the embeddings. This is BPR with sampled L2 regularization as used in common LightGCN implementations; the paper's equation (15) writes the unnormalized sum of triplet losses and the global norm. Therefore the numeric value of `λ` is sensitive to this reduction convention.

## Train and inspect the result

From the repository root:

```bash
uv sync
uv run python -m lightgcn.train \
  --batch-size 524288 --max-epochs 30 --eval-every 2 \
  --patience 5 --validation-users 20000
```

The larger mini-batch is a hardware/runtime choice for this 7.2-million-edge dataset. The paper uses 1,024 (or 2,048 on Amazon-Book); the model and loss remain the same. `--validation-users 0` evaluates every eligible validation user at each check. The 20,000-user default is a fixed, evenly spaced subset for checkpoint selection; the selected checkpoint is also evaluated on **full** validation. Test is evaluated at the end unless `--skip-test` is passed. `--batch-size`, `--layers`, `--embedding-dim`, `--learning-rate`, `--l2`, `--seed`, and stopping options are configurable. The dataset and output paths are configurable as well.

The completed architecture search placed its selected checkpoint under `artifacts/lightgcn_ml20m/`. Running the command above again will replace that directory with a new baseline run; use `--output-dir` to keep runs separate. Training-only search trials can pass `--skip-test`, and `--validation-users 0 --eval-every 1` checks the entire validation set every epoch.

Outputs under a training run's output directory:

| File | Contents |
| --- | --- |
| `best.pt` | Best validation checkpoint: layer-zero PyTorch weights, original ID maps, config, epoch and selection score. |
| `config.json` | Exact run configuration and split counts. |
| `history.json` | Mean train loss, duration, and periodic validation scores. |
| `metrics.json` | Selection and full-validation metrics; also final test metrics unless `--skip-test` was passed. |

The checkpoint can be loaded with `torch.load(path, map_location="cpu", weights_only=True)`, then `LightGCN(n_users, n_items, embedding_dim, layers, layer_weights=tuple(checkpoint["config"]["layer_weights"])).load_state_dict(checkpoint["state_dict"])`. Use the saved config for the architecture and layer weights. To rank items, reconstruct the **train-only** adjacency, call `propagate`, score candidates by dot product, and exclude movies already observed in train. This project does not include serving.

## Evaluation protocol

Recall@20 and NDCG@20 use all 11,508 movies as ranking candidates, excluding each user's train movies for validation and train plus validation movies for test. The denominator for Recall@20 is that user's number of target interactions. NDCG@20 divides discounted hits by the ideal discounted gain for up to 20 relevant movies. Users with zero target interactions are omitted from that split's mean; there are 4,790 such users in validation due to the split rounding. The test set is touched only after checkpoint selection. Intermediate validation runs use the same fixed subset for every epoch.

## Hyperparameter search strategy

Use the supplied validation split for every comparison, keep the test set untouched until the final selected run, and use the same fixed seed and evaluation users across candidates. This search changes only choices present in the paper:

1. Start with the paper baseline: 64 dimensions, `K=3`, `α_k=1/(K+1)`, Adam learning rate `0.001`, and `λ=1e-4`.
2. Search `λ ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}` on a common 20,000-user validation subset. The paper examines these regularization values and notes that values above `1e-3` may hurt; `1e-2` is useful as a boundary check. Compare the best Recall@20 and use NDCG@20 to break close ties.
3. At the best `λ`, search the paper's propagation depths `K ∈ {1,2,3,4}`. Keep uniform layer weights. The paper finds three layers satisfactory in most cases, but the best value can differ on MovieLens.
4. If gains have not stabilized, vary the learning rate around the paper's `0.001`, for example `{0.0003, 0.001, 0.003}`. At the best depth, one may also compare uniform `α` against a few fixed, nonnegative layer weight vectors from equation (4), keeping their sum at 1. Keep Adam, Xavier initialization, BPR, uniform negative sampling, and 64 dimensions. An optional capacity check can compare 32/64/128 dimensions, but that would no longer reproduce the paper's fixed-size comparison.
5. Re-run the best few configurations on **all** eligible validation users with identical stopping rules. Select one checkpoint by Recall@20, using NDCG@20 as secondary evidence. Evaluate test once, then record the exact config and data manifest hashes next to the checkpoint.

Keep the same batch size during candidate comparisons because the sampled regularizer and Adam dynamics depend on it. A full Cartesian search is costly: each candidate uses the whole training graph. The staged search above narrows this to the paper's most relevant choices without adding architectural variants or sampling methods.

## Initial baseline run

The initial baseline is archived at [`artifacts/lightgcn_search/baseline_k3_d64_l1e-4_lr1e-3/`](../artifacts/lightgcn_search/baseline_k3_d64_l1e-4_lr1e-3/). It used Apple MPS, seed 42, 64 dimensions, `K=3`, uniform `α`, Adam learning rate `0.001`, `λ=0.0001`, and mini-batches of 524,288. Every epoch traversed all 7,238,638 train interactions once with one freshly sampled negative per positive. Validation was checked every two epochs on a fixed 20,000-user subset. Early stopping ended at epoch 16 after five checks without an improvement over epoch 6.

| Measurement | Users | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: |
| Checkpoint selection, validation subset, epoch 6 | 20,000 | 0.145441 | 0.087688 |
| Selected checkpoint, full validation | 124,967 | 0.144253 | 0.087958 |
| Selected checkpoint, full test | 129,757 | 0.148601 | 0.141182 |

The mean sampled training loss was 0.354451 at the selected epoch and 0.230488 at epoch 16. The fall in training loss after epoch 6 did not improve validation Recall@20, so the earlier weights were retained. The archived `history.json` contains every epoch's measured loss and periodic validation metrics. These MovieLens numbers should not be compared directly with the paper's Gowalla, Yelp2018, or Amazon-Book results because the datasets and split protocols differ.

## Selected model after architecture search

The current checkpoint at [`artifacts/lightgcn_ml20m/best.pt`](../artifacts/lightgcn_ml20m/best.pt) is the selected K=2, 32-dimensional model with layer weights `(0.975, 0.0125, 0.0125)`, Adam learning rate `0.001`, `λ=0.001`, mini-batch size 524,288, and seed 42. The best checkpoint was at epoch 6. Its full-validation Recall@20 and NDCG@20 are **0.175330** and **0.104337**; its one-time final test scores are **0.181018** and **0.168574**. See [`lightgcn_search.md`](lightgcn_search.md) for the completed search, candidate comparisons, seed sensitivity, and exact reproduction command.
