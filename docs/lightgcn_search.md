# Completed LightGCN architecture search

## Scope and selection

The search used the existing `data/processed/ml20m_lightgcn/` splits, the train-only graph, one uniformly sampled unobserved item per positive interaction, BPR, Adam, Xavier initialization, and the paper's LightGCN propagation. Only hyperparameters described in the paper were varied: propagation depth `K`, embedding dimension, layer-combination coefficients `α`, L2 coefficient `λ`, and learning rate. Mini-batch size stayed at 524,288 for comparability. No new propagation, loss, dropout, or negative-sampling method was introduced.

The staged search first compared depths 1–4 with the paper's uniform layer weights, then explored 32/64/128 embedding dimensions, the paper's regularization range, learning rates around 0.001, and fixed nonnegative layer weights summing to one. Promising configurations were rerun with **all 124,967 eligible validation users evaluated every epoch**. Full-validation Recall@20 was the selection metric, with NDCG@20 as secondary evidence. The test split was evaluated only after selecting the final checkpoint in this search. Exact configurations and full-validation metrics for **59 recorded runs** (including confirmation runs and the archived baseline) are in [`search_summary.json`](../artifacts/lightgcn_search/search_summary.json).

## Selected architecture and results

| Parameter | Selected value |
| --- | ---: |
| Propagation layers `K` | 2 |
| Embedding dimension | 32 |
| Layer weights `(α₀, α₁, α₂)` | `(0.975, 0.0125, 0.0125)` |
| Learning rate | 0.001 |
| L2 coefficient `λ` | 0.001 |
| Mini-batch size | 524,288 |
| Seed; selected epoch | 42; 6 |

| Evaluation | Users | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: |
| Full validation | 124,967 | 0.175330 | 0.104337 |
| Final test | 129,757 | 0.181018 | 0.168574 |

The checkpoint, ID maps, configuration, epoch history, and final metrics are under [`artifacts/lightgcn_ml20m/`](../artifacts/lightgcn_ml20m/). The initial K=3, 64-dimensional baseline is preserved under [`artifacts/lightgcn_search/baseline_k3_d64_l1e-4_lr1e-3/`](../artifacts/lightgcn_search/baseline_k3_d64_l1e-4_lr1e-3/). Its full-validation Recall@20 was 0.144253. The new selected checkpoint improves that measure by about 21.5% relative, on the same split and evaluation protocol.

## Evidence behind the choice

At 32 dimensions, `λ=0.001`, learning rate `0.001`, and `α₀=0.975` with the remaining weight spread evenly across propagated layers, the full-validation comparison was:

| Layers | Recall@20 | NDCG@20 |
| ---: | ---: | ---: |
| 1 | 0.169925 | 0.101754 |
| **2** | **0.175330** | 0.104337 |
| 3 | 0.173417 | 0.103393 |
| 4 | 0.175305 | **0.104394** |

K=2 had the highest Recall@20. K=4 was effectively tied, with slightly higher NDCG@20 but twice as many propagation steps; K=2 was selected for its simpler computation. At K=2 with the same other hyperparameters, 64 and 128 dimensions reached 0.155050 and 0.162198 Recall@20, both below 32 dimensions. At K=2 and 32 dimensions, `α₀=0.95`, `0.96`, `0.975`, and `0.98` yielded Recall@20 of 0.166557, 0.171481, 0.175330, and 0.172924, respectively. `λ=0` and `1e-4` yielded 0.164315 and 0.164667 at `α₀=0.95`, below `λ=1e-3` at 0.166557.

### Seed sensitivity

The two closest depths were repeated with seeds 42, 43, and 44, keeping every other hyperparameter identical. Recall@20 on full validation was:

| Layers | Seed 42 | Seed 43 | Seed 44 | Mean |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.175330 | 0.145344 | 0.145495 | 0.155390 |
| 4 | 0.175305 | 0.145459 | 0.145327 | 0.155364 |

These means are virtually identical. The selected seed-42 checkpoint is the **best observed checkpoint**, while its advantage over the other seeds is large; do not treat the 0.175330 validation score as a typical result guaranteed by this architecture. K=2 is a practical choice among the measured candidates, not a proof of a global optimum. The detailed seed runs remain in `artifacts/lightgcn_search/`.

## Reproduce and verify

From the repository root, with the processed data in place:

```bash
uv sync
uv run python -m lightgcn.train \
  --output-dir artifacts/lightgcn_reproduction \
  --layers 2 --embedding-dim 32 \
  --layer-weights 0.975,0.0125,0.0125 \
  --learning-rate 1e-3 --l2 1e-3 \
  --batch-size 524288 --max-epochs 12 \
  --eval-every 1 --patience 5 --validation-users 0 \
  --seed 42 --skip-test
```

After selecting a checkpoint by validation, evaluate it once on test:

```bash
uv run python -m lightgcn.evaluate_checkpoint \
  --checkpoint artifacts/lightgcn_reproduction/best.pt \
  --output artifacts/lightgcn_reproduction/final_metrics.json
```

The training code records the data manifest's canonical SHA-256 hashes in each run's `config.json` and checkpoint. MPS results may vary slightly across PyTorch or device versions. The search used PyTorch 2.14.0 on Apple MPS.
