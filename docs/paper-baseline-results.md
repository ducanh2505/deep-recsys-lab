# MovieLens 20M paper-style validation baseline

This is the first full-data **validation** result for spec #42 / ticket #47. The paper benchmark is separate from the chronological lifecycle. The Known-User and History-Only test cohorts remain sealed; no test score is used here. This is one deterministic seed (42), with no uncertainty interval or quality gate.

## Reproduce and provenance

The benchmark uses the official MovieLens 20M archive (`ml-20m.zip`, published MD5 `cd245b17a1ae2cc31bb14903e1204af3`, SHA-256 `96f243c338a8665f6bcc89c53edf6ee39162a846940de6b7c8c48aeada765ff3`), its locally prepared chronological Parquet (SHA-256 `96bd65571ab2871cbba61448b092b02cefa5d7989caa28ca82bb445559a3af1a`), and all 20,000,263 ratings. Positive Interaction means rating >= 4.0; 9,995,410 source ratings qualify. Both runs use seed 42 and per-source pool depth 200, with final LHF Top-100. Mult-VAE uses one epoch / batch 256; Known-User LightGCN uses one epoch / batch 65,536. These are the historical full-profile training budgets, explicitly recorded as the new baseline before tuning.

The exact Mult-VAE configuration is hidden 32, latent 16, dropout 0, learning rate .01, KL weight .2, one epoch, batch 256, CPU. Known-User LightGCN uses embedding 16, two propagation layers, one uniform negative per positive, learning rate .01, regularization .0001, one epoch, batch 65,536, CPU. LHF caps inner training negatives at 20 per Query. Full serialized configurations and fitted-model metadata are in the local reports.

The run host is an Apple M4 with 24 GiB unified memory, macOS 26.6.2, Python 3.12.13. Dependency versions were DuckDB 1.5.5, LightGBM 4.7.0, NumPy 2.5.3, PyArrow 22.0.0, SciPy 1.18.1, and PyTorch 2.14.0. Exact training and validation split membership, validation Gold Sets, catalog rules, cold exclusions, aggregate source-pool diagnostics, per-Subject validation scores, model provenance, timings, and segmentation are in the ignored local JSON reports. History-Only test Subject IDs and a split hash, plus a Known-User test membership hash, are recorded. History-Only test fold-in histories, both modes’ test Gold Sets, and test metrics remain sealed. Raw per-Query Candidate pool IDs are omitted by default. The archive, Parquet source, and full generated reports are not committed.

```bash
uv sync --dev
PAPER_CACHE=/absolute/path/to/the/existing/var/datasets
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --no-sync movie-recsys history-only-benchmark \
  --cache "$PAPER_CACHE" --seed 42 \
  --output artifacts/paper-history-only/report.json
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --no-sync movie-recsys known-user-benchmark \
  --cache "$PAPER_CACHE" --seed 42 \
  --output artifacts/paper-known-user/report.json
```

The runs were executed sequentially from clean revision `1ea6d45` with `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, and `MKL_NUM_THREADS=4`, using the existing Python 3.12.13 virtual environment and this worktree as `PYTHONPATH`. The checkout-local equivalent above gives the same settings. The runs are sequential to stay within host memory. They do not start Kafka. History-Only defaults to disjoint 10,000 validation and 10,000 sealed test Subjects. No `--evaluate-test` flag was used.

| Protocol | Code revision used for run | Training Subjects / interactions | Validation Queries / eligible Gold Movies | Catalog Movies | Exclusions |
| --- | --- | ---: | ---: | ---: | --- |
| Known-User, LightGCN/NGCF 10-core | `1ea6d459e9e06df7046c84e7b909862373e45d4b` (clean) | 129,757 / 7,032,784 | 129,757 / 844,758 | 11,508 | 8,530 Subjects and 83,531 Positive Interactions removed by 10-core; 0 cold validation Gold Movies |
| History-Only, Mult-VAE disjoint Subjects | `1ea6d459e9e06df7046c84e7b909862373e45d4b` (clean) | 116,677 / 8,531,332 | 10,000 / 148,839 | 20,268 | 1,610 Subjects under five positives; 4,728 positive interactions with them; 222 cold fold-in Movies, 47 cold held-out Gold Movies |

Known-User applies iterative 10-core filtering, per-Subject 80/20 train/test, then 10% validation from the training portion; all eligible Subjects are included. History-Only retains Subjects with at least five positives, assigns disjoint 10,000-Subject validation and test cohorts, and splits each held-out Subject's positives into 80% fold-in / 20% Gold Set. Training-visible catalog membership and history exclusions are applied before scoring. Evaluation uses no sampled negatives or synthetic Gold Movie insertion.

## Paper-style validation quality

Each row is a real ranking's macro per-Subject score. Known-User Recall divides by the complete eligible Gold Set; History-Only Recall divides by `min(K, eligible Gold Set size)`. NDCG uses binary relevance. The final LHF order is the baseline result; source rows explain retrieval and ranking contributions.

| Mode / order | Recall@10 | Recall@20 | Recall@50 | Recall@100 | NDCG@10 | NDCG@20 | NDCG@50 | NDCG@100 | Query Coverage@100 | Catalog Coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Known-User, Popularity | .0876 | .1423 | .2310 | .3273 | .0753 | .0931 | .1207 | .1466 | .7306 | .0248 |
| Known-User, ItemKNN | .1653 | .2410 | .3670 | .4848 | .1391 | .1636 | .2025 | .2344 | .8595 | .3427 |
| Known-User, Mult-VAE | .1658 | .2563 | .4068 | .5341 | .1332 | .1635 | .2102 | .2451 | .8775 | .3064 |
| Known-User, LightGCN | .0073 | .0116 | .0208 | .0325 | .0066 | .0079 | .0107 | .0138 | .1450 | .9559 |
| **Known-User, final LHF** | **.1726** | **.2628** | **.4119** | **.5357** | **.1393** | **.1692** | **.2154** | **.2494** | **.8807** | **.2802** |
| History-Only, Popularity | .1374 | .1678 | .2405 | .3351 | .1382 | .1466 | .1717 | .2039 | .8715 | .0134 |
| History-Only, ItemKNN | .2474 | .2823 | .3847 | .4955 | .2509 | .2568 | .2908 | .3291 | .9416 | .1157 |
| History-Only, Mult-VAE | .2422 | .2947 | .4220 | .5415 | .2384 | .2544 | .2995 | .3418 | .9451 | .1287 |
| **History-Only, final LHF** | **.2550** | **.3022** | **.4232** | **.5401** | **.2508** | **.2632** | **.3053** | **.3465** | **.9455** | **.1208** |

The Oracle Union is the unbounded deduplicated union of the configured source pools. It is a diagnostic set without a served order, so Top-K NDCG does not apply. Gold occurrences recovered by the union use a micro denominator; Known-User's recorded Oracle Union Recall is a separate macro per-Subject measure. Neither is directly comparable to the ordered final Top-100 at a different pool budget.

| Mode | Gold occurrences retrieved / eligible (micro) | Queries with a union Gold Movie | Union Catalog Coverage | Macro Oracle Union Recall |
| --- | ---: | ---: | ---: | ---: |
| Known-User | 549,505 / 844,758 (.6505) | 123,122 / 129,757 (.9489) | .9921 | .7239 |
| History-Only | 96,757 / 148,839 (.6501) | 9,803 / 10,000 (.9803) | — | — |

The History-Only union contains 3,246,524 Candidate occurrences across 10,000 Queries. The Known-User report does not record this occurrence count. Catalog coverage describes distinct retrieved Movies relative to the training catalog; History-Only records it for source and final Top-100 rankings, not for the unbounded union.

| Mode / source | Gold occurrences in full 200-item pool | Exclusive union Gold occurrences | Queries with an exclusive union Gold Movie |
| --- | ---: | ---: | ---: |
| Known-User, Popularity | 338,251 | 12,603 | 10,324 |
| Known-User, ItemKNN | 447,224 | 21,809 | 17,123 |
| Known-User, Mult-VAE | 481,041 | 72,584 | 43,542 |
| Known-User, LightGCN | 39,108 | 6,438 | 5,814 |
| History-Only, Popularity | 60,844 | 2,387 | 1,532 |
| History-Only, ItemKNN | 80,386 | 3,732 | 2,401 |
| History-Only, Mult-VAE | 86,201 | 12,699 | 4,818 |

Exclusive contribution means the Gold Movie is absent from every other source pool for the same Query. LightGCN has broad Catalog Coverage@100 (.9559) but low standalone Recall@100 (.0325); it still contributes 6,438 union Gold occurrences exclusively.

## Gold-loss diagnosis

The categories partition **raw held-out Subject–Movie occurrences**, including exclusions. Query counts may overlap because a Query can contain Gold Movies in several categories. Cold Movies are excluded before the paper Recall denominator. Under the current train-positive catalog rule, a catalog Movie without a training Positive Interaction is impossible and remains an explicit zero count. The present LHF sees the full diagnostic union, so pre-fusion truncation is also zero.

| Category | Known-User Gold occurrences / Queries | History-Only Gold occurrences / Queries |
| --- | ---: | ---: |
| Outside training catalog, excluded | 0 / 0 | 47 / 28 |
| In catalog with no training Positive Interaction | 0 / 0 | 0 / 0 |
| Outside every source pool | **295,253 / 80,512** | **52,082 / 7,504** |
| In source union but dropped before fusion | 0 / 0 | 0 / 0 |
| In fusion input but below final Top-100 | 182,121 / 72,547 | 30,640 / 7,096 |
| Recovered in final Top-100 | 367,384 / 114,274 | 66,117 / 9,455 |

**Largest addressable Known-User failure:** 295,253 eligible Gold occurrences (35.0% of 844,758) are absent from all four 200-item source pools. A further 182,121 union Gold occurrences fall below the final Top-100. Known-User has no validation Gold Movies excluded as cold.

**Largest addressable History-Only failure:** 52,082 eligible Gold occurrences (35.0% of 148,839) are absent from all three 200-item source pools. Source retrieval is the first optimization target for this mode. The 30,640 Gold occurrences found by sources but ranked below final Top-100 are the next measured opportunity. This diagnosis is from the new paper validation protocol only.

The complete JSON report segments every category by fixed Query history lengths (`0`, `1–4`, `5–19`, `20–99`, `100–499`, `500+`) and training Movie popularity (`0`, `1–9`, `10–99`, `100–999`, `1000+`). Known-User misses 153,214 Gold occurrences outside all source pools in the 100–499 history segment and 99,181 in 20–99. By training Movie popularity, 148,136 absent Gold occurrences are in the 1000+ bucket, 122,262 in 100–999, and 1,028 of 1,048 in 1–9. History-Only's 47 cold Gold Movies appear in popularity bucket `0`; its largest lost pools by history are 29,009 Gold occurrences for histories of 100–499 Movies and 16,452 for 20–99 Movies. In the 1000+ training-popularity bucket, 28,971 Gold occurrences are outside all source pools; 404 of 408 Gold occurrences in the 1–9 bucket are absent from all sources. Segment counts use raw Gold occurrences; segment Recall uses only eligible Gold Sets.

## Runtime and inference

Inference is warm, in-process, one Query at a time, from source-pool generation through final LHF ranking. The first five scored Queries are excluded from latency percentiles but included in quality metrics. It excludes source loading, model fitting, and network time. No runtime or latency result gates quality selection.

| Mode | Split | Inner fit + fusion rows | LHF fit | Outer retriever fit | Validation scoring | Total wall | Peak RSS | Warm p50 / p95 / p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Known-User | 80.7 s | 1,370.8 s | 11.4 s | 97.0 s | 1,668.0 s | 3,235.5 s | 10.56 GB | 10.56 / 11.98 / 13.00 ms |
| History-Only | 51.8 s | 185.8 s | 1.1 s | 42.6 s | 152.6 s | 454.0 s | 7.22 GB | 12.64 / 14.35 / 25.36 ms |

Known-User Mult-VAE trained on all 7,032,784 outer training Positive Interactions for 4.08 s on CPU. LightGCN trained on the same 7,032,784 graph edges for 13.47 s on CPU. Known-User LHF used 3,079,889 inner training rows from 129,757 inner Queries. Its complete run took 3,242.82 s by `/usr/bin/time -l`, with zero swaps.

The measured History-Only Mult-VAE saw all 8,531,332 training Positive Interactions, trained on CPU for 6.08 s, and the History-Only LHF was trained from inner pseudo-held-out training Subjects. `/usr/bin/time -l` recorded 456.42 s real and zero swaps for this run. Its final metrics, all source and union metrics, gold-loss counts and segments, split membership, and all per-Subject validation metrics exactly match the archived pre-ItemKNN-optimization run; the faster code changed no measured ranking.

## Separate chronological lifecycle evidence

The historical **90→100% chronological Future Window** uses one first-future Gold Candidate per Query, a 5,000-Subject cohort cap, Coverage@200, and Top-10 metrics. It is a different task with a different denominator and must not be used as the paper Recall@100 baseline. Source: `docs/full-profile-results.md`.

| Temporal mode and served order | Queries | Coverage@200 | EndToEndRecall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Known-User, LHF | 4,908 | .2412 | .0371 | .0186 |
| History-Only, LHF | 4,908 | .2414 | .0379 | .0194 |

The new paper benchmark makes no chronological deployment claim. Test outcomes remain sealed for later validation-selected configurations and paired Subject-level confidence intervals.
