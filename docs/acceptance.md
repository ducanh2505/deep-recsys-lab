# Acceptance record

This record contains only results verified from local artifacts. The complete
test-fold evaluation below is a comparison with the reported paper values, not
a claim of exact paper reproduction.

## Complete MovieLens test-fold evaluation

`outputs/multvae-20260818-224228/best.pt` was evaluated with no batch bound on
all 10,000 test users (20 batches of 500), with fold-in items masked and all
137,785 fold-out interactions used as ground truth. The test-fold file hashes
match `data/processed/manifest.json`. The resulting metrics are saved in
`outputs/multvae-20260818-224228/test_metrics.json`.

| Metric | Verified test result | Paper target | Difference |
| --- | ---: | ---: | ---: |
| Recall@20 | `0.394377` | `0.395` | `-0.000623` |
| Recall@50 | `0.538899` | `0.537` | `+0.001899` |
| NDCG@100 | `0.419612` | approximately `0.426` | approximately `-0.006388` |

Recall@50 is above the reported target. Recall@20 and NDCG@100 are below their
reported targets; the NDCG comparison is approximate because its paper target
is reported approximately.

## Other completed local checks

- GroupLens `ml-20m.zip` was downloaded and verified against its MD5 sidecar.
- Complete preprocessing produced 116,677 train users, 10,000 validation
  users, 10,000 test users, and the expected 20,108-item catalog.
- The synthetic workflow passed `smoke acceptance: PASS` through preprocessing,
  bounded training, FP32 ONNX export and parity validation, immutable Bento
  Model Store registration, Service startup, and API prediction.
- Model Store tests cover duplicate tags, schema-v3 checksums and runtime
  metadata, dynamic batches, PyTorch parity, schema-v2 migration errors, mapping
  order, incompatible checkpoints, malformed graphs, non-finite inference,
  missing models, and corrupt-model startup failure.
- Bento ASGI contract tests cover deterministic unseen-item ranking, metadata,
  strict input limits, domain errors, API-key isolation, 64 KiB requests,
  request IDs, health probes, and generated OpenAPI documentation.
- Vectorized inference tests prove that different histories and `top_k` values
  match sequential ranking exactly, one dense batch reaches one ONNX call, and
  unknown-item/catalog-exhausted failures cannot poison valid batch members.
  Empty and all-invalid batches skip ONNX Runtime entirely.
- Service-graph tests verify the separate `deep_recsys_inference` dependency,
  one worker, metrics, concurrency eight, maximum batch size eight, and 10 ms
  batching latency budget.
- CI builds and containerizes the versioned Bento, then verifies `/readyz`,
  `/recommend`, `/model_info`, `/metrics`, and the absence of Torch on port 3000.
- CI gates a repeated same-process ONNX-versus-PyTorch p50 comparison and
  requires the adaptive-batch histogram sum to exceed its count. HTTP p50/p95
  latency and throughput remain non-gating CI artifacts.
- The local verification suite passes 42 tests with 87.58% core-package
  coverage, strict mypy, Ruff, source and wheel builds, synthetic smoke, and a
  live Torch-free Bento OCI container check.

## Controlled ONNX model-forward measurement

The schema-v3 export of `outputs/multvae-20260818-224228/best.pt` contains one
embedded-weight, opset-20 `model.onnx` file of 97,750,444 bytes for the
20,108-item catalog. Registration passed finite-output checks, numerical parity
with `rtol=1e-4` and `atol=2e-5`, and exact unseen top-k ordering.

The batch-size-one benchmark used 25 warm-up calls and five alternating rounds
of 200 measured calls in the same Python process on the local Apple Silicon CPU.

| Backend | p50 | p95 | Throughput |
| --- | ---: | ---: | ---: |
| PyTorch 2.13.0 | `1.466750 ms` | `1.628900 ms` | `666.021 inferences/s` |
| ONNX Runtime 1.29.0 | `1.390500 ms` | `1.454731 ms` | `714.016 inferences/s` |

ONNX Runtime passed the non-regression gate with a `1.054836x` p50 speedup. The
maximum absolute score difference was `0.00001621`. This is a controlled
model-forward measurement on one host, not a portable production capacity
claim.

## Non-gating Bento container measurements

The schema-v3 image built and started successfully with ONNX Runtime 1.29.0 and
no discoverable Torch package. `/model_info` reported the expected backend,
provider, opset, mapping hash, and 20,108-item catalog; `/recommend` returned
ranked unseen MovieLens metadata.

The adaptive-serving measurement used 14 dispatcher warm-up requests followed
by 80 attempts in synchronized waves of eight. It recorded p50
`29.704 ms`, p95 `49.034 ms`, and throughput `207.411 attempted requests/second`
for successful responses against the complete-catalog OCI image. BentoML's
adaptive-batch histogram recorded a sum of `72` over `35` ONNX calls, a mean
batch size of `2.057`; `72 > 35` passed the functional batching gate. Of the 80
exact attempts, 58 returned 200 and 22 were shed with 503 under the selected
hard 10 ms budget. HTTP status counts, latency, and throughput remain non-gating;
the separate post-load request returned the expected unseen recommendations.

For historical context, the pre-ONNX synthetic container recorded p50
`5.919 ms`, p95 `9.135 ms`, and `603.616 requests/second` under the same request
counts. It had only eight items, so the measurements are not comparable. Both
records are operational smoke results rather than production capacity claims;
CI retains HTTP measurements without a performance threshold.

## MPS note

The requested MPS command was exercised, but this runtime reports
`torch.backends.mps.is_available() == false`; the command therefore fails
closed with an actionable device error. The equivalent bounded CPU recipe was
run so the data/registration/API path was still verified. On an M4 runtime with
MPS available, use:

```bash
deep-recsys train --config-name config \
  data_dir=data/processed output_dir=outputs/movielens-mps-smoke device=mps \
  trainer.epochs=1 trainer.batch_size=500 \
  trainer.max_train_batches=10 trainer.max_eval_batches=2
```

The evaluated checkpoint above is from the completed 200-epoch run configured
with `batch_size=500`, `total_anneal_steps=200000`, and no train or evaluation
batch bounds.
