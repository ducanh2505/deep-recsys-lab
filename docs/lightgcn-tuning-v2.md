# LightGCN tuning v2

## Mục tiêu và protocol

Run này tìm cấu hình tốt nhất của vanilla LightGCN trên MovieLens-20M bằng validation, sau đó retrain trên train + validation và chỉ mở test một lần ở bước cuối.

- Dataset split: per-user random 80/10/10, seed `98765`.
- Positive interaction: rating `>= 4.0`, tối thiểu 5 interaction/user.
- Device: CPU, FP32, 8 torch threads, batch size `262144`.
- Selection criterion: `Recall@20`, sau đó `NDCG@20`.

## Search space

Tổng cộng `3 x 4 x 3 x 3 = 108` cấu hình:

| Nhóm | Giá trị |
| --- | --- |
| Embedding dimension | `32`, `64`, `128` |
| Number of propagation layers | `1`, `2`, `3`, `4` |
| Learning rate | `0.0005`, `0.001`, `0.002` |
| L2 regularization | `0.0001`, `0.001`, `0.01` |

Successive-halving dùng 50 bước cho toàn bộ 108 cấu hình, giữ top-6 chạy đến 300 bước. Winner được train đến tối đa 4000 bước với validation mỗi 50 bước, minimum 500 bước và patience 10. Best step được dùng cho final retrain trên train + validation.

## Kết quả

Cấu hình tối ưu trong search space:

```yaml
embedding_dim: 64
layers: 1
learning_rate: 0.002
l2: 0.001
best_step: 4000
```

| Metric | Baseline | Tuning v2 | Thay đổi tương đối |
| --- | ---: | ---: | ---: |
| Recall@20 | 0.2192882 | 0.2738946 | +24.90% |
| NDCG@20 | 0.1440564 | 0.1838455 | +27.62% |

Validation của winner ở bước 4000 là Recall@20 `0.2650920` và NDCG@20 `0.1710086`. Run hoàn tất với trạng thái `verified`; test access được ghi nhận sau model selection.

## Reproduce

```bash
deep-recsys lightgcn prepare
deep-recsys lightgcn reproduce --config-name lightgcn_tuning
deep-recsys lightgcn report --config-name lightgcn_tuning
```

Config: [`configs/lightgcn_tuning.yaml`](../configs/lightgcn_tuning.yaml). Report: [`public/reports/lightgcn-v2.json`](../public/reports/lightgcn-v2.json).

Kết quả MovieLens-20M này là kết quả local trong protocol trên, không phải con số benchmark trực tiếp của paper LightGCN (paper dùng Gowalla, Yelp2018 và Amazon-Book).
