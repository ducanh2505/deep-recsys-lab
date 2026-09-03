# Multi-VAE trên Yelp2018

Workflow này retrain Multi-VAE trên các file đã xử lý chính thức của
[LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data/yelp2018).

## Chạy lại

```bash
deep-recsys yelp2018 prepare
deep-recsys multvae reproduce --config-name multvae_yelp2018
deep-recsys multvae report --run-dir outputs/multvae-yelp2018-v1
```

`yelp2018 prepare` tải `user_list.txt`, `item_list.txt`, `train.txt` và
`test.txt`, kiểm tra duplicate/leakage và lưu checksum. Official train/test
được giữ nguyên; validation là một carve-out deterministic 10% từ official
train, có seed 2020 và không làm mất item khỏi catalog train.

Sweep gồm 32 cấu hình Multi-VAE: hai kiến trúc, hai mức dropout, hai beta cap,
hai learning rate và hai weight decay. Successive halving chạy 20 epoch cho
toàn bộ cấu hình, 60 epoch cho top 8 và tối đa 200 epoch cho top 2 với
early stopping. Chọn theo validation Recall@20 rồi NDCG@20. Cấu hình thắng
được khởi tạo lại và retrain trên train + validation; official test chỉ được
đánh giá một lần sau model selection.

Aggregate report được ghi ở
`public/reports/multvae-yelp2018.json`, còn loss/NLL/KL/beta và validation
curve được lưu trong run directory dưới dạng JSONL. HTML report là
`multvae-yelp2018-results.html`.
