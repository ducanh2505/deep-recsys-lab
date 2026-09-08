# Data contract

An interaction contains two required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `user_id` | strict string or integer | Stable external user identifier |
| `item_id` | strict string or integer | Stable external catalog identifier |
| `value` | finite number, optional | Interaction weight; defaults to `1.0` |
| `timestamp` | timestamp, optional | Event time required by temporal splitting |

Boolean identifiers are rejected even though Python treats booleans as integers. Empty
string identifiers, non-finite values, and unparseable configured timestamps are rejected.

## Generic tables

The tabular adapter reads CSV or Parquet. Column mapping is configured explicitly:

```yaml
dataset:
  name: tabular
  path: var/data/raw/interactions.parquet
  format: parquet
  columns:
    user_id: account
    item_id: product
    value: strength
    timestamp: occurred_at
```

Additional event columns are retained. One catalog metadata record per item is carried into
the model artifact and returned by the serving response.

## Mapping and splits

String and integer IDs retain their types. Mappings are sorted deterministically and are
stored as typed JSON values. Duplicate event rows are retained; their values sum when a
sparse matrix is materialized.

Two split strategies are available:

- `user_holdout` uses a seeded per-user permutation.
- `temporal` orders each user's events and holds out the most recent rows. It fails when
  any event lacks a timestamp.

Users with too few events to produce train, validation, and test rows remain entirely in
training. Split ratios and seed are part of the dataset digest.

## Prepared payload

A prepared dataset contains raw normalized event rows in Parquet, catalog metadata,
typed mappings, split-row indices, sparse NPZ matrices, and a checksum manifest. Loading
verifies all payloads before returning data to a trainer. Sequential trainers consume only
the event rows assigned to the training split.
