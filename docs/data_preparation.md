# MovieLens 20M data preparation for LightGCN

This document describes the current processing flow in [`scripts/prepare_ml20m.py`](../scripts/prepare_ml20m.py). The split and 10-core rules are project-specific choices; they are not rules defined in Section 4.1 of the LightGCN paper.

## Input and validation

The script reads `data/raw/ml-20m/ratings.csv` with four columns: `userId` (Int64), `movieId` (Int64), `rating` (Float64), and `timestamp` (Int64). Paths are resolved from the project root, so the script can be invoked from another working directory.

Before processing, the script requires the input file to exist, contain exactly **20,000,263** rows, have no null values, and have no duplicate `(userId, movieId)` pairs. It raises an error if any check fails. Polars reads the CSV with a lazy scan and collects it using the streaming engine; subsequent steps operate on in-memory DataFrames.

## Processing steps

1. **Select positive interactions:** Keep rows where `rating >= 4.0`. The original rating and timestamp remain in the output.
2. **Build the 10-core:** Count interactions per user and per item in the positive set. On each pass, remove rows whose user has fewer than 10 interactions **or** whose item has fewer than 10 interactions. Repeat until the row count stops changing. The script raises an error if no users remain.
3. **Split per user:** Sort the 10-core by `userId` and `movieId`, then shuffle each user's interactions with a NumPy random generator seeded with `42`. For a user with `n` interactions:

   - `n_test = floor(0.20 × n)`
   - `n_valid = floor(0.10 × (n - n_test))`
   - `n_train = n - n_test - n_valid`

   The first shuffled rows go to test, the next rows to validation, and the rest to train. The split does **not** use timestamp order. Because counts are rounded down for each user, the overall fractions can differ from 20% test and 8% validation.
4. **Remove cold-start items from evaluation:** Keep only validation and test rows whose `movieId` appears in train. Train is unchanged. The script does not filter users again after this step or remap `userId` and `movieId` to contiguous IDs; both retain their original values.

## Outputs

The script writes these files to `data/processed/ml20m_lightgcn/`:

| File | Contents |
| --- | --- |
| `interactions.parquet` | All interactions after the 10-core step, before splitting. |
| `train.parquet` | Training interactions. |
| `valid.parquet` | Validation interactions after cold-start item removal. |
| `test.parquet` | Test interactions after cold-start item removal. |
| `manifest.json` | Counts, split formulas, seed, diagnostics, and hashes for checking the results. |

The Parquet files retain the four input columns. They are sorted by `userId` and `movieId`, compressed with Zstandard, and written through temporary files before replacing the destinations. `manifest.json` is also written through a temporary file. The SHA-256 hashes in the manifest are computed from each split sorted by the two IDs. Each row is serialized as `userId,movieId,rating,timestamp\n`, with the rating formatted to one decimal place. These hashes check the **canonicalized content**, not the Parquet file bytes.

The manifest also records how many validation and test rows were removed, along with the number of distinct items in those removed rows. Its zero-evaluation-user diagnostics distinguish users who already had zero rows because of rounding from users who lost all their rows through cold-start item removal.

## Current dataset counts

The existing [`manifest.json`](../data/processed/ml20m_lightgcn/manifest.json) reports:

| Stage | Interactions |
| --- | ---: |
| Input CSV | 20,000,263 |
| Rating ≥ 4.0 | 9,995,410 |
| After 10-core | 9,911,879 |
| Train | 7,238,638 |
| Validation | 741,976 |
| Test | 1,931,265 |

The 10-core contains **129,757 users** and **11,508 items**. The interaction counts across filtering passes are `9,995,410 → 9,912,315 → 9,911,897 → 9,911,879 → 9,911,879`; the final repeated count confirms convergence. This run removed no validation or test rows for cold-start items. It has 4,790 users with no validation rows because of rounding.

Note: The current manifest also contains `archive_md5`, `archive_sha256`, and `source_urls`. The current version of `prepare_ml20m.py` does **not** generate these fields, so rerunning the script will overwrite the manifest without them.

## Rerunning the script

From the project root, after placing `ratings.csv` at the expected input path:

```bash
uv run python scripts/prepare_ml20m.py
```

Python dependencies are declared in [`pyproject.toml`](../pyproject.toml). Rerunning the script replaces the output files and prints key statistics to the terminal.
