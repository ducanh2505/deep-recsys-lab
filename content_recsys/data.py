"""Audited inputs, movie templates, token caches and sparse user histories."""

from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from content_recsys.common import MODEL_ID, REVISION, emit, fingerprint, sha256, write_json, save_checkpoint
from multvae.data import Dataset, Split, load_dataset
from multvae.train import verify_checkpoint

LIMITS = {"single": 512, "plot": 384, "topic": 192, "people": 192}
BLOCK_WEIGHTS = (0.50, 0.25, 0.15, 0.10)
TOKENIZE_BATCH_SIZE = 256


def names(entries: list | None, limit: int, ordered: bool = False) -> str:
    """Select named metadata entries, optionally ordered by cast credit."""
    entries = entries or []
    if ordered:
        entries = sorted(entries, key=lambda x: x.get("order") if x.get("order") is not None else 10**9)
    return ", ".join(x["name"] for x in entries[:limit] if x.get("name"))


def movie_texts(row: dict) -> dict[str, str]:
    """Serialize one movie into the four deterministic English views."""
    trusted = row.get("tmdb_mapping_status") != "unverified"
    title = (row.get("title") if trusted else None) or row.get("movielens_title") or "Unknown title"
    genres = row.get("genres" if trusted else "genres_movielens") or []
    topic = "Genres: " + ", ".join(genres) if genres else ""
    if trusted and row.get("keywords"):
        topic += "\nKeywords: " + names(row["keywords"], 20)
    topic = "Title: " + title + ("\n" + topic if topic else "")
    people = []
    if trusted:
        for field, label, count in (("directors", "Directors", 5), ("writers", "Writers", 5), ("cast", "Cast", 10)):
            text = names(row.get(field), count, ordered=field == "cast")
            if text:
                people.append(f"{label}: {text}")
    context = []
    if trusted:
        for field, label in (("original_language", "Original language"), ("release_year", "Release year"),
                             ("runtime_minutes", "Runtime minutes")):
            if row.get(field) is not None:
                context.append(f"{label}: {row[field]}")
        countries = names(row.get("production_countries"), 1000)
        if countries:
            context.append("Production countries: " + countries)
    plot = []
    if trusted:
        for field, label in (("tagline", "Tagline"), ("overview", "Overview")):
            if row.get(field):
                plot.append(f"{label}: {row[field]}")
    views = {"plot": "\n".join(plot), "topic": topic, "people": "\n".join(people)}
    views["single"] = "\n".join(x for x in [topic, views["people"],
                                                        "\n".join(context), views["plot"]] if x)
    return views


def context_features(rows: list[dict]) -> tuple[np.ndarray, dict]:
    """Build train-catalog categorical features and missing-aware numeric features."""
    safe = [r if r.get("tmdb_mapping_status") != "unverified" else {} for r in rows]
    languages = sorted({r["original_language"] for r in safe if r.get("original_language")})
    countries = sorted({f'{c.get("source")}:{c.get("id") or c["name"]}' for r in safe
                        for c in (r.get("production_countries") or [])})
    language_ids = {name: i for i, name in enumerate(languages)}
    country_ids = {name: i + len(languages) for i, name in enumerate(countries)}
    categorical_size = len(languages) + len(countries)
    values = np.zeros((len(rows), categorical_size + 4), dtype=np.float32)
    stats = {}
    for offset, field in enumerate(("release_year", "runtime_minutes")):
        observed = np.array([r[field] for r in safe if r.get(field) is not None], dtype=np.float64)
        mean = float(observed.mean()) if len(observed) else 0.0
        std = float(observed.std()) if len(observed) else 1.0
        std = max(std, 1e-8)
        stats[field] = {"mean": mean, "std": std}
        for i, row in enumerate(tqdm(safe, desc=f"Context: {field}", unit="movie", dynamic_ncols=True)):
            if row.get(field) is None:
                values[i, categorical_size + 2 + offset] = 1
            else:
                values[i, categorical_size + offset] = np.clip((row[field] - mean) / std, -3, 3)
    present = []
    for i, row in enumerate(tqdm(safe, desc="Context: categories", unit="movie", dynamic_ncols=True)):
        if row.get("original_language"):
            values[i, language_ids[row["original_language"]]] = 1
        for country in row.get("production_countries") or []:
            key = f'{country.get("source")}:{country.get("id") or country["name"]}'
            values[i, country_ids[key]] = 1
        present.append(any(row.get(key) is not None for key in
                           ("release_year", "runtime_minutes", "original_language"))
                       or bool(row.get("production_countries")))
    values[~np.array(present)] = 0
    return values, {"languages": languages, "countries": countries, "numeric": stats,
                    "present": present, "dimension": values.shape[1]}


def prepare(data_dir: Path, catalog: Path, baseline: Path, output: Path) -> dict:
    """Audit the original splits and save ID-aligned reusable preparation artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def progress(stage: str, **values: object) -> None:
        emit(output, "content_preparation", stage=stage,
             elapsed_seconds=time.monotonic() - started, **values)

    progress("hash_inputs")
    source = {"catalog": sha256(catalog), "baseline": sha256(baseline),
              "splits": {name: sha256(data_dir / f"{name}.parquet") for name in ("train", "valid", "test")},
              "split_manifest": sha256(data_dir / "manifest.json"), "model": MODEL_ID,
              "revision": REVISION, "limits": LIMITS, "block_weights": BLOCK_WEIGHTS, "format": 2}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        progress("verify_cached_inputs")
        existing = json.loads(manifest_path.read_text())
        if existing["fingerprint"] != fingerprint(source):
            raise ValueError("Prepared inputs differ; choose a new output directory")
        load_prepared(output)
        progress("complete", cached=True)
        return existing
    progress("audit_splits")
    data = load_dataset(data_dir)
    progress("verify_baseline", users=data.n_users, movies=data.n_items)
    checkpoint = torch.load(baseline, weights_only=True, map_location="cpu")
    verify_checkpoint(checkpoint, data)
    progress("load_catalog")
    frame = pl.read_parquet(catalog)
    if frame["movieId"].n_unique() != frame.height or set(frame["movieId"].to_list()) != set(data.item_ids):
        raise ValueError("Catalog IDs must exactly match the unique checkpoint item set")
    by_id = {row["movieId"]: row for row in frame.iter_rows(named=True)}
    rows = [by_id[int(item)] for item in data.item_ids]
    progress("build_context", movies=len(rows))
    contexts, metadata = context_features(rows)
    progress("load_tokenizer", model=MODEL_ID, revision=REVISION)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, padding_side="left")
    progress("serialize_movie_texts", movies=len(rows))
    views = [movie_texts(row) for row in tqdm(rows, desc="Movie texts", unit="movie", dynamic_ncols=True)]
    tokens, truncation = {}, {}
    for view, limit in LIMITS.items():
        progress("tokenize", view=view, completed=0, total=len(views))
        sequences = []
        with tqdm(total=len(views), desc=f"Tokenize {view}", unit="movie", dynamic_ncols=True) as bar:
            for start in range(0, len(views), TOKENIZE_BATCH_SIZE):
                batch = views[start:start + TOKENIZE_BATCH_SIZE]
                sequences.extend(tokenizer([texts[view] for texts in batch],
                                           truncation=False, padding=False)["input_ids"])
                bar.update(len(batch))
        sequences = [ids if texts[view] else [] for ids, texts in zip(sequences, views)]
        lengths = np.array([len(ids) for ids in sequences])
        tokens[view] = [torch.tensor(ids[:limit], dtype=torch.long) for ids in sequences]
        truncation[view] = {"limit": limit, "truncated": int((lengths > limit).sum()),
                            "fraction": float((lengths > limit).mean()), "missing": int((lengths == 0).sum())}
        progress("tokenize", view=view, completed=len(views), total=len(views),
                 percent=100.0, **truncation[view])
    progress("save_tokens_and_context")
    save_checkpoint(output / "tokens.pt", tokens)
    np.save(output / "context.npy", contexts)
    write_json(output / "context.json", metadata)
    bundle = {"user_ids": torch.from_numpy(data.user_ids), "item_ids": torch.from_numpy(data.item_ids),
              "manifest": data.manifest, "audit": data.audit, "baseline": str(baseline.resolve())}
    for name in ("train", "valid", "test"):
        progress("build_history", split=name)
        split = getattr(data, name)
        ratings = pl.read_parquet(data_dir / f"{name}.parquet").sort(["userId", "movieId"])["rating"].to_numpy()
        bundle[name] = {"items": torch.from_numpy(split.items), "offsets": torch.from_numpy(split.offsets),
                        "keys": torch.from_numpy(split.keys),
                        "rating_weights": torch.from_numpy((ratings - 3).astype(np.float32))}
        progress("build_history", split=name, completed=len(split.items), total=len(split.items))
    progress("save_history_bundle")
    save_checkpoint(output / "data.pt", bundle)
    progress("hash_prepared_artifacts")
    manifest = {"fingerprint": fingerprint(source), "sources": source, "data_audit": data.audit,
                "baseline": str(baseline.resolve()), "truncation": truncation,
                "artifacts": {name: sha256(output / name) for name in
                              ("tokens.pt", "context.npy", "context.json", "data.pt")}}
    write_json(manifest_path, manifest)
    progress("complete", cached=False, movies=len(rows))
    return manifest


@lru_cache(maxsize=1)
def load_prepared(directory: Path) -> tuple[Dataset, dict, dict, dict]:
    """Verify prepared artifacts and reconstruct the existing dataset interface."""
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Prepared artifact fingerprint mismatch: {name}")
    bundle = torch.load(directory / "data.pt", weights_only=True, map_location="cpu")
    splits, weights = {}, {}
    for name in ("train", "valid", "test"):
        value = bundle[name]
        splits[name] = Split(*(value[key].numpy() for key in ("items", "offsets", "keys")))
        weights[name] = value["rating_weights"].numpy()
    data = Dataset(bundle["user_ids"].numpy(), bundle["item_ids"].numpy(), splits["train"],
                   splits["valid"], splits["test"], bundle["manifest"], bundle["audit"])
    tokens = torch.load(directory / "tokens.pt", weights_only=True, map_location="cpu")
    return data, weights, tokens, manifest


def profiles(data: Dataset, embedding: torch.Tensor, weights: dict, profile: str,
             include_valid: bool = False, normalize: bool = True) -> torch.Tensor:
    """Aggregate all user histories with CPU CSR multiplication, never truncating them."""
    result = torch.zeros((data.n_users, embedding.shape[1]), dtype=torch.float32)
    for name in (("train", "valid") if include_valid else ("train",)):
        split = getattr(data, name)
        values = weights[name] if profile == "rating" else np.ones(len(split.items), dtype=np.float32)
        matrix = torch.sparse_csr_tensor(torch.from_numpy(split.offsets), torch.from_numpy(split.items.astype(np.int64)),
                                         torch.from_numpy(values), size=(data.n_users, data.n_items))
        result.add_(torch.sparse.mm(matrix, embedding.cpu()))
    return F.normalize(result, dim=1) if normalize else result
