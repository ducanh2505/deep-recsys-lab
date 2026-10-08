#!/usr/bin/env python3
"""Enrich the training catalog with resumable TMDB and Wikimedia metadata."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import random
import re
import shlex
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen
import zipfile

import polars as pl


ROOT = Path(__file__).resolve().parent.parent
LANGUAGE = "en-US"
USER_AGENT = "MovieLensContentMetadata/1.0 (local non-commercial research)"
PERSON = pl.Struct({"id": pl.String, "name": pl.String, "source": pl.String,
                    "order": pl.Int64})
TERM = pl.Struct({"id": pl.String, "name": pl.String, "source": pl.String})
CATALOG_SCHEMA = {
    "movieId": pl.Int64, "imdb_id": pl.String, "tmdb_id": pl.Int64,
    "wikidata_id": pl.String, "movielens_title": pl.String, "title": pl.String,
    "original_title": pl.String, "release_date": pl.String, "release_year": pl.Int64,
    "genres_movielens": pl.List(pl.String), "genres_tmdb": pl.List(pl.String),
    "genres": pl.List(pl.String), "tmdb_overview": pl.String,
    "wikipedia_plot": pl.String, "wikipedia_intro": pl.String,
    "overview": pl.String, "overview_source": pl.String, "text_language": pl.String,
    "tagline": pl.String, "keywords": pl.List(TERM), "cast": pl.List(PERSON),
    "directors": pl.List(PERSON), "writers": pl.List(PERSON),
    "runtime_minutes": pl.Int64, "original_language": pl.String,
    "production_countries": pl.List(TERM), "spoken_languages": pl.List(TERM),
    "collection_id": pl.Int64, "collection_name": pl.String,
    "tmdb_mapping_status": pl.String, "wikipedia_url": pl.String,
    "wikipedia_section": pl.String,
}
PROVENANCE_SCHEMA = {
    "movieId": pl.Int64, "field": pl.String, "source": pl.String,
    "external_id": pl.String, "language": pl.String, "fetched_at": pl.String,
    "raw_path": pl.String, "source_url": pl.String, "revision_id": pl.String,
}
STATUS_SCHEMA = {
    "movieId": pl.Int64, "source": pl.String, "status": pl.String,
    "external_id": pl.String, "attempts": pl.Int64, "last_attempt_at": pl.String,
    "http_status": pl.Int64, "error": pl.String, "raw_path": pl.String,
}


def utc_now() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str | None:
    """Normalize empty text and dataset null markers to None."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text != "\\N" else None


def positive_int(value: Any) -> int | None:
    """Parse a positive identifier or measurement without fabricating zeroes."""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def atomic_json(path: Path, value: dict, compressed: bool = False) -> None:
    """Atomically save a JSON object, optionally compressed with gzip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    opener = gzip.open if compressed else open
    with opener(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                  indent=None if compressed else 2)
        stream.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> dict | None:
    """Read a gzip cache envelope, returning None for absent or corrupt files."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError, EOFError):
        return None


def load_token(env_file: Path) -> str | None:
    """Read the token from the environment, then a literal local .env value."""
    token = clean(os.environ.get("TMDB_READ_ACCESS_TOKEN"))
    if token or not env_file.is_file():
        return token
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == "TMDB_READ_ACCESS_TOKEN":
            try:
                parts = shlex.split(value, comments=True)
            except ValueError:
                raise ValueError("Invalid quoting for TMDB_READ_ACCESS_TOKEN in .env") from None
            if len(parts) > 1:
                raise ValueError("TMDB_READ_ACCESS_TOKEN must be one literal value")
            return clean(parts[0]) if parts else None
    return None


class AuthenticationError(RuntimeError):
    """A TMDB authentication failure that must stop further requests."""


class RateLimiter:
    def __init__(self, rate: float, stop: threading.Event) -> None:
        self.interval = 1.0 / rate
        self.next_at = 0.0
        self.lock = threading.Lock()
        self.stop = stop

    def wait(self) -> None:
        while not self.stop.is_set():
            with self.lock:
                delay = self.next_at - time.monotonic()
                if delay <= 0:
                    self.next_at = time.monotonic() + self.interval
                    return
            self.stop.wait(delay)
        raise AuthenticationError("Further requests stopped")

    def pause(self, seconds: float) -> None:
        with self.lock:
            self.next_at = max(self.next_at, time.monotonic() + max(0.0, seconds))


class Client:
    """Fetch public API responses with shared throttling and an atomic disk cache."""
    def __init__(self, args: argparse.Namespace, token: str | None) -> None:
        self.args = argparse.Namespace(**vars(args))
        self.token = token
        self.stop = threading.Event()
        self.limiters = {"tmdb": RateLimiter(args.tmdb_rate, self.stop),
                         "wikimedia": RateLimiter(args.wikimedia_rate, self.stop)}
        self.lock = threading.Lock()
        self.cache_locks: dict[str, threading.Lock] = {}
        self.fetched_paths: set[str] = set()
        self.failed_results: dict[str, dict] = {}
        self.sparql_disabled = False
        self.requests = {source: 0 for source in ("tmdb", "wikidata", "wikipedia")}
        self.cache_hits = dict.fromkeys(self.requests, 0)

    def path(self, source: str, kind: str, key: str) -> Path:
        safe = key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else hashlib.sha256(key.encode()).hexdigest()
        return self.args.raw_dir / source / LANGUAGE / kind / (safe + ".json.gz")

    def fetch(self, source: str, kind: str, key: str, endpoint: str,
              parameters: dict[str, str]) -> dict:
        path = self.path(source, kind, key)
        with self.lock:
            lock = self.cache_locks.setdefault(str(path), threading.Lock())
        with lock:
            if str(path) in self.failed_results:
                return self.failed_results[str(path)]
            cached = read_json(path)
            if cached and (self.args.rebuild_only or (
                    (not self.args.refresh or str(path) in self.fetched_paths)
                    and cached.get("outcome") in {"success", "not_found"})):
                with self.lock:
                    self.cache_hits[source] += 1
                return {**cached, "raw_path": str(path), "cache_hit": True}
            if self.args.rebuild_only:
                return {"outcome": "not_attempted", "attempts": 0, "raw_path": None,
                        "response": None, "external_id": key}
            if source == "tmdb" and not self.token:
                raise AuthenticationError("TMDB_READ_ACCESS_TOKEN is not configured")
            limiter = self.limiters["tmdb" if source == "tmdb" else "wikimedia"]
            headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
            if source == "tmdb":
                headers["Authorization"] = "Bearer " + self.token
            url = endpoint + "?" + urlencode(parameters)
            result = {"source": source, "endpoint": endpoint, "parameters": parameters,
                      "external_id": key, "language": LANGUAGE, "response": None,
                      "outcome": "request_error", "http_status": None, "error": None}
            for attempt in range(1, self.args.attempts + 1):
                limiter.wait()
                with self.lock:
                    self.requests[source] += 1
                result.update(attempts=attempt, fetched_at=utc_now())
                try:
                    with urlopen(Request(url, headers=headers), timeout=self.args.timeout) as response:
                        result["http_status"] = response.status
                        result["response"] = json.load(response)
                    if result["response"].get("error"):
                        code = str(result["response"]["error"].get("code", "unknown"))
                        result["error"] = "API error: " + code
                        if code in {"maxlag", "ratelimited", "readonly"}:
                            limiter.pause(max(5, min(2 ** (attempt - 1), 30)))
                            continue
                        break
                    result.update(outcome="success", error=None)
                    break
                except HTTPError as error:
                    result["http_status"] = error.code
                    result["error"] = "HTTP " + str(error.code)
                    if source == "tmdb" and error.code in {401, 403}:
                        result["outcome"] = "auth_error"
                        target = self.path(source, "failed_" + kind, key + "-" + str(time.time_ns())) if (
                            cached and cached.get("outcome") == "success") else path
                        atomic_json(target, result, compressed=True)
                        self.stop.set()
                        raise AuthenticationError("TMDB rejected authentication (HTTP " + str(error.code) + ")") from None
                    if error.code == 404:
                        result["outcome"] = "not_found"
                        break
                    if error.code != 429 and not 500 <= error.code < 600:
                        break
                    if source == "wikidata" and endpoint == "https://query.wikidata.org/sparql" and error.code == 429:
                        self.sparql_disabled = True
                        result["error"] = "WDQS rate limited; use Wikidata Action API search instead"
                        result["outcome"] = "service_unavailable"
                        break
                    retry_after = error.headers.get("Retry-After", "")
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        try:
                            delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                        except (TypeError, ValueError, OverflowError):
                            delay = 0.0
                    limiter.pause(max(delay, min(2 ** (attempt - 1), 30) + random.random()))
                except (URLError, TimeoutError, OSError, ValueError) as error:
                    result["error"] = type(error).__name__
                    limiter.pause(min(2 ** (attempt - 1), 30) + random.random())
                print(f"{source}/{kind}: attempt {attempt}/{self.args.attempts}: {result['error']}", flush=True)
            target = path
            if result["outcome"] != "success" and cached and cached.get("outcome") == "success":
                target = self.path(source, "failed_" + kind, key + "-" + str(time.time_ns()))
            atomic_json(target, result, compressed=True)
            self.fetched_paths.add(str(path))
            returned = {**result, "raw_path": str(target), "cache_hit": False}
            if target != path:
                self.failed_results[str(path)] = returned
            return returned

    def batch(self, source: str, kind: str, keys: list[str], endpoint: str,
              parameters_for: Any, batch_size: int) -> dict[str, dict]:
        """Cache each key against the complete, original response of its batch."""
        result: dict[str, dict] = {}
        pending = []
        for key in sorted(set(keys)):
            path = self.path(source, kind, key)
            cached = read_json(path)
            if cached and (self.args.rebuild_only or (
                    not self.args.refresh and cached.get("outcome") == "success")):
                result[key] = {**cached, "raw_path": str(path), "cache_hit": True}
                self.cache_hits[source] += 1
            elif self.args.rebuild_only:
                result[key] = {"outcome": "not_attempted", "attempts": 0, "response": None,
                               "raw_path": None, "external_id": key}
            else:
                pending.append(key)
        for offset in range(0, len(pending), batch_size):
            group = pending[offset:offset + batch_size]
            batch_key = hashlib.sha256("\n".join(group).encode()).hexdigest()
            if endpoint == "https://query.wikidata.org/sparql" and self.sparql_disabled:
                fetched = {"outcome": "service_unavailable", "attempts": 0, "response": None,
                           "error": "WDQS unavailable; using Wikidata Action API", "cache_hit": False}
            else:
                fetched = self.fetch(source, kind + "_batches", batch_key, endpoint, parameters_for(group))
            for key in group:
                path = self.path(source, kind, key)
                envelope = {k: v for k, v in fetched.items() if k not in {"raw_path", "cache_hit"}}
                envelope["batch_keys"] = group
                envelope["indexed_for"] = key
                previous = read_json(path)
                target = path
                if envelope["outcome"] != "success" and previous and previous.get("outcome") == "success":
                    target = self.path(source, "failed_" + kind, key + "-" + str(time.time_ns()))
                atomic_json(target, envelope, compressed=True)
                result[key] = {**envelope, "raw_path": str(target), "cache_hit": fetched["cache_hit"]}
        return result


def load_catalog(args: argparse.Namespace) -> list[dict]:
    """Build the complete catalog using original training IDs and MovieLens metadata."""
    items = pl.read_parquet(args.data_dir / "train.parquet", columns=["movieId"]).unique().sort("movieId")
    with zipfile.ZipFile(args.archive) as archive:
        movies = pl.read_csv(io.BytesIO(archive.read("ml-20m/movies.csv")))
        links = pl.read_csv(io.BytesIO(archive.read("ml-20m/links.csv")),
                            schema_overrides={"imdbId": pl.String, "tmdbId": pl.String})
    if movies["movieId"].n_unique() != movies.height or links["movieId"].n_unique() != links.height:
        raise ValueError("Duplicate movieId in MovieLens movies/links")
    joined = items.join(movies, on="movieId", how="left").join(links, on="movieId", how="left")
    if joined["title"].null_count():
        raise ValueError("Some training movies have no MovieLens title")
    manifest_path = args.data_dir / "manifest.json"
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text()).get("ten_core", {}).get("items")
        if expected is not None and joined.height != expected:
            raise ValueError("Training catalog does not match input manifest")
    records = []
    for item in joined.iter_rows(named=True):
        record = dict.fromkeys(CATALOG_SCHEMA)
        title = item["title"]
        year = re.search(r"\s*\((\d{4})\)\s*$", title)
        genres = [g for g in item["genres"].split("|") if g != "(no genres listed)"]
        imdb = clean(item["imdbId"])
        if imdb and not imdb.startswith("tt"):
            imdb = "tt" + imdb.zfill(7)
        record.update(movieId=item["movieId"], imdb_id=imdb,
                      tmdb_id=positive_int(item["tmdbId"]), movielens_title=title,
                      title=title[:year.start()].strip() if year else title,
                      release_year=int(year[1]) if year else None,
                      genres_movielens=genres or None, genres=genres or None,
                      tmdb_mapping_status="not_attempted")
        records.append(record)
    return records


class Outputs:
    def __init__(self, args: argparse.Namespace, records: list[dict], client: Client) -> None:
        self.args = args
        self.records = {r["movieId"]: r for r in records}
        self.baseline_records = {r["movieId"]: dict(r) for r in records}
        self.client = client
        self.started_at = utc_now()
        self.provenance: dict[tuple[int, str], dict] = {}
        self.statuses: dict[tuple[int, str], dict] = {}
        for record in records:
            for source in ("tmdb", "wikidata", "wikipedia"):
                self.status(record, source, "not_attempted")
            for field in ("movieId", "imdb_id", "tmdb_id", "movielens_title", "title",
                          "release_year", "genres_movielens", "genres"):
                if record[field] is not None:
                    self.provenance[(record["movieId"], field)] = {
                        "movieId": record["movieId"], "field": field, "source": "movielens",
                        "external_id": str(record["movieId"]), "language": None,
                        "fetched_at": None, "raw_path": str(args.archive),
                        "source_url": "https://grouplens.org/datasets/movielens/20m/",
                        "revision_id": None}
        self.baseline_provenance = dict(self.provenance)

    def status(self, record: dict, source: str, status: str, response: dict | None = None,
               external_id: Any = None, error: str | None = None) -> None:
        response = response or {}
        self.statuses[(record["movieId"], source)] = {
            "movieId": record["movieId"], "source": source, "status": status,
            "external_id": clean(external_id), "attempts": response.get("attempts", 0),
            "last_attempt_at": response.get("fetched_at"), "http_status": response.get("http_status"),
            "error": error or response.get("error"), "raw_path": response.get("raw_path")}

    def put(self, record: dict, field: str, value: Any, source: str, response: dict,
            external_id: Any, source_url: str | None = None, revision: Any = None) -> None:
        if value is None or value == [] or value == "":
            return
        record[field] = value
        self.provenance[(record["movieId"], field)] = {
            "movieId": record["movieId"], "field": field, "source": source,
            "external_id": str(external_id), "language": "en" if source != "movielens" else None,
            "fetched_at": response.get("fetched_at"), "raw_path": response.get("raw_path"),
            "source_url": source_url, "revision_id": clean(revision)}

    def save(self, selected: list[int], stage: str) -> dict:
        """Write catalog, field provenance, crawl status, and a consistent manifest."""
        directory = self.args.output_dir
        directory.mkdir(parents=True, exist_ok=True)
        frames = {
            "catalog": pl.DataFrame(list(self.records.values()), schema=CATALOG_SCHEMA).sort("movieId"),
            "provenance": pl.DataFrame(list(self.provenance.values()), schema=PROVENANCE_SCHEMA).sort(["movieId", "field"]),
            "crawl_status": pl.DataFrame(list(self.statuses.values()), schema=STATUS_SCHEMA).sort(["movieId", "source"]),
        }
        for name, frame in frames.items():
            target = directory / (name + ".parquet")
            temporary = target.with_name(target.name + ".tmp")
            frame.write_parquet(temporary, compression="zstd")
            temporary.replace(target)
        rows = list(self.records.values())
        fields = tuple(field for field in CATALOG_SCHEMA if field not in {"movieId", "tmdb_mapping_status"})
        coverage = {field: {"movies": sum(bool(row[field]) for row in rows),
                            "fraction": sum(bool(row[field]) for row in rows) / len(rows)} for field in fields}
        status_counts = {source: {} for source in self.client.requests}
        for value in self.statuses.values():
            counts = status_counts[value["source"]]
            counts[value["status"]] = counts.get(value["status"], 0) + 1
        incomplete = [row["movieId"] for row in rows if not row["overview"]]
        manifest = {
            "schema_version": 1, "run_started_at": self.started_at, "updated_at": utc_now(),
            "stage": stage, "language": LANGUAGE, "catalog_items": len(rows),
            "selected_items": len(selected), "selected_movie_ids": selected,
            "attempted_items": sum(any(self.statuses[(row["movieId"], source)]["status"]
                not in {"not_attempted", "skipped_not_needed"} for source in self.client.requests) for row in rows),
            "catalog_ids_sha256": hashlib.sha256("\n".join(str(row["movieId"]) for row in rows).encode()).hexdigest(),
            "input": {"data_dir": str(self.args.data_dir), "archive": str(self.args.archive)},
            "config": {"tmdb_rate": self.args.tmdb_rate, "wikimedia_rate": self.args.wikimedia_rate,
                       "workers": self.args.workers, "max_attempts": self.args.attempts,
                       "refresh": self.args.refresh, "rebuild_only": self.args.rebuild_only},
            "network_requests_this_run": dict(self.client.requests), "cache_hits_this_run": dict(self.client.cache_hits),
            "coverage": coverage, "status_counts": status_counts,
            "tmdb_mapping_counts": {state: sum(row["tmdb_mapping_status"] == state for row in rows)
                for state in sorted({row["tmdb_mapping_status"] for row in rows})},
            "movies_without_overview": incomplete,
            "selected_movies_without_overview": [i for i in selected if not self.records[i]["overview"]],
            "output_files": {name: {"rows": frame.height, "schema": {k: str(v) for k, v in frame.schema.items()}}
                             for name, frame in frames.items()},
            "attribution": {
                "movielens": "GroupLens MovieLens 20M; research use, see original README.txt.",
                "tmdb": "This product uses the TMDB API but is not endorsed or certified by TMDB.",
                "wikidata": "Wikidata structured data: CC0.",
                "wikipedia": "English Wikipedia text: retain article links/revisions and applicable CC BY-SA attribution.",
            },
        }
        atomic_json(directory / "manifest.json", manifest)
        return manifest


def tmdb_movie(client: Client, record: dict) -> tuple[dict, str, int | None]:
    """Resolve a movie by its linked TMDB ID, falling back to an exact IMDb lookup."""
    def details(identifier: int) -> dict:
        return client.fetch("tmdb", "movies", str(identifier),
                            "https://api.themoviedb.org/3/movie/" + str(identifier),
                            {"language": LANGUAGE, "append_to_response": "credits,keywords"})

    def identity(response: dict, expected_id: int) -> str:
        if response["outcome"] != "success":
            return response["outcome"]
        body = response["response"]
        if positive_int(body.get("id")) != expected_id:
            return "mapping_mismatch"
        imdb = clean(body.get("imdb_id"))
        if imdb and imdb != record["imdb_id"]:
            return "mapping_mismatch"
        return "verified" if imdb else "unverified"

    identifier = record["tmdb_id"]
    response = None
    state = "not_found"
    if identifier:
        response = details(identifier)
        state = identity(response, identifier)
        if state in {"verified", "unverified", "request_error", "not_attempted"}:
            return response, state, identifier
    imdb = record["imdb_id"]
    if not imdb:
        return response or {"outcome": "not_found", "attempts": 0}, state, identifier
    found = client.fetch("tmdb", "find", imdb, "https://api.themoviedb.org/3/find/" + imdb,
                         {"external_source": "imdb_id", "language": LANGUAGE})
    if found["outcome"] != "success":
        return found, found["outcome"], identifier
    candidates = found["response"].get("movie_results", [])
    ids = sorted({i for item in candidates if (i := positive_int(item.get("id")))})
    if len(ids) != 1:
        state = "ambiguous" if len(ids) > 1 else "not_found"
        if not ids and any(found["response"].get(k) for k in ("tv_results", "tv_episode_results")):
            state = "unsupported_media"
        return found, state, identifier
    response = details(ids[0])
    state = identity(response, ids[0])
    return response, state, ids[0]


def apply_tmdb(outputs: Outputs, record: dict, result: tuple[dict, str, int | None]) -> None:
    """Normalize a resolved TMDB movie and track the source of populated fields."""
    response, state, identifier = result
    record["tmdb_mapping_status"] = state
    if state not in {"verified", "unverified"}:
        outputs.status(record, "tmdb", state, response, identifier)
        return
    if outputs.args.refresh:
        identifier_ml = record["movieId"]
        record.clear()
        record.update(outputs.baseline_records[identifier_ml])
        record["tmdb_mapping_status"] = state
        for field in CATALOG_SCHEMA:
            key = (identifier_ml, field)
            if key in outputs.baseline_provenance:
                outputs.provenance[key] = outputs.baseline_provenance[key]
            else:
                outputs.provenance.pop(key, None)
    body = response["response"]
    put = lambda field, value: outputs.put(record, field, value, "tmdb", response, identifier,
                                           "https://www.themoviedb.org/movie/" + str(identifier))
    put("tmdb_id", identifier)
    for field in ("title", "original_title", "release_date", "tagline", "original_language"):
        put(field, clean(body.get(field)))
    date = clean(body.get("release_date"))
    if date and re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        put("release_year", int(date[:4]))
    genres = [g["name"] for g in body.get("genres", []) if clean(g.get("name"))]
    put("genres_tmdb", genres)
    put("genres", genres)
    overview = clean(body.get("overview"))
    put("tmdb_overview", overview)
    if overview:
        put("overview", overview)
        put("overview_source", "tmdb")
        put("text_language", "en")
    keywords = (body.get("keywords") or {}).get("keywords", [])
    put("keywords", [{"id": str(k["id"]), "name": k["name"], "source": "tmdb"}
                     for k in keywords if k.get("id") is not None and clean(k.get("name"))])
    credits = body.get("credits") or {}
    def people(entries: list[dict]) -> list[dict]:
        unique = {}
        for item in entries:
            if item.get("id") is not None:
                unique.setdefault(item["id"], {"id": str(item["id"]), "name": clean(item.get("name")),
                    "source": "tmdb", "order": item.get("order")})
        return list(unique.values())
    put("cast", people(sorted(credits.get("cast", []), key=lambda item: item.get("order", 999999))))
    crew = credits.get("crew", [])
    put("directors", people([p for p in crew if p.get("job") == "Director"]))
    put("writers", people([p for p in crew if p.get("department") == "Writing"]))
    put("runtime_minutes", positive_int(body.get("runtime")))
    for field, code in (("production_countries", "iso_3166_1"), ("spoken_languages", "iso_639_1")):
        put(field, [{"id": p[code], "name": clean(p.get("english_name") or p.get("name")), "source": "tmdb"}
                    for p in body.get(field, []) if clean(p.get(code))])
    collection = body.get("belongs_to_collection") or {}
    put("collection_id", positive_int(collection.get("id")))
    put("collection_name", clean(collection.get("name")))
    outputs.status(record, "tmdb", "success", response, identifier)


def needs_fallback(record: dict) -> bool:
    """Identify missing descriptions or key attributes Wikimedia may supply."""
    return any(not record[field] for field in (
        "overview", "genres", "cast", "directors", "runtime_minutes", "original_language", "production_countries"))


def claim_values(entity: dict, prop: str) -> list[Any]:
    """Read nondeprecated Wikidata values, honoring preferred statements."""
    claims = [claim for claim in entity.get("claims", {}).get(prop, []) if claim.get("rank") != "deprecated"]
    preferred = [claim for claim in claims if claim.get("rank") == "preferred"]
    return [claim["mainsnak"]["datavalue"]["value"] for claim in preferred or claims
            if "datavalue" in claim.get("mainsnak", {})]


class PlainText(HTMLParser):
    """Extract article prose while excluding tables, references, and edit controls."""
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        blocked = tag in {"table", "script", "style", "sup"} or any(
            c in {"reference", "reflist", "mw-editsection", "navbox", "hatnote"} for c in classes)
        if tag not in self.VOID:
            self.stack.append((tag, blocked))
        if tag in {"p", "li", "br", "h2", "h3"} and not any(b for _, b in self.stack):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if tag in {"p", "li", "h2", "h3"} and not any(b for _, b in self.stack):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not any(blocked for _, blocked in self.stack):
            self.parts.append(data)

    def text(self) -> str | None:
        return clean("\n\n".join(re.sub(r"\s+", " ", line).strip()
                     for line in "".join(self.parts).splitlines() if line.strip()))


def plain_text(html: str) -> str | None:
    """Convert Wikipedia HTML to cleaned prose without references or tables."""
    parser = PlainText()
    parser.feed(html)
    return parser.text()


def wikipedia_movie(client: Client, url: str) -> tuple[dict, str, str | None]:
    """Fetch a linked English article's plot, falling back to its introduction."""
    parsed = urlparse(url)
    if parsed.hostname != "en.wikipedia.org" or not parsed.path.startswith("/wiki/"):
        return {"outcome": "request_error", "error": "Invalid English Wikipedia sitelink"}, "", None
    title = unquote(parsed.path[6:])
    sections = client.fetch("wikipedia", "sections", title, "https://en.wikipedia.org/w/api.php",
                            {"action": "parse", "page": title, "prop": "sections|revid", "redirects": "1", "format": "json"})
    if sections["outcome"] != "success":
        return sections, "", None
    selected = None
    acceptable = {"plot", "synopsis", "plot synopsis", "plot summary"}
    for section in sections["response"].get("parse", {}).get("sections", []):
        name = plain_text(section.get("line", "")) or ""
        if name.casefold() in acceptable:
            selected = section
            break
    for section in ([selected, None] if selected else [None]):
        index = section["index"] if section else "0"
        name = plain_text(section["line"]) if section else "Introduction"
        content = client.fetch("wikipedia", "content", title + ":" + index,
                               "https://en.wikipedia.org/w/api.php",
                               {"action": "parse", "page": title, "section": index,
                                "prop": "text|revid", "redirects": "1", "format": "json"})
        if content["outcome"] != "success":
            return content, name, None
        html = content["response"].get("parse", {}).get("text", {}).get("*", "")
        text = plain_text(html)
        if text:
            return content, name, text
    return content, "Introduction", None


def crawl_wikimedia(outputs: Outputs, selected: list[int], replay: bool = False) -> None:
    """Fill missing content using exact Wikidata mappings and English articles."""
    client = outputs.client
    records = [outputs.records[i] for i in selected if needs_fallback(outputs.records[i])]
    for identifier in selected:
        record = outputs.records[identifier]
        if not needs_fallback(record):
            for source in ("wikidata", "wikipedia"):
                if outputs.statuses[(identifier, source)]["status"] == "not_attempted":
                    outputs.status(record, source, "skipped_not_needed")
    if not replay:
        print(f"Wikimedia fallback: {len(records)} movies", flush=True)
    imdb_ids = [r["imdb_id"] for r in records if r["imdb_id"]]
    def query_parameters(keys: list[str]) -> dict[str, str]:
        values = " ".join(json.dumps(key) for key in keys)
        query = ('SELECT ?imdb ?item ?article WHERE { VALUES ?imdb { ' + values +
                 ' } ?item wdt:P345 ?imdb . OPTIONAL { ?article schema:about ?item ; '
                 'schema:isPartOf <https://en.wikipedia.org/> . } }')
        return {"query": query, "format": "json"}
    def statement_search(imdb: str) -> dict:
        searched = client.fetch("wikidata", "statement_search", imdb, "https://www.wikidata.org/w/api.php",
                                {"action": "query", "list": "search", "srsearch": "haswbstatement:P345=" + imdb,
                                 "srnamespace": "0", "srlimit": "50", "maxlag": "5", "format": "json"})
        if searched["outcome"] != "success":
            return searched
        bindings = [{"imdb": {"value": imdb}, "item": {"value": "http://www.wikidata.org/entity/" + hit["title"]}}
                    for hit in searched["response"].get("query", {}).get("search", [])
                    if re.fullmatch(r"Q\d+", hit.get("title", ""))]
        # Keep the original response on disk and adapt its shape only in memory.
        return {**searched, "response": {"results": {"bindings": bindings}}, "statement_search": True}

    mappings = {}
    query_ids = []
    for imdb in sorted(set(imdb_ids)):
        cached_search = read_json(client.path("wikidata", "statement_search", imdb))
        if cached_search and cached_search.get("outcome") == "success" and (
                client.args.rebuild_only or not client.args.refresh):
            mappings[imdb] = statement_search(imdb)
        else:
            query_ids.append(imdb)
    mappings.update(client.batch("wikidata", "imdb", query_ids, "https://query.wikidata.org/sparql", query_parameters, 25))
    # WDQS can be unavailable even while the independent Wikidata Action API works.
    # Fall back to its exact statement search and validate P345 on the entity below.
    for imdb, response in list(mappings.items()):
        if response["outcome"] not in {"success", "not_attempted"}:
            mappings[imdb] = statement_search(imdb)
    resolved: dict[int, tuple[str, str | None, dict]] = {}
    for record in records:
        response = mappings.get(record["imdb_id"], {"outcome": "not_found", "attempts": 0})
        if response["outcome"] != "success":
            outputs.status(record, "wikidata", response["outcome"], response, record["imdb_id"])
            outputs.status(record, "wikipedia", "not_attempted" if response["outcome"] == "not_attempted" else "no_sitelink")
            continue
        bindings = response["response"].get("results", {}).get("bindings", [])
        matches = [row for row in bindings if row.get("imdb", {}).get("value") == record["imdb_id"]]
        qids = {row["item"]["value"].rsplit("/", 1)[-1] for row in matches}
        if len(qids) != 1:
            outputs.status(record, "wikidata", "ambiguous" if qids else "not_found", response, record["imdb_id"])
            outputs.status(record, "wikipedia", "no_sitelink")
            continue
        qid = next(iter(qids))
        urls = {row["article"]["value"] for row in matches if "article" in row}
        url = next(iter(urls)) if len(urls) == 1 else None
        outputs.put(record, "wikidata_id", qid, "wikidata", response, qid, "https://www.wikidata.org/wiki/" + qid)
        if url:
            outputs.put(record, "wikipedia_url", url, "wikidata", response, qid, "https://www.wikidata.org/wiki/" + qid)
        resolved[record["movieId"]] = (qid, url, response)
    if not replay:
        outputs.save(selected, "wikidata_mapping")
    def entity_parameters(keys: list[str]) -> dict[str, str]:
        return {"action": "wbgetentities", "ids": "|".join(keys), "props": "claims|labels|sitelinks",
                "languages": "en", "languagefallback": "1", "maxlag": "5", "format": "json"}
    entities = client.batch("wikidata", "entities", [item[0] for item in resolved.values()],
                            "https://www.wikidata.org/w/api.php", entity_parameters, 50)
    refs: set[str] = set()
    for identifier, (qid, _, _) in resolved.items():
        response = entities[qid]
        if response["outcome"] != "success":
            continue
        entity = response["response"].get("entities", {}).get(qid, {})
        record = outputs.records[identifier]
        properties = {"cast": "P161", "directors": "P57", "writers": "P58", "genres": "P136",
                      "production_countries": "P495", "original_language": "P364"}
        for field, prop in properties.items():
            if not record[field]:
                refs.update(value["id"] for value in claim_values(entity, prop) if isinstance(value, dict) and "id" in value)
    labels = client.batch("wikidata", "labels", list(refs), "https://www.wikidata.org/w/api.php",
                          lambda keys: {"action": "wbgetentities", "ids": "|".join(keys), "props": "labels",
                                        "languages": "en", "languagefallback": "1", "maxlag": "5", "format": "json"}, 50)
    language_ids = set()
    for identifier, (qid, _, _) in resolved.items():
        if not outputs.records[identifier]["original_language"]:
            entity = (entities[qid].get("response") or {}).get("entities", {}).get(qid, {})
            language_ids.update(value["id"] for value in claim_values(entity, "P364")
                                if isinstance(value, dict) and "id" in value)
    languages = client.batch("wikidata", "entities", list(language_ids), "https://www.wikidata.org/w/api.php",
                             entity_parameters, 50)
    def label(qid: str) -> str | None:
        result = labels.get(qid, {})
        body = result.get("response") or {}
        values = body.get("entities", {}).get(qid, {}).get("labels", {})
        return clean(values.get("en", {}).get("value"))
    for count, record in enumerate(records, 1):
        resolved_item = resolved.get(record["movieId"])
        if resolved_item:
            qid, url, mapping = resolved_item
            response = entities[qid]
            entity = (response.get("response") or {}).get("entities", {}).get(qid, {})
            if mapping.get("statement_search") and response["outcome"] == "success":
                if record["imdb_id"] not in claim_values(entity, "P345"):
                    outputs.status(record, "wikidata", "mapping_mismatch", response, qid)
                    outputs.status(record, "wikipedia", "no_sitelink")
                    record["wikidata_id"] = None
                    outputs.provenance.pop((record["movieId"], "wikidata_id"), None)
                    continue
                title = entity.get("sitelinks", {}).get("enwiki", {}).get("title")
                if title:
                    url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_-")
                    outputs.put(record, "wikipedia_url", url, "wikidata", response, qid,
                                "https://www.wikidata.org/wiki/" + qid)
            if response["outcome"] == "success" and "missing" not in entity and entity:
                put = lambda field, value: outputs.put(record, field, value, "wikidata", response, qid,
                    "https://www.wikidata.org/wiki/" + qid, entity.get("lastrevid"))
                for field, prop in (("cast", "P161"), ("directors", "P57"), ("writers", "P58")):
                    if not record[field]:
                        people = [{"id": value["id"], "name": label(value["id"]), "source": "wikidata", "order": None}
                                  for value in claim_values(entity, prop) if isinstance(value, dict) and "id" in value]
                        put(field, people)
                if not record["genres"]:
                    put("genres", [name for value in claim_values(entity, "P136")
                                   if isinstance(value, dict) and (name := label(value.get("id", "")))])
                if not record["production_countries"]:
                    put("production_countries", [{"id": value["id"], "name": label(value["id"]), "source": "wikidata"}
                        for value in claim_values(entity, "P495") if isinstance(value, dict) and "id" in value])
                if not record["original_language"]:
                    for value in claim_values(entity, "P364"):
                        language_id = value.get("id") if isinstance(value, dict) else None
                        language = (languages.get(language_id, {}).get("response") or {}).get("entities", {}).get(language_id, {})
                        codes = claim_values(language, "P218")
                        if codes:
                            put("original_language", clean(codes[0]))
                            break
                if not record["runtime_minutes"]:
                    for value in claim_values(entity, "P2047"):
                        if isinstance(value, dict) and value.get("unit", "").endswith("/Q7727"):
                            try:
                                put("runtime_minutes", positive_int(round(float(value["amount"]))))
                            except (KeyError, ValueError, OverflowError):
                                pass
                            break
                if not record["release_date"]:
                    dates = [value for value in claim_values(entity, "P577") if isinstance(value, dict)]
                    dates = [value for value in dates if value.get("precision", 0) >= 11
                             and re.match(r"^\+\d{4}-\d{2}-\d{2}T", value.get("time", ""))
                             and value.get("calendarmodel", "").endswith("/Q1985727")]
                    if dates:
                        date = min(value["time"][1:11] for value in dates)
                        put("release_date", date)
                        put("release_year", int(date[:4]))
                outputs.status(record, "wikidata", "success", response, qid)
            else:
                outputs.status(record, "wikidata", response["outcome"] if response["outcome"] != "success" else "not_found", response, qid)
            if record["tmdb_overview"]:
                outputs.status(record, "wikipedia", "skipped_not_needed", external_id=url)
            elif not url:
                outputs.status(record, "wikipedia", "no_sitelink")
            else:
                response, section, text = wikipedia_movie(client, url)
                if response["outcome"] == "success" and text:
                    revision = response["response"].get("parse", {}).get("revid")
                    field = "wikipedia_intro" if section == "Introduction" else "wikipedia_plot"
                    for name, value in ((field, text), ("overview", text), ("overview_source", "wikipedia"),
                                        ("text_language", "en"), ("wikipedia_section", section)):
                        outputs.put(record, name, value, "wikipedia", response, url, url, revision)
                    outputs.status(record, "wikipedia", "success", response, url)
                else:
                    outputs.status(record, "wikipedia", "no_text" if response["outcome"] == "success" else response["outcome"], response, url)
        if not replay and count % outputs.args.checkpoint_every == 0:
            outputs.save(selected, "wikimedia")
            print(f"Wikimedia: {count}/{len(records)} movies; network={client.requests}", flush=True)


def arguments() -> argparse.Namespace:
    """Parse repository-anchored paths and bounded collection settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ml20m_lightgcn")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/ml-20m.zip")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/movie_metadata")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/movie_content")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--movie-ids", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tmdb-rate", type=float, default=5.0)
    parser.add_argument("--wikimedia-rate", type=float, default=1.0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--rebuild-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.tmdb_rate <= 5 or not 0 < args.wikimedia_rate <= 1:
        parser.error("Rates must be positive and at most 5/s for TMDB, 1/s for Wikimedia")
    if min(args.workers, args.attempts, args.checkpoint_every, args.timeout) <= 0:
        parser.error("Workers, attempts, checkpoint interval and timeout must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.refresh and args.rebuild_only:
        parser.error("--refresh and --rebuild-only are mutually exclusive")
    for name in ("data_dir", "archive", "raw_dir", "output_dir", "env_file"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir == args.data_dir:
        parser.error("--output-dir must differ from the input split directory")
    return args


def main() -> int:
    """Crawl the selected movies and always retain the complete training catalog."""
    args = arguments()
    token = load_token(args.env_file)
    if not token and not args.rebuild_only:
        print("TMDB_READ_ACCESS_TOKEN is missing. Configure it or use --rebuild-only.", file=sys.stderr)
        return 2
    records = load_catalog(args)
    available = {r["movieId"] for r in records}
    if args.movie_ids and set(args.movie_ids) - available:
        raise ValueError("Requested movie IDs are not in the training catalog")
    selected = sorted(set(args.movie_ids)) if args.movie_ids else sorted(available)
    if args.limit:
        selected = selected[:args.limit]
    client = Client(args, token)
    outputs = Outputs(args, records, client)
    # Replay existing raw data for every item so a partial run cannot erase earlier enrichment.
    if any(args.raw_dir.glob("tmdb/" + LANGUAGE + "/movies/*.json.gz")):
        client.args.rebuild_only = True
        for record in records:
            apply_tmdb(outputs, record, tmdb_movie(client, record))
        cached_wikimedia_ids = [r["movieId"] for r in records if r["imdb_id"] and (
            client.path("wikidata", "imdb", r["imdb_id"]).is_file()
            or client.path("wikidata", "statement_search", r["imdb_id"]).is_file())]
        if cached_wikimedia_ids:
            crawl_wikimedia(outputs, cached_wikimedia_ids, replay=True)
        client.args.rebuild_only = args.rebuild_only
    outputs.save(selected, "starting")
    print(f"Catalog: {len(records)} movies; selected: {len(selected)}; rebuild_only={args.rebuild_only}", flush=True)
    stage = "complete"
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {}
    try:
        for identifier in selected:
            record = outputs.records[identifier]
            futures[executor.submit(tmdb_movie, client, record)] = identifier
        for count, future in enumerate(as_completed(futures), 1):
            identifier = futures.pop(future)
            apply_tmdb(outputs, outputs.records[identifier], future.result())
            if count % args.checkpoint_every == 0 or count == len(selected):
                outputs.save(selected, "tmdb")
                print(f"TMDB: {count}/{len(selected)} movies; network={client.requests['tmdb']}; cache={client.cache_hits['tmdb']}", flush=True)
        executor.shutdown(wait=True)
        crawl_wikimedia(outputs, selected)
    except AuthenticationError as error:
        stage = "authentication_failed"
        print(str(error), file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        stage = "interrupted"
        print("Interrupted; cached responses and completed rows are retained.", file=sys.stderr, flush=True)
    except Exception:
        stage = "failed"
        raise
    finally:
        client.stop.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        manifest = outputs.save(selected, stage)
        print(json.dumps({"stage": stage, "catalog_items": manifest["catalog_items"],
                          "coverage": manifest["coverage"], "network_requests_this_run": client.requests,
                          "selected_movies_without_overview": len(manifest["selected_movies_without_overview"]),
                          "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)
    return 0 if stage == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
