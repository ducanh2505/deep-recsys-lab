#!/usr/bin/env python3
"""Measure Wikidata and Wikipedia coverage for MovieLens and every model split."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import html
import io
import json
from os.path import relpath
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import Request, urlopen
import zipfile

import polars as pl


ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = "MovieLensWikimediaCoverage/1.0 (local research; https://grouplens.org/datasets/movielens/20m/)"
NARRATIVE_PARSER_VERSION = 2
PROPERTIES = {
    "genres": "P136", "cast_or_voice_cast": "P161|wdt:P725", "directors": "P57",
    "writers": "P58", "runtime": "P2047", "original_language": "P364",
    "production_countries": "P495", "release_date": "P577", "main_subject": "P921",
    "based_on": "P144",
}
PLOT_HEADINGS = {
    "plot", "synopsis", "plot synopsis", "plot summary", "story", "storyline",
    "premise", "nội dung", "cốt truyện", "tóm tắt", "tóm tắt nội dung", "nội dung phim",
}
HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)


def utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: dict, compressed: bool = False) -> None:
    """Write an atomic JSON result or compressed API response."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with (gzip.open if compressed else open)(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=None if compressed else 2)
        stream.write("\n")
    temporary.replace(path)


class Client:
    """Cache public API requests and share a conservative request rate."""

    def __init__(self, directory: Path, rate: float, timeout: int, attempts: int,
                 cache_only: bool = False) -> None:
        self.directory = directory
        self.interval = 1 / rate
        self.next_request = 0.0
        self.timeout = timeout
        self.attempts = attempts
        self.cache_only = cache_only
        self.requests: Counter[str] = Counter()
        self.hits: Counter[str] = Counter()

    def fetch(self, kind: str, endpoint: str, parameters: dict[str, str]) -> dict:
        """Return a cached or freshly retrieved JSON envelope, preserving errors."""
        key = hashlib.sha256(json.dumps([endpoint, parameters], sort_keys=True).encode()).hexdigest()
        path = self.directory / kind / (key + ".json.gz")
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                previous = json.load(stream)
            if previous.get("outcome") == "success":
                self.hits[kind] += 1
                return {**previous, "raw_path": str(path)}
        if self.cache_only:
            return {"outcome": "not_cached", "response": None, "raw_path": None}
        result = {"endpoint": endpoint, "parameters": parameters, "outcome": "request_error",
                  "response": None, "http_status": None}
        for attempt in range(1, self.attempts + 1):
            time.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + self.interval
            self.requests[kind] += 1
            result.update(attempts=attempt, fetched_at=utc_now())
            try:
                encoded = urlencode(parameters)
                headers = {"User-Agent": USER_AGENT, "Accept": "application/json", "Accept-Encoding": "gzip"}
                if len(endpoint) + len(encoded) > 7000:
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
                    request = Request(endpoint, data=encoded.encode(), headers=headers)
                else:
                    request = Request(endpoint + "?" + encoded, headers=headers)
                with urlopen(request, timeout=self.timeout) as response:
                    result["http_status"] = response.status
                    body = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    parsed = json.loads(body)
                if parsed.get("error") or parsed.get("errors"):
                    raise ValueError(json.dumps(parsed.get("error") or parsed.get("errors"))[:250])
                result.update(outcome="success", response=parsed, error=None)
                break
            except HTTPError as error:
                result.update(http_status=error.code, error=f"HTTP {error.code}")
                retry = error.headers.get("Retry-After", "")
                try:
                    delay = float(retry)
                except ValueError:
                    try:
                        delay = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        delay = 0
                self.next_request = time.monotonic() + max(delay, 2 ** attempt)
                if error.code not in {429, 500, 502, 503, 504}:
                    break
            except (URLError, TimeoutError, OSError, ValueError) as error:
                result["error"] = f"{type(error).__name__}: {error}"[:300]
                self.next_request = time.monotonic() + 2 ** attempt
            print(f"{kind}: attempt {attempt}/{self.attempts}: {result['error']}", flush=True)
        save_json(path, result, compressed=True)
        return {**result, "raw_path": str(path)}


def catalog_rows(archive_path: Path, split_directory: Path) -> list[dict]:
    """Read all MovieLens movies and membership in train, valid, and test."""
    memberships = {
        name: set(pl.read_parquet(split_directory / (name + ".parquet"), columns=["movieId"])
                  ["movieId"].to_list()) for name in ("train", "valid", "test")
    }
    with zipfile.ZipFile(archive_path) as archive:
        movies = pl.read_csv(io.BytesIO(archive.read("ml-20m/movies.csv")))
        links = pl.read_csv(io.BytesIO(archive.read("ml-20m/links.csv")),
                            schema_overrides={"imdbId": pl.String, "tmdbId": pl.String})
    if movies["movieId"].n_unique() != movies.height or links["movieId"].n_unique() != links.height:
        raise ValueError("Duplicate MovieLens movie IDs")
    if not set.union(*memberships.values()).issubset(set(movies["movieId"].to_list())):
        raise ValueError("Split IDs missing from the original catalog")
    rows = []
    for row in movies.join(links, on="movieId", how="left").sort("movieId").iter_rows(named=True):
        identifier = str(row.get("imdbId") or "").strip()
        identifier = identifier if identifier.startswith("tt") else "tt" + identifier.zfill(7) if identifier else None
        result = {"movieId": row["movieId"], "title": row["title"], "imdb_id": identifier,
                  **{"in_" + name: row["movieId"] in values for name, values in memberships.items()},
                  "mapping_status": "not_attempted", "wikidata_id": None, "candidate_qids": [],
                  "wikidata_description_en": None, "mapping_raw_path": None, "mapping_fetched_at": None,
                  **{"wd_" + name: False for name in PROPERTIES}}
        for language in ("en", "vi"):
            result.update({language + "_url": None, language + "_status": "no_mapping",
                           language + "_pageid": None, language + "_revid": None,
                           language + "_page_qid": None, language + "_lead_chars": 0,
                           language + "_plot_chars": 0, language + "_plot_section": None,
                           language + "_raw_path": None, language + "_raw_paths": [],
                           language + "_fetched_at": None})
        rows.append(result)
    return rows


def mapping_query(identifiers: list[str]) -> str:
    """Build an exact IMDb lookup with truthy metadata presence and sitelinks."""
    flags = " ".join("?has_" + name for name in PROPERTIES)
    tests = " ".join(f"BIND(EXISTS {{ ?item wdt:{prop} ?v_{name} }} AS ?has_{name})"
                     for name, prop in PROPERTIES.items())
    values = " ".join(json.dumps(identifier) for identifier in identifiers)
    return (f"SELECT ?imdb ?item ?en ?vi ?description {flags} WHERE {{ VALUES ?imdb {{ {values} }} "
            "?item wdt:P345 ?imdb . OPTIONAL { ?en schema:about ?item ; "
            "schema:isPartOf <https://en.wikipedia.org/> . } OPTIONAL { ?vi schema:about ?item ; "
            "schema:isPartOf <https://vi.wikipedia.org/> . } OPTIONAL { ?item schema:description "
            '?description . FILTER(LANG(?description)="en") } ' + tests + " }")


def apply_mapping(row: dict, bindings: list[dict], envelope: dict) -> None:
    """Accept a unique exact QID and keep ambiguous or failed matches separate."""
    row["mapping_raw_path"] = envelope.get("raw_path")
    row["mapping_fetched_at"] = envelope.get("fetched_at")
    if envelope["outcome"] != "success":
        row["mapping_status"] = envelope["outcome"]
        return
    matches = [binding for binding in bindings if binding.get("imdb", {}).get("value") == row["imdb_id"]]
    qids = sorted({binding["item"]["value"].rsplit("/", 1)[-1] for binding in matches})
    row["candidate_qids"] = qids
    if len(qids) != 1:
        row["mapping_status"] = "ambiguous" if qids else "not_found"
        return
    row.update(mapping_status="matched", wikidata_id=qids[0])
    row["wikidata_description_en"] = next((binding["description"]["value"] for binding in matches
                                               if binding.get("description", {}).get("value")), None)
    for name in PROPERTIES:
        row["wd_" + name] = any(binding.get("has_" + name, {}).get("value") == "true" for binding in matches)
    for language in ("en", "vi"):
        urls = {binding[language]["value"] for binding in matches if language in binding}
        row[language + "_url"] = next(iter(urls)) if len(urls) == 1 else None
        row[language + "_status"] = "not_attempted" if len(urls) == 1 else "no_sitelink"


def clean_wikitext(text: str) -> str:
    """Keep approximate narrative prose, removing markup without expanding templates."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Match self-closing references first so they cannot consume later prose.
    text = re.sub(r"<ref\b[^>]*?/\s*>|<ref\b[^>]*>.*?</ref\s*>", "", text,
                  flags=re.DOTALL | re.IGNORECASE)
    # Balanced templates include infoboxes, citations, and navigation, not prose.
    parts = []
    depth = 0
    index = 0
    while index < len(text):
        if text.startswith("{{", index):
            depth += 1
            index += 2
        elif depth and text.startswith("}}", index):
            depth -= 1
            index += 2
        else:
            if not depth:
                parts.append(text[index])
            index += 1
    text = "".join(parts)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[(?:File|Image|Category|Tập tin|Hình|Thể loại):.*?\]\]", "", text,
                  flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\[\[([^]\n]+)\]\]", lambda match: match[1].split("|")[-1], text)
    text = re.sub(r"\[(?:https?://)\S+\s*([^]]*)\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = HEADING.sub("", text).replace("'''", "").replace("''", "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def narrative_sections(wikitext: str) -> tuple[str, str, str | None]:
    """Return a cleaned lead and the longest recognized plot/premise section."""
    headings = list(HEADING.finditer(wikitext))
    lead = clean_wikitext(wikitext[:headings[0].start()] if headings else wikitext)
    plots = []
    for index, heading in enumerate(headings):
        name = clean_wikitext(heading[2]).casefold()
        if name not in PLOT_HEADINGS:
            continue
        end = next((following.start() for following in headings[index + 1:]
                    if len(following[1]) <= len(heading[1])), len(wikitext))
        plots.append((clean_wikitext(wikitext[heading.end():end]), heading[2].strip()))
    plot, section = max(plots, key=lambda item: len(item[0])) if plots else ("", None)
    return lead, plot, section


def resolve_title(title: str, query: dict) -> str:
    """Follow API title normalization and redirects without accepting redirect loops."""
    changes = {item["from"]: item["to"] for key in ("normalized", "converted", "redirects")
               for item in query.get(key, [])}
    visited = set()
    while title in changes and title not in visited:
        visited.add(title)
        title = changes[title]
    return title


def apply_article(row: dict, language: str, query: dict, envelope: dict) -> None:
    """Count narrative text only when the article's QID verifies its identity."""
    row[language + "_raw_path"] = envelope.get("raw_path")
    row[language + "_raw_paths"] = envelope.get("raw_paths") or ([envelope["raw_path"]] if envelope.get("raw_path") else [])
    row[language + "_fetched_at"] = envelope.get("fetched_at")
    if envelope["outcome"] != "success":
        row[language + "_status"] = envelope["outcome"]
        return
    title = unquote(urlparse(row[language + "_url"]).path[6:]).replace("_", " ")
    target = resolve_title(title, query)
    pages = {page["title"]: page for page in query.get("pages", [])}
    page = pages.get(target)
    if page is None:
        row[language + "_status"] = "response_incomplete"
        return
    if page.get("missing") or page.get("invalid") or page.get("ns") != 0:
        row[language + "_status"] = "missing_or_nonarticle"
        return
    qid = page.get("pageprops", {}).get("wikibase_item")
    row[language + "_page_qid"] = qid
    row[language + "_pageid"] = page.get("pageid")
    if qid != row["wikidata_id"]:
        row[language + "_status"] = "mapping_mismatch" if qid else "unverified"
        return
    revisions = page.get("revisions") or []
    if not revisions or "content" not in revisions[0].get("slots", {}).get("main", {}):
        row[language + "_status"] = "content_unavailable"
        return
    revision = revisions[0]
    lead, plot, section = narrative_sections(revision["slots"]["main"]["content"])
    row.update({language + "_status": "verified", language + "_revid": revision.get("revid"),
                language + "_lead_chars": len(lead), language + "_plot_chars": len(plot),
                language + "_plot_section": section})


def coverage_summary(rows: list[dict], threshold: int) -> dict:
    """Compute coverage by original catalog and each split, keeping unknowns explicit."""
    scopes = {"all_movielens": rows, "train_valid_test_union": [row for row in rows
              if any(row["in_" + name] for name in ("train", "valid", "test"))],
              **{name: [row for row in rows if row["in_" + name]] for name in ("train", "valid", "test")}}
    result = {}
    for scope, subset in scopes.items():
        total = len(subset)
        def metric(count: int) -> dict:
            return {"movies": count, "fraction": count / total if total else None}
        metrics = {"wikidata_exact_unique": metric(sum(row["mapping_status"] == "matched" for row in subset)),
                   "wikidata_description_en": metric(sum(bool(row["wikidata_description_en"]) for row in subset)),
                   **{"wikidata_" + name: metric(sum(row["wd_" + name] for row in subset)) for name in PROPERTIES}}
        for language in ("en", "vi"):
            verified = [row for row in subset if row[language + "_status"] == "verified"]
            metrics.update({language + "_sitelink": metric(sum(bool(row[language + "_url"]) for row in subset)),
                            language + "_verified_article": metric(len(verified)),
                            language + "_lead": metric(sum(row[language + "_lead_chars"] >= threshold for row in verified)),
                            language + "_plot": metric(sum(row[language + "_plot_chars"] >= threshold for row in verified)),
                            language + "_lead_or_plot": metric(sum(max(row[language + "_lead_chars"],
                                row[language + "_plot_chars"]) >= threshold for row in verified))})
        result[scope] = {"catalog_movies": total, "coverage": metrics,
                         "mapping_status_counts": dict(Counter(row["mapping_status"] for row in subset)),
                         "article_status_counts": {language: dict(Counter(row[language + "_status"] for row in subset))
                                                   for language in ("en", "vi")}}
    return result


def checkpoint(args: argparse.Namespace, rows: list[dict], client: Client, stage: str) -> dict:
    """Save per-movie evidence and a manifest separate from training and metadata."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.from_dicts(rows, infer_schema_length=None).sort("movieId")
    temporary = args.output_dir / "coverage.parquet.tmp"
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(args.output_dir / "coverage.parquet")
    summary = {"schema_version": 1, "updated_at": utc_now(), "stage": stage,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "narrative_parser_version": NARRATIVE_PARSER_VERSION,
               "input": {"archive": str(args.archive), "split_directory": str(args.data_dir)},
               "catalog_ids_sha256": hashlib.sha256("\n".join(str(row["movieId"]) for row in rows).encode()).hexdigest(),
               "threshold_characters": args.min_chars, "mapping_method": "exact unique IMDb P345 via WDQS truthy statements",
               "narrative_method": "latest wikitext revision, cleaned without template expansion; article QID verified",
               "languages_requested": args.languages, "requests_this_run": dict(client.requests),
               "cache_hits_this_run": dict(client.hits), "scopes": coverage_summary(rows, args.min_chars)}
    save_json(args.output_dir / "manifest.json", summary)
    return summary


def write_report(summary: dict, directory: Path) -> None:
    """Render a concise coverage report with scope, thresholds, and source status."""
    scopes = summary["scopes"]
    all_movies = scopes["all_movielens"]
    union = scopes["train_valid_test_union"]
    labels = {"all_movielens": "Toàn bộ MovieLens gốc", "train_valid_test_union": "Hợp train + valid + test",
              "train": "Train", "valid": "Validation", "test": "Test"}
    names = {
        "wikidata_exact_unique": "Wikidata: khớp duy nhất IMDb ID",
        "wikidata_description_en": "Wikidata: description tiếng Anh",
        "en_sitelink": "Wikipedia EN: có sitelink",
        "en_verified_article": "Wikipedia EN: bài đã xác minh QID",
        "en_lead": "Wikipedia EN: phần giới thiệu",
        "en_plot": "Wikipedia EN: phần plot/premise",
        "en_lead_or_plot": "Wikipedia EN: giới thiệu hoặc plot/premise",
        "vi_sitelink": "Wikipedia VI: có sitelink",
        "vi_verified_article": "Wikipedia VI: bài đã xác minh QID",
        "vi_lead": "Wikipedia VI: phần giới thiệu",
        "vi_plot": "Wikipedia VI: phần cốt truyện",
        "vi_lead_or_plot": "Wikipedia VI: giới thiệu hoặc cốt truyện",
    }
    def cell(metric: dict) -> str:
        return f"{metric['movies']:,} ({metric['fraction'] * 100:.2f}%)" if metric["fraction"] is not None else "—"
    lines = ["# Kết quả độ phủ Wikidata và Wikipedia", "",
             f"Snapshot kết quả: {summary['updated_at']}. Stage: **{summary['stage']}**.", "",
             "Đọc toàn bộ movies.csv và cả train/valid/test; mẫu số là movieId duy nhất.", "",
             "| Phạm vi | Số phim | Wikidata | Wikipedia EN: có mô tả | Wikipedia EN: plot/premise | Wikipedia VI: có mô tả |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key in ("all_movielens", "train_valid_test_union", "train", "valid", "test"):
        data = scopes[key]
        values = [cell(data["coverage"][metric]) for metric in
                  ("wikidata_exact_unique", "en_lead_or_plot", "en_plot", "vi_lead_or_plot")]
        lines.append(f"| {labels[key]} | {data['catalog_movies']:,} | " + " | ".join(values) + " |")
    lines += ["", "Hợp ba split được tính từ cả ba file; mỗi movieId chỉ tính một lần."]
    if union["catalog_movies"] == scopes["train"]["catalog_movies"]:
        lines += ["Với dữ liệu hiện tại, mọi phim valid/test đều có trong train."]
    lines += ["", "## Chi tiết độ phủ", "",
              "| Tiêu chí | Catalog gốc | Hợp train/valid/test |", "| --- | ---: | ---: |"]
    for key, name in names.items():
        lines.append(f"| {name} | {cell(all_movies['coverage'][key])} | {cell(union['coverage'][key])} |")
    lines += ["", "## Wikidata metadata", "",
              "Đo sự hiện diện của truthy statement; chưa xác nhận chất lượng giá trị hoặc chuẩn hóa đơn vị.", "",
              "| Trường | Catalog gốc | Hợp train/valid/test |", "| --- | ---: | ---: |"]
    for key in PROPERTIES:
        lines.append(f"| {key} | {cell(all_movies['coverage']['wikidata_' + key])} | {cell(union['coverage']['wikidata_' + key])} |")
    lines += ["", "## Trạng thái nguồn", "", "| Nguồn/trạng thái | Catalog gốc | Hợp train/valid/test |",
              "| --- | ---: | ---: |"]
    for source in ("mapping", "en", "vi"):
        left = all_movies["mapping_status_counts"] if source == "mapping" else all_movies["article_status_counts"][source]
        right = union["mapping_status_counts"] if source == "mapping" else union["article_status_counts"][source]
        for status in sorted(set(left) | set(right)):
            lines.append(f"| {source}: {status} | {left.get(status, 0):,} | {right.get(status, 0):,} |")
    lines += ["", "## Cách đọc kết quả", "",
              "- Ghép chính xác qua IMDb P345 và chỉ chấp nhận một QID; không tìm theo title.",
              "- Wikipedia phải trả QID khớp nguồn Wikidata; các redirect sang franchise/bản khác được tách riêng.",
              f"- Mô tả/plot cần ít nhất **{summary['threshold_characters']} ký tự** sau làm sạch wikitext.",
              "- Plot/premise là nhóm heading được nhận diện; parser không mở rộng template hoặc dựng HTML.",
              "- Wikidata description là mô tả ngắn, không phải cốt truyện. Wikipedia không cung cấp corpus user review trong phép đo này.",
              "- Đây là độ phủ theo cách ghép IMDb ID và quy tắc trích xuất trên; bài thiếu P345 hoặc text trong template có thể bị bỏ sót.",
              "- Request lỗi hoặc response thiếu dữ liệu là unknown; không được coi là tác phẩm không có trên nguồn.", "",
              "[Dữ liệu từng phim](coverage.parquet), [manifest](manifest.json), "
              f"[phương pháp và tái lập]({relpath(ROOT / 'docs/wikimedia_coverage.md', directory)}).", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Measure all original movies, then expose train/valid/test and union coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/ml-20m.zip")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ml20m_lightgcn")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/wikimedia_coverage")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/movie_content/wikimedia_coverage")
    parser.add_argument("--mapping-batch", type=int, default=100)
    parser.add_argument("--article-batch", type=int, default=50)
    parser.add_argument("--rate", type=float, default=1)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--languages", nargs="+", choices=["en", "vi"], default=["en", "vi"])
    parser.add_argument("--mapping-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.rate <= 1 or not 1 <= args.article_batch <= 50 or not 1 <= args.mapping_batch <= 100:
        parser.error("rate must be <=1/s, article batch <=50, mapping batch <=100")
    if min(args.timeout, args.attempts, args.min_chars) <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("timeouts, attempts, thresholds, and limits must be positive")
    for name in ("archive", "data_dir", "raw_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir == args.data_dir or args.output_dir == ROOT / "data/processed/movie_content":
        parser.error("Use a separate measurement output directory")
    rows = catalog_rows(args.archive, args.data_dir)
    selected = rows[:args.limit] if args.limit else rows
    client = Client(args.raw_dir, args.rate, args.timeout, args.attempts, args.cache_only)
    indexed: dict[str, list[dict]] = {}
    for row in selected:
        if row["imdb_id"]:
            indexed.setdefault(row["imdb_id"], []).append(row)
        else:
            row["mapping_status"] = "no_imdb_id"
    keys = sorted(indexed)
    checkpoint(args, rows, client, "mapping")
    for offset in range(0, len(keys), args.mapping_batch):
        group = keys[offset:offset + args.mapping_batch]
        envelope = client.fetch("wikidata_mapping_v1", "https://query.wikidata.org/sparql",
                                {"query": mapping_query(group), "format": "json"})
        bindings = (envelope.get("response") or {}).get("results", {}).get("bindings", [])
        for key in group:
            for row in indexed[key]:
                apply_mapping(row, bindings, envelope)
        if offset % (args.mapping_batch * 10) == 0 or offset + args.mapping_batch >= len(keys):
            checkpoint(args, rows, client, "mapping")
            print(f"Wikidata: {min(offset + len(group), len(keys))}/{len(keys)} IMDb IDs; requests={dict(client.requests)}", flush=True)
    checkpoint(args, rows, client, "mapping_complete")
    if not args.mapping_only:
        for language in args.languages:
            by_title: dict[str, list[dict]] = {}
            for row in selected:
                url = row[language + "_url"]
                if url:
                    title = unquote(urlparse(url).path[6:]).replace("_", " ")
                    by_title.setdefault(title, []).append(row)
            titles = sorted(by_title)
            for offset in range(0, len(titles), args.article_batch):
                group = titles[offset:offset + args.article_batch]
                parameters = {"action": "query", "titles": "|".join(group), "prop": "revisions|pageprops",
                              "rvprop": "ids|content", "rvslots": "main", "ppprop": "wikibase_item",
                              "redirects": "1", "maxlag": "5", "format": "json", "formatversion": "2"}
                envelope = client.fetch(language + "wiki_revisions_v1", f"https://{language}.wikipedia.org/w/api.php", parameters)
                query = (envelope.get("response") or {}).get("query", {})
                # In multi-page latest-revision mode the normal batch has no continuation.
                # Treat a continued response conservatively until its remaining data is fetched.
                if envelope["outcome"] == "success" and envelope["response"].get("continue"):
                    pages = {page["title"]: page for page in query.get("pages", [])}
                    response = envelope["response"]
                    raw_paths = [envelope["raw_path"]]
                    for continuation_index in range(20):
                        continued = client.fetch(language + "wiki_revisions_v1", f"https://{language}.wikipedia.org/w/api.php",
                                                 {**parameters, **response["continue"]})
                        if continued["outcome"] != "success":
                            envelope = continued
                            break
                        raw_paths.append(continued["raw_path"])
                        response = continued["response"]
                        addition = response.get("query", {})
                        for page in addition.get("pages", []):
                            previous = pages.get(page["title"], {})
                            revisions = previous.get("revisions", []) + page.get("revisions", [])
                            pages[page["title"]] = {**previous, **page, "revisions": revisions}
                        for key in ("normalized", "converted", "redirects"):
                            query[key] = query.get(key, []) + addition.get(key, [])
                        if not response.get("continue"):
                            break
                    else:
                        envelope = {**envelope, "outcome": "response_incomplete"}
                    query["pages"] = list(pages.values())
                    envelope = {**envelope, "raw_paths": raw_paths}
                for title in group:
                    for row in by_title[title]:
                        apply_article(row, language, query, envelope)
                if offset % (args.article_batch * 10) == 0 or offset + args.article_batch >= len(titles):
                    checkpoint(args, rows, client, language + "wiki")
                    print(f"{language}wiki: {min(offset + len(group), len(titles))}/{len(titles)} articles; requests={dict(client.requests)}", flush=True)
    incomplete = any(row["mapping_status"] in {"not_attempted", "not_cached", "request_error"} for row in selected)
    if not args.mapping_only:
        incomplete |= any(row[language + "_status"] in {"not_attempted", "not_cached", "request_error", "response_incomplete"}
                          for row in selected for language in args.languages)
    stage = "incomplete" if incomplete else "partial" if args.limit else "mapping_complete" if args.mapping_only else "complete"
    summary = checkpoint(args, rows, client, stage)
    write_report(summary, args.output_dir)
    print(json.dumps({scope: {"movies": data["catalog_movies"], "coverage": {key: value for key, value in data["coverage"].items()
                           if key in {"wikidata_exact_unique", "en_sitelink", "vi_sitelink", "en_lead_or_plot", "en_plot", "vi_lead_or_plot", "vi_plot"}}}
                      for scope, data in summary["scopes"].items()}, ensure_ascii=False, indent=2), flush=True)
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
