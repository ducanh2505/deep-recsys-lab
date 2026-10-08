# Movie catalog metadata collection

`scripts/crawl_movie_metadata.py` enriches the existing training catalog with English
movie content. It uses MovieLens metadata from the local ZIP, TMDB as the main
provider, and Wikidata/English Wikipedia for missing descriptions or important
attributes. It does not change ratings, train/validation/test splits, or model
artifacts, and it does not build embeddings.

## Run

The existing Python environment, Polars, and Python's standard library are enough;
there are no additional dependencies. Paths default to the repository root even
when the script is invoked from another directory.

Set `TMDB_READ_ACCESS_TOKEN` in the process environment, or put its literal value
in the repository's ignored `.env` file. Environment values take precedence. The
script supports optional `export` and quoted values in this file, without shell
expansion; it reads only the requested token. Tokens are never included in request
parameters, cached responses, manifests, or logs. Do not commit the `.env` file.

```bash
uv run python scripts/crawl_movie_metadata.py --limit 30
uv run python scripts/crawl_movie_metadata.py
```

To select original MovieLens IDs explicitly:

```bash
uv run python scripts/crawl_movie_metadata.py --movie-ids 1 142 690 720 730 1533 3799 8795 27528 91752
```

Every output catalog retains **all 11,508 training movies**, even during a limited
run. Unselected movies retain MovieLens metadata and any previously cached
enrichment; their missing values are null. A partial run is therefore not a
completed collection of the entire catalog. The manifest separates selected
items, attempted items, and coverage across the full catalog.

Rerunning the normal command reuses successful responses and permanent 404s,
while retrying transient failures. `--refresh` re-fetches selected movies and their
needed fallback data. To rebuild using only local cached responses:

```bash
uv run python scripts/crawl_movie_metadata.py --rebuild-only
```

If a refresh fails, a previous successful raw response is retained; the failed
attempt is saved separately so previously populated field provenance remains
resolvable.

Other options are `--data-dir`, `--archive`, `--raw-dir`, `--output-dir`,
`--env-file`, `--workers`, `--checkpoint-every`, `--timeout`, `--attempts`,
`--tmdb-rate`, and `--wikimedia-rate`. The default rates are maximum **5 TMDB
requests/second** and **1 Wikimedia request/second**, shared across workers and
Wikidata/Wikipedia endpoints. Transient errors get at most five total attempts,
with backoff and `Retry-After`. TMDB authentication failures stop subsequent
requests. An interrupted run keeps atomic raw responses and checkpoints; the
manifest records its stage.

## Mapping and source selection

- The catalog is exactly the unique original `movieId` values in `train.parquet`,
  checked against the input manifest. `movies.csv` and `links.csv` are read inside
  `data/raw/ml-20m.zip`; ratings do not need to be extracted or prepared again.
- IMDb IDs are strings such as `tt0114709`. The input has 24 movies without TMDB
  IDs, and three TMDB IDs shared by pairs of MovieLens movies.
- A TMDB detail response is checked against the expected IMDb ID when the
  response supplies one. Missing/404/mismatched mappings use TMDB's exact IMDb
  find endpoint. Multiple candidates are marked ambiguous. TV-only matches are
  reported as unsupported media rather than silently attached as movies. A
  response without an IMDb ID is marked unverified.
  For unresolved records, `tmdb_id` can remain the original MovieLens link;
  mapping status and provenance distinguish these links from resolved metadata.
- TMDB is requested with `language=en-US` and
  `append_to_response=credits,keywords`. MovieLens genres are retained separately
  from TMDB genres. TMDB genres take precedence when present.
- Wikidata is queried for movies missing an overview, genres, cast, directors,
  runtime, original language, or production countries. The absence of keywords
  alone does not trigger fallback: Wikidata is not treated as a TMDB keyword
  replacement. IMDb matching is exact; ambiguous entities are not merged.
- If the Wikidata Query Service is rate-limited during an outage, the crawler
  stops using that endpoint for the run and uses the independent Wikidata Action
  API statement search. The matched entity's IMDb claim is verified before
  attaching content; its English Wikipedia sitelink is read from the entity.
- Missing structured values can be supplied by Wikidata claims. Referenced people
  retain their Wikidata IDs; they are not mixed with TMDB person IDs. Original
  language is a two-letter ISO code when available, and runtime is accepted only
  from quantities with a minutes unit. Country identifiers are provider-specific,
  with the provider included in each value.
- For a missing TMDB overview, the English Wikipedia sitelink identifies the
  article. The script extracts Plot/Synopsis/Plot summary, then the introduction
  if no usable plot exists. It strips tables, references, and edit controls.
  Wikipedia content, section, URL, and revision are retained for attribution.
- `overview` uses TMDB first and Wikipedia as fallback. `tmdb_overview`,
  `wikipedia_plot`, and `wikipedia_intro` remain separate so later experiments can
  distinguish short descriptions from long plot text. `overview_source` records
  the chosen provider.

## Files and schemas

Raw responses are gzip JSON envelopes under:

```text
data/raw/movie_metadata/<provider>/en-US/<request-kind>/<external-id-or-hash>.json.gz
```

Each envelope contains the endpoint, non-secret parameters, original JSON
response, timestamp, HTTP status, outcome, and request attempts. Wikidata batch
responses are retained in full and indexed per requested ID; empty results are
also cached to avoid repeated lookups. Shared external IDs reuse one response.
Service-disabled entries have zero attempts and no API response; the actual
Action API fallback responses are stored in their own cache files.

Normalized output is under `data/processed/movie_content/`:

| File | Contents |
| --- | --- |
| `catalog.parquet` | One sorted row per original `movieId`, including baseline, enriched fields, mapping status and Wikipedia link. |
| `provenance.parquet` | One row per populated movie/field, with source, external ID, language, fetch timestamp, raw path, source URL and revision where available. |
| `crawl_status.parquet` | One row per movie/provider with status, external ID, attempts for the referenced cached request, last attempt time, HTTP code and error. |
| `manifest.json` | Input paths, catalog ID fingerprint, run configuration/stage, request/cache counters, status totals, field coverage, mapping counts and movies without descriptions. |

Catalog scalar fields include `movieId`, `imdb_id`, `tmdb_id`, `wikidata_id`,
titles, release year/date, runtime, language, overview variants, tagline,
collection ID/name and mapping status. Dates are ISO strings, and absent values
are null rather than fabricated zeroes or empty descriptions.

`genres_movielens`, `genres_tmdb`, and `genres` are string lists. `keywords`,
`production_countries`, and `spoken_languages` are lists of `{id, name, source}`.
`cast`, `directors`, and `writers` are lists of `{id, name, source, order}`;
TMDB cast order is preserved, whereas Wikidata order is null. Source-specific IDs
must always be interpreted together with their source.

Raw files are saved after each request; catalog checkpoints are saved after every
100 TMDB completions and Wikimedia movies by default. Each file is replaced
atomically; the manifest is written last. These separate files are not a database
transaction: after a hard process kill, rerun/rebuild to reconcile the outputs.

## Inspect results

No unit tests or new test framework are introduced. Validate real output after a
pilot that includes popular movies, missing TMDB IDs, and shared TMDB IDs:

1. Read all three Parquet files and compare their schemas with the manifest.
2. Require 11,508 unique sorted `movieId` values, exactly matching the training
   item set; check that IMDb strings preserve leading zeroes.
3. Compare selected movies against raw detail/find responses, including identity,
   credits, source and missing values.
4. Recompute coverage/status totals and compare them with the manifest. Inspect
   every unresolved/mismatched mapping and movies without an overview.
5. Rerun from cache and require zero new requests for successful responses; a
   rebuild-only run must make zero network requests for every provider.

An HTTP 200, a mapping ID, or a `stage=complete` run does not imply every movie has
every field. Coverage and unresolved lists are the collection result. The
manifest counts network calls in the current run separately from cache reads.

## Sources and attribution

- [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/): observe the original
  `README.txt` research-use and redistribution terms; cite Harper and Konstan
  (2015), *The MovieLens Datasets: History and Context*.
- [TMDB](https://developer.themoviedb.org/docs/faq): non-commercial API use requires
  attribution, approved branding when exposed in an application, and the notice:
  **This product uses the TMDB API but is not endorsed or certified by TMDB.**
- [Wikidata](https://www.wikidata.org/wiki/Wikidata:Data_access): structured data is
  CC0; acknowledge the source and follow access etiquette.
- [Wikipedia](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use): preserve
  article/revision links and comply with the applicable text license, generally
  CC BY-SA. Wikipedia text is not assigned Wikidata's CC0 license.

MovieLens user tags and Tag Genome are not incorporated into this crawl. They
can be added as separate experiments; Genome is derived partly from ratings,
tags and reviews and should not be assumed independent of evaluation data.

See [the proposed enrichment data layout](movie_enrichment_data_layout.md) for
full-catalog snapshots, multilingual documents, reviews, semantic graph tables,
and feature exports aligned with model item IDs.
