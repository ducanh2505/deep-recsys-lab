# MovieLens 20M source contract

Status: research note, not an architectural decision
Reviewed: 2026-09-18

## Question

Which exact MovieLens 20M payload should the Lab mean by `movielens_20m`, what does its
publisher guarantee, and how should that payload map into the repository's Dataset contract?

This note uses only GroupLens first-party material. It separates publisher facts from the
recommended Lab mapping; the latter is a repository proposal, not a claim made by GroupLens.

## Conclusion

Bind the initial `movielens_20m` Dataset Source to the current official `ml-20m.zip` payload,
whose README was generated on 2016-10-17. Do not use the generic label `movielens`, because
the April 2015 release and the October 2016 update are distinguishable source payloads.

For the initial mapping:

```text
ratings.csv -> event-level Interaction Events
movies.csv  -> authoritative Candidate table for this source payload
```

Preserve each explicit rating and its UTC occurrence time. Treat the archive's physical row
order as storage order, not chronology. Use `movies.csv` rather than ratings-derived distinct
IDs as the Candidate Catalog source, so a movie represented only by a tag is not silently
lost. Retain the other archive members as source provenance but leave them outside the
initial Adaptation until a concrete Approach requires them.

The publisher supplies an MD5 checksum for the archive. The Lab should verify that MD5 and
also compute and record its own SHA-256. Those two values must be labelled separately: an
SHA-256 produced by the Lab is not an official GroupLens checksum.

## Official source identity and history

GroupLens calls the dataset **MovieLens 20M**, uses the archive key `ml-20m`, and distributes
it as [`ml-20m.zip`](https://grouplens.org/datasets/movielens/20m/). GroupLens announced the
initial release on 2015-04-06
([release announcement](https://grouplens.org/blog/new-movielens-datasets-released/)). The
current landing page describes the dataset as released in April 2015 and updated in October
2016 to update `links.csv` and add Tag Genome data
([MovieLens 20M landing page](https://grouplens.org/datasets/movielens/20m/)).

The current archive README says it was generated on 2016-10-17
([current README](https://files.grouplens.org/datasets/movielens/ml-20m-README.html)). The
archived original README says it was generated on 2015-03-31
([April 2015 README](https://files.grouplens.org/datasets/movielens/old/ml-20m-README-4-2015.html)).
These are not safely interchangeable payload identities even though both are called
MovieLens 20M.

GroupLens publishes this MD5 in its official sidecar:

```text
archive: ml-20m.zip
publisher checksum algorithm: MD5
publisher checksum: cd245b17a1ae2cc31bb14903e1204af3
```

Source: [official `ml-20m.zip.md5` sidecar](https://files.grouplens.org/datasets/movielens/ml-20m.zip.md5).
No official SHA-256 was found in the reviewed first-party material. Any value stored under a
field such as `lab_computed_sha256` must therefore be described as computed by this Lab after
download, never as supplied or endorsed by GroupLens.

The current archive contains:

- `README.txt`
- `ratings.csv`
- `movies.csv`
- `tags.csv`
- `links.csv`
- `genome-scores.csv`
- `genome-tags.csv`

The [current README](https://files.grouplens.org/datasets/movielens/ml-20m-README.html)
reports 20,000,263 ratings and 465,564 tag applications across 27,278 movies and 138,493
users. It says the observations span 1995-01-09 through 2015-03-31, the included users were
selected from users who had rated at least 20 movies, identifiers were anonymized, and no
demographic information is included.

The dataset's requested academic citation is Harper and Konstan, “The MovieLens Datasets:
History and Context,” *ACM Transactions on Interactive Intelligent Systems* 5(4), Article 19
([official paper](https://files.grouplens.org/papers/harper-tiis2015.pdf),
[DOI](https://doi.org/10.1145/2827872)).

## Publisher-defined file semantics

### Ratings

`ratings.csv` has this header:

```text
userId,movieId,rating,timestamp
```

The README defines each row as one user's rating of one movie. Ratings range from 0.5 to 5.0
in half-star increments, and timestamps are Unix seconds since the UTC epoch. The physical
file is ordered first by `userId` and then by `movieId`; that order is therefore not a
chronological sequence and must not be used as one
([current README, ratings section](https://files.grouplens.org/datasets/movielens/ml-20m-README.html)).

The README does not formally specify CSV nullability or guarantee uniqueness of the
`(userId, movieId)` pair. An Adaptation should validate those properties against the exact
payload rather than promote assumptions into the source contract.

### Movies

`movies.csv` has this header:

```text
movieId,title,genres
```

The README describes one row per movie. The title is intended to include a release year but
may contain errors or inconsistencies. `genres` is a pipe-separated value drawn from the
README's controlled list, with `(no genres listed)` representing an absent classification.
The file includes movies having at least one rating **or** one tag. Consequently, a movie can
belong to `movies.csv` without appearing in `ratings.csv`
([current README, movies section](https://files.grouplens.org/datasets/movielens/ml-20m-README.html)).

That makes `movies.csv` an appropriate authority for the Candidate Catalog of this exact
source payload. It does not make MovieLens a universal inventory or availability authority,
and it does not imply that a movie remains commercially available.

Movie identifiers are intended to be consistent across `ratings.csv`, `tags.csv`,
`movies.csv`, and `links.csv`, but the Lab should still validate referential integrity while
adapting the downloaded bytes rather than silently dropping orphan references.

## Recommended Lab mapping

The following rules are repository recommendations, not GroupLens guarantees.

1. Use stable Source key `movielens_20m` and bind it to the current 2016-10 payload. Record
   the official archive name, README generation date, source URLs, citation, license terms,
   publisher MD5, and Lab-computed SHA-256 in Dataset provenance.
2. Keep downloaded archive bytes outside Git. Verify the publisher MD5 before Adaptation,
   compute SHA-256 over the exact downloaded archive, and use the stronger Lab checksum as
   part of exact local payload provenance.
3. Map every validated `ratings.csv` row to one Interaction Event:
   `userId -> subject_id`, `movieId -> candidate_id`, `rating -> value`, and
   `timestamp -> occurred_at` in UTC. Preserve the explicit 0.5–5.0 value; an implicit
   Approach may later apply an explicit Signal Preparation rule.
4. Map every validated `movies.csv` row to exactly one Candidate row. Use integer source IDs
   losslessly; do not renumber them merely to create dense model indices. Dense indices are
   Artifact-local implementation state, not source identity.
5. Ignore physical CSV row order for domain chronology. Use only the declared timestamp for
   temporal Protocols, and subject equal-resolution timestamps to the Lab's chronology rules.
6. Validate required headers, parseability, finite/range-conforming ratings, unique Candidate
   rows, and event-to-Candidate references. Fail Adaptation on violations; do not repair or
   discard rows silently.
7. If a content-based Variant is introduced, an initial declared `candidate_text` may be
   derived deterministically from `title` and `genres`. The exact transform belongs to that
   Dataset Adaptation/Binding; arbitrary concatenation of every available file is not a
   default.
8. Leave `tags.csv`, `links.csv`, and the two Tag Genome files unused by the initial
   Adaptation. Add them only when a concrete research use case defines their semantics,
   leakage boundary, and validation rules.

This mapping deliberately retains tag-only movies in the Candidate Catalog. Such Candidates
may have no Fit rating evidence and can receive tied/default scores from collaborative
Approaches; that is a model capability result, not grounds for deleting them from the source
Catalog.

## License and redistribution boundary

The MovieLens README contains custom research-use conditions rather than an unrestricted
data license. It requires acknowledgement, prohibits redistribution without separate
permission, requires permission for commercial or revenue-bearing use, disclaims warranties,
and prohibits implying GroupLens endorsement
([current README, usage license](https://files.grouplens.org/datasets/movielens/ml-20m-README.html),
[MovieLens datasets page](https://www.grouplens.org/datasets/movielens/)).

Therefore the repository may version the Adaptation code, schema, provenance template, and
download instructions, but it should not commit or redistribute the MovieLens archive or
derived records without separately establishing permission. This note is not legal advice;
the exact README shipped with the bound payload remains the controlling first-party text for
the Lab's use assessment.

## Uncertainties and explicit non-claims

- The reviewed official material publishes MD5 but no SHA-256. A Lab SHA-256 verifies local
  byte identity after it is first computed; it is not a publisher attestation.
- The README does not formally guarantee nullability or `(userId, movieId)` uniqueness.
  Validation is required before relying on either property.
- A title's embedded year and genre labels may contain source errors. The Lab should preserve
  the source payload rather than silently “correct” it during canonical Adaptation.
- `movies.csv` is authoritative for the Candidate Catalog of the bound MovieLens 20M
  payload, not for real-world availability, completeness, or current movie metadata.
- The ≥20-ratings user-selection rule means this is not an unbiased sample of all MovieLens
  users. Experiments must not generalize cold-Subject behavior from this source without a
  separate study design.
- The 2015 and 2016 payloads need separate identities if both are ever used. Matching the
  human-readable dataset name is insufficient evidence that their bytes or schemas match.
