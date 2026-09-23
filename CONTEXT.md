# Movie Recommender Lifecycle Showcase

A local-first ML system that demonstrates how movie preference events become evaluated and
served recommendations. Its language distinguishes ingestion, retrieval, ranking, temporal
evidence, and serving so that quality claims remain precise.

## Signals and time

**Rating Event**:
An immutable observation that a Subject assigned a numeric MovieLens rating to a Movie at an
Event Time.
_Avoid_: Rating row, Feedback, Interaction

**Positive Interaction**:
A binary preference signal derived from a Rating Event whose rating is at least 4.0. Ratings
below the threshold are absent signals, not negative preferences.
_Avoid_: Positive rating, Click, Like

**Event Store**:
The append-only collection of consumed Rating Events from which every Data Snapshot is
materialized.
_Avoid_: Ratings table, Training database, Kafka log

**Data Snapshot**:
The deduplicated set of Rating Events available at one declared temporal cutoff, identified
by its cumulative percentage of the source event stream.
_Avoid_: Dataset split, Batch, Checkpoint

**Future Window**:
The next ten percent of chronologically ordered Rating Events withheld from a Data Snapshot
and used as its evaluation evidence before being ingested.
_Avoid_: Test set, Next batch, Holdout dataset

**Lifecycle Stage**:
One ingest, materialize, fit, evaluate, and artifact-production cycle associated with a Data
Snapshot.
_Avoid_: Epoch, Experiment, Pipeline step

## Recommendation

**Subject**:
The MovieLens user whose observed Positive Interactions define a recommendation context.
_Avoid_: Customer, Account, Viewer

**Candidate**:
A Movie eligible to be recommended for a Query after previously observed Movies have been
excluded.
_Avoid_: Item when referring specifically to a recommendable movie

**Candidate Retriever**:
An approach that searches the available Candidate Catalog and returns an ordered Candidate
Pool for a Query.
_Avoid_: Model when referring to retrieval behaviour, Recommender

**Candidate Pool**:
An ordered, bounded set of Candidates produced by one Candidate Retriever or by Learned
Hybrid Fusion.
_Avoid_: Recommendation list, Result set

**Candidate Catalog**:
The Movies whose metadata is available at a Query's temporal boundary and that are eligible
for retrieval after history exclusion.
_Avoid_: Item universe, Movie database

**Learned Hybrid Fusion**:
A validation-trained gradient-boosted classifier that orders the union of multiple Candidate
Pools using retriever evidence and serving-time user and item context.
_Avoid_: Score averaging, Ensemble, LHF reranker

**Oracle Union**:
The unbudgeted union of all constituent Candidate Pools, used only as a diagnostic ceiling
for retriever complementarity.
_Avoid_: Oracle model, Fused pool

## Queries and serving

**Known-User Query**:
A Query identified by a known Subject whose stored Positive Interactions form its history.
_Avoid_: User query, Personalized request

**History-Only Query**:
A Query containing Movie identifiers as preference history without relying on a known Subject
identity.
_Avoid_: Anonymous query, Session query

**Empty-History Query**:
A Query with no usable stored or supplied Positive Interactions, served by global popularity.
_Avoid_: Cold-start query when the more precise term is available

**Serving Artifact**:
The immutable, self-describing bundle of fitted retrieval and fusion state loaded by the
recommendation API for one Data Snapshot.
_Avoid_: Model file, Checkpoint, Deployment

## Evaluation

**Gold Candidate**:
The first Positive Interaction for a Subject in the Future Window, used as the target of one
evaluation Query.
_Avoid_: Ground-truth item, Positive sample

**Empty-History Evaluation Cohort**:
Subjects with a Positive Interaction in the Future Window and no Positive Interactions in the
Data Snapshot. Their first Future Window Positive Interaction is the Gold Candidate, and the
served Query has no Subject identity or preference history.
_Avoid_: Masked existing-user cohort

**Retrieval Coverage**:
The share of evaluation Queries for which the Gold Candidate appears in a bounded Candidate
Pool.
_Avoid_: Accuracy, Recall when discussing the retrieval ceiling

**Conditional Ranking Success**:
The share of covered evaluation Queries for which the Gold Candidate is placed in the final
Top-N output.
_Avoid_: Conditional accuracy

**User Cold Start**:
A Query regime in which little or no Subject history is available; it includes low-history,
history-only, and empty-history cases with explicitly different serving behaviour.
_Avoid_: Cold start when item cold start could be inferred

**Interaction-New Candidate**:
A Candidate with no Positive Interactions in the current Data Snapshot, regardless of whether
its metadata is already in the Candidate Catalog.
_Avoid_: New movie, Cold-start item
