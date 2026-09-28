"""Bounded-memory gold-loss diagnostics for the two paper evaluation protocols."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

from .models import Candidate

LOSS_CATEGORIES = (
    "outside_training_catalog",
    "without_training_interaction",
    "outside_retriever_pools",
    "lost_before_fusion",
    "below_final_top_100",
    "recovered_top_100",
)
ADDRESSABLE_CATEGORIES = (
    "without_training_interaction",
    "outside_retriever_pools",
    "lost_before_fusion",
    "below_final_top_100",
)


def _history_segment(length: int) -> str:
    if length == 0:
        return "0"
    if length < 5:
        return "1-4"
    if length < 20:
        return "5-19"
    if length < 100:
        return "20-99"
    if length < 500:
        return "100-499"
    return "500+"


def _popularity_segment(count: int) -> str:
    if count == 0:
        return "0"
    if count < 10:
        return "1-9"
    if count < 100:
        return "10-99"
    if count < 1000:
        return "100-999"
    return "1000+"


@dataclass(slots=True)
class _CategoryCount:
    movie_occurrences: int = 0
    query_count: int = 0
    movie_ids: set[int] = field(default_factory=set)

    def add(self, movie_ids: Collection[int]) -> None:
        if movie_ids:
            self.movie_occurrences += len(movie_ids)
            self.query_count += 1
            self.movie_ids.update(movie_ids)

    def to_dict(self) -> dict[str, int]:
        return {
            "gold_movie_occurrences": self.movie_occurrences,
            "query_count": self.query_count,
            "unique_movie_count": len(self.movie_ids),
        }


@dataclass(slots=True)
class _SegmentCount:
    query_count: int = 0
    gold_movie_occurrences: int = 0
    final_top_100_hits: int = 0
    recall_sum: float = 0.0
    category_occurrences: Counter[str] = field(default_factory=Counter)

    def to_dict(self, *, query_segment: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "gold_movie_occurrences": self.gold_movie_occurrences,
            "final_top_100_hits": self.final_top_100_hits,
            "loss_category_gold_movie_occurrences": {
                name: self.category_occurrences[name] for name in LOSS_CATEGORIES
            },
        }
        if query_segment:
            result["query_count"] = self.query_count
            result["macro_Recall@100"] = (
                self.recall_sum / self.query_count if self.query_count else 0.0
            )
        else:
            result["gold_hit_fraction@100"] = (
                self.final_top_100_hits / self.gold_movie_occurrences
                if self.gold_movie_occurrences
                else 0.0
            )
        return result


@dataclass(slots=True)
class PaperGoldDiagnostics:
    """Accumulate one held-out cohort without retaining per-Query Candidate pools."""

    mode: str
    source_names: tuple[str, ...]
    training_catalog: frozenset[int]
    training_popularity: Mapping[int, int]
    raw_gold_movie_occurrences: int = 0
    eligible_gold_movie_occurrences: int = 0
    evaluated_query_count: int = 0
    categories: dict[str, _CategoryCount] = field(init=False)
    histories: dict[str, _SegmentCount] = field(default_factory=dict)
    popularities: dict[str, _SegmentCount] = field(default_factory=dict)
    source_hits_top_100: Counter[str] = field(default_factory=Counter)
    source_hits_pool: Counter[str] = field(default_factory=Counter)
    source_exclusive_hits: Counter[str] = field(default_factory=Counter)
    source_exclusive_queries: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if self.mode not in {"known_user", "history_only"}:
            raise ValueError(f"unsupported paper Query mode: {self.mode}")
        self.categories = {name: _CategoryCount() for name in LOSS_CATEGORIES}

    def observe(
        self,
        *,
        raw_gold_movie_ids: Sequence[int],
        eligible_gold_movie_ids: Sequence[int],
        history: Sequence[int],
        pools: Mapping[str, Sequence[Candidate]],
        fusion_input_movie_ids: Collection[int],
        final_ranking: Sequence[Candidate],
    ) -> None:
        raw_gold = set(raw_gold_movie_ids)
        eligible_gold = set(eligible_gold_movie_ids)
        if len(raw_gold) != len(raw_gold_movie_ids) or len(eligible_gold) != len(
            eligible_gold_movie_ids
        ):
            raise ValueError("Gold Sets must contain distinct Movie IDs")
        if eligible_gold != raw_gold & self.training_catalog:
            raise ValueError(
                "eligible Gold Set must contain exactly the catalog-visible held-out Movies"
            )
        if eligible_gold & set(history):
            raise ValueError("fold-in or training history overlaps the eligible Gold Set")
        if set(pools) != set(self.source_names):
            raise ValueError("source pools do not match the Query mode retriever bank")

        self.raw_gold_movie_occurrences += len(raw_gold)
        self.eligible_gold_movie_occurrences += len(eligible_gold)
        counts_by_category: dict[str, set[int]] = {name: set() for name in LOSS_CATEGORIES}
        counts_by_category["outside_training_catalog"] = raw_gold - eligible_gold
        pool_ids_by_source = {
            name: {candidate.movie_id for candidate in pools[name]}
            for name in self.source_names
        }
        top_100_ids_by_source = {
            name: {candidate.movie_id for candidate in pools[name][:100]}
            for name in self.source_names
        }
        diagnostic_union = set().union(*pool_ids_by_source.values())
        input_ids = set(fusion_input_movie_ids)
        if not input_ids <= diagnostic_union:
            raise ValueError("fusion input contains a Movie outside natural retriever pools")
        final_ranks = {candidate.movie_id: rank for rank, candidate in enumerate(final_ranking, 1)}
        if not set(final_ranks) <= input_ids:
            raise ValueError("final ranking contains a Movie outside the fusion input")

        for movie_id in eligible_gold:
            evidence = self.training_popularity.get(movie_id, 0)
            if evidence <= 0:
                category = "without_training_interaction"
            elif movie_id not in diagnostic_union:
                category = "outside_retriever_pools"
            elif movie_id not in input_ids:
                category = "lost_before_fusion"
            elif movie_id not in final_ranks:
                category = "below_final_top_100"
            else:
                category = "recovered_top_100"
            counts_by_category[category].add(movie_id)
            popularity = self.popularities.setdefault(
                _popularity_segment(evidence), _SegmentCount()
            )
            popularity.gold_movie_occurrences += 1
            popularity.category_occurrences[category] += 1
            popularity.final_top_100_hits += int(category == "recovered_top_100")

        for name in self.source_names:
            source_hits = eligible_gold & pool_ids_by_source[name]
            self.source_hits_pool[name] += len(source_hits)
            self.source_hits_top_100[name] += len(eligible_gold & top_100_ids_by_source[name])
            other_ids = set().union(
                *(pool_ids_by_source[other] for other in self.source_names if other != name)
            )
            exclusive = source_hits - other_ids
            self.source_exclusive_hits[name] += len(exclusive)
            self.source_exclusive_queries[name] += int(bool(exclusive))

        for name, ids in counts_by_category.items():
            self.categories[name].add(ids)
        if eligible_gold:
            self.evaluated_query_count += 1
            history_segment = self.histories.setdefault(
                _history_segment(len(set(history))), _SegmentCount()
            )
            history_segment.query_count += 1
            history_segment.gold_movie_occurrences += len(eligible_gold)
            hits = len(counts_by_category["recovered_top_100"])
            history_segment.final_top_100_hits += hits
            denominator = (
                len(eligible_gold)
                if self.mode == "known_user"
                else min(100, len(eligible_gold))
            )
            history_segment.recall_sum += hits / denominator
            for name, ids in counts_by_category.items():
                history_segment.category_occurrences[name] += len(ids)

    def finish(self) -> dict[str, object]:
        if sum(item.movie_occurrences for item in self.categories.values()) != (
            self.raw_gold_movie_occurrences
        ):
            raise RuntimeError("gold-loss categories do not partition raw held-out Movies")
        largest = max(
            ADDRESSABLE_CATEGORIES,
            key=lambda name: (
                self.categories[name].movie_occurrences,
                -ADDRESSABLE_CATEGORIES.index(name),
            ),
        )
        return {
            "query_mode": self.mode,
            "raw_held_out_gold_movie_occurrences": self.raw_gold_movie_occurrences,
            "eligible_gold_movie_occurrences": self.eligible_gold_movie_occurrences,
            "evaluated_query_count": self.evaluated_query_count,
            "category_definitions": {
                "outside_training_catalog": (
                    "raw held-out Movie absent from the train-visible catalog; "
                    "excluded before Recall denominator"
                ),
                "without_training_interaction": (
                    "eligible Movie with zero positive training interactions"
                ),
                "outside_retriever_pools": (
                    "eligible Movie absent from every configured source pool"
                ),
                "lost_before_fusion": "in diagnostic source union but outside actual LHF input",
                "below_final_top_100": "in LHF input but omitted from the final Top-100",
                "recovered_top_100": "present in the final LHF Top-100",
            },
            "categories": {name: item.to_dict() for name, item in self.categories.items()},
            "largest_addressable_failure_category": largest,
            "source_contributions": {
                name: {
                    "top_100_gold_movie_occurrences": self.source_hits_top_100[name],
                    "full_pool_gold_movie_occurrences": self.source_hits_pool[name],
                    "exclusive_union_gold_movie_occurrences": self.source_exclusive_hits[name],
                    "queries_with_exclusive_union_gold": self.source_exclusive_queries[name],
                }
                for name in self.source_names
            },
            "history_length_segments": {
                name: item.to_dict(query_segment=True)
                for name, item in sorted(self.histories.items())
            },
            "training_item_popularity_segments": {
                name: item.to_dict(query_segment=False)
                for name, item in sorted(self.popularities.items())
            },
            "segment_definitions": {
                "history_length": ["0", "1-4", "5-19", "20-99", "100-499", "500+"],
                "training_item_popularity": ["0", "1-9", "10-99", "100-999", "1000+"],
                "counts": (
                    "Gold Movie occurrences are Subject-Movie pairs; Query counts may "
                    "overlap across loss categories"
                ),
            },
        }
