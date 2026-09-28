from __future__ import annotations

import pytest

from deep_recsys_lifecycle.models import Candidate
from deep_recsys_lifecycle.paper_diagnostics import PaperGoldDiagnostics


def _candidate(movie_id: int, rank: int) -> Candidate:
    return Candidate(movie_id=movie_id, score=float(10 - rank), rank=rank)


def test_gold_loss_partitions_raw_heldout_and_counts_queries_separately() -> None:
    diagnostic = PaperGoldDiagnostics(
        mode="history_only",
        source_names=("popularity", "itemknn"),
        training_catalog=frozenset({1, 2, 3, 4, 5}),
        training_popularity={1: 50, 2: 0, 3: 1, 4: 10, 5: 100},
    )
    diagnostic.observe(
        raw_gold_movie_ids=(1, 2, 3, 4, 5, 99),
        eligible_gold_movie_ids=(1, 2, 3, 4, 5),
        history=(),
        pools={
            "popularity": (_candidate(3, 1), _candidate(4, 2)),
            "itemknn": (_candidate(4, 1), _candidate(5, 2)),
        },
        fusion_input_movie_ids=(3, 4),
        final_ranking=(_candidate(4, 1),),
    )
    # A Subject with only cold held-out Movies contributes to exclusions, not Recall.
    diagnostic.observe(
        raw_gold_movie_ids=(98,),
        eligible_gold_movie_ids=(),
        history=(1,),
        pools={"popularity": (), "itemknn": ()},
        fusion_input_movie_ids=(),
        final_ranking=(),
    )
    report = diagnostic.finish()

    assert report["raw_held_out_gold_movie_occurrences"] == 7
    assert report["eligible_gold_movie_occurrences"] == 5
    assert report["evaluated_query_count"] == 1
    categories = report["categories"]
    assert {name: value["gold_movie_occurrences"] for name, value in categories.items()} == {
        "outside_training_catalog": 2,
        "without_training_interaction": 1,
        "outside_retriever_pools": 1,
        "lost_before_fusion": 1,
        "below_final_top_100": 1,
        "recovered_top_100": 1,
    }
    assert categories["outside_training_catalog"]["query_count"] == 2
    assert report["largest_addressable_failure_category"] == "without_training_interaction"
    source_contributions = report["source_contributions"]
    assert source_contributions["popularity"]["exclusive_union_gold_movie_occurrences"] == 1
    assert source_contributions["itemknn"]["exclusive_union_gold_movie_occurrences"] == 1
    assert report["history_length_segments"]["0"]["macro_Recall@100"] == pytest.approx(0.2)
    assert report["training_item_popularity_segments"]["0"]["gold_movie_occurrences"] == 1


def test_gold_loss_rejects_fabricated_candidates_and_denominator_changes() -> None:
    diagnostic = PaperGoldDiagnostics(
        mode="known_user",
        source_names=("popularity",),
        training_catalog=frozenset({1}),
        training_popularity={1: 1},
    )
    with pytest.raises(ValueError, match="eligible Gold Set"):
        diagnostic.observe(
            raw_gold_movie_ids=(1,),
            eligible_gold_movie_ids=(),
            history=(),
            pools={"popularity": ()},
            fusion_input_movie_ids=(),
            final_ranking=(),
        )
    with pytest.raises(ValueError, match="natural retriever pools"):
        diagnostic.observe(
            raw_gold_movie_ids=(1,),
            eligible_gold_movie_ids=(1,),
            history=(),
            pools={"popularity": ()},
            fusion_input_movie_ids=(1,),
            final_ranking=(_candidate(1, 1),),
        )
