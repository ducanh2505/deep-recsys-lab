"""Focused offline checks for source identity and narrative coverage decisions."""
import unittest

from scripts.measure_wikimedia_coverage import (
    PROPERTIES, apply_article, apply_mapping, coverage_summary, narrative_sections, resolve_title,
)


class WikimediaCoverageTests(unittest.TestCase):
    def test_plot_keeps_subsections_but_excludes_later_sections(self) -> None:
        lead, plot, section = narrative_sections("{{Infobox|nested={{x}}}}\nA film.\n== Plot ==\nThe hero leaves.\n"
                                                  "=== Ending ===\nShe returns.\n== Cast ==\nActor names.")
        self.assertEqual(lead, "A film.")
        self.assertEqual(plot, "The hero leaves. She returns.")
        self.assertEqual(section, "Plot")

    def test_empty_markup_is_not_narrative_and_vietnamese_heading_works(self) -> None:
        self.assertEqual(narrative_sections("== Plot ==\n{{Empty}}<ref>citation</ref>[[File:image.jpg]]")[1], "")
        self.assertEqual(narrative_sections("Giới thiệu.\n== Nội dung ==\nMột câu chuyện.\n== Diễn viên ==\nA.")[1],
                         "Một câu chuyện.")

    def test_reused_references_do_not_consume_following_prose_or_template_closures(self) -> None:
        source = ('{{Infobox|country={{Plainlist|France<ref name="source" />}}}}\n'
                  'A French film.<ref name="source">{{Cite web|title=Reference}}</ref> '
                  'The lead continues.\n== Plot ==\n'
                  'The hero leaves.<ref name="source"/> She returns.<ref>citation</ref>')
        lead, plot, _ = narrative_sections(source)
        self.assertEqual(lead, "A French film. The lead continues.")
        self.assertEqual(plot, "The hero leaves. She returns.")

    def test_exact_mapping_does_not_pick_an_ambiguous_or_wrong_imdb(self) -> None:
        row = {"imdb_id": "tt0000001"}
        bindings = [{"imdb": {"value": "tt0000001"}, "item": {"value": "http://www.wikidata.org/entity/" + qid}}
                    for qid in ("Q1", "Q2")]
        apply_mapping(row, bindings, {"outcome": "success"})
        self.assertEqual(row["mapping_status"], "ambiguous")
        row = {"imdb_id": "tt0000002"}
        apply_mapping(row, bindings, {"outcome": "success"})
        self.assertEqual(row["mapping_status"], "not_found")
        apply_mapping(row, [], {"outcome": "request_error"})
        self.assertEqual(row["mapping_status"], "request_error")

    def test_redirect_to_franchise_does_not_count_as_movie_content(self) -> None:
        row = {"wikidata_id": "Q222939", "en_url": "https://en.wikipedia.org/wiki/Jumanji_(film)"}
        query = {"redirects": [{"from": "Jumanji (film)", "to": "Jumanji"}], "pages": [
            {"title": "Jumanji", "ns": 0, "pageid": 1, "pageprops": {"wikibase_item": "Q30684088"},
             "revisions": [{"revid": 2, "slots": {"main": {"content": "Lots of text about a franchise."}}}]}]}
        apply_article(row, "en", query, {"outcome": "success"})
        self.assertEqual(row["en_status"], "mapping_mismatch")
        self.assertNotIn("en_lead_chars", row)

    def test_verified_redirect_resolves_and_missing_response_stays_unknown(self) -> None:
        query = {"normalized": [{"from": "film_name", "to": "Film name"}],
                 "redirects": [{"from": "Film name", "to": "Film title"}]}
        self.assertEqual(resolve_title("film_name", query), "Film title")
        row = {"wikidata_id": "Q1", "en_url": "https://en.wikipedia.org/wiki/Film_name"}
        apply_article(row, "en", query, {"outcome": "success"})
        self.assertEqual(row["en_status"], "response_incomplete")

    def test_split_union_counts_shared_movies_once_and_includes_test_only_movies(self) -> None:
        rows = []
        for train, valid, test in [(True, True, False), (False, False, True), (False, False, False)]:
            row = {"in_train": train, "in_valid": valid, "in_test": test, "mapping_status": "matched",
                   "wikidata_description_en": "film", **{"wd_" + field: False for field in PROPERTIES}}
            for language in ("en", "vi"):
                row.update({language + "_status": "verified", language + "_url": "url",
                            language + "_lead_chars": 150, language + "_plot_chars": 0})
            rows.append(row)
        summary = coverage_summary(rows, 100)
        self.assertEqual(summary["all_movielens"]["catalog_movies"], 3)
        self.assertEqual(summary["train_valid_test_union"]["catalog_movies"], 2)
        self.assertEqual(summary["test"]["catalog_movies"], 1)
        self.assertEqual(summary["train_valid_test_union"]["coverage"]["en_lead_or_plot"]["movies"], 2)


if __name__ == "__main__":
    unittest.main()
