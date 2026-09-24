"""Tests for the pure evaluation metrics and for the integrity of ``evaluation/dataset.json``."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.metrics import (
    AbstentionOutcome,
    abstention_metrics,
    answer_correctness,
    citation_page_accuracy,
    citation_validity,
    forbidden_terms_found,
    hit_at_k,
    keyword_coverage,
    latency_summary,
    mrr,
    normalize_for_matching,
    percentile,
    recall_at_k,
)
from app.models.schemas import Citation

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "evaluation" / "dataset.json"
MANIFEST_PATH = REPO_ROOT / "sample_data" / "manifest.json"

A1 = ("a.pdf", 1)
A2 = ("a.pdf", 2)
B1 = ("b.pdf", 1)
B3 = ("b.pdf", 3)
C9 = ("c.pdf", 9)


def cite(chunk_id: str, *, filename: str = "a.pdf", page: int = 1) -> Citation:
    return Citation(
        citation_id="S1",
        chunk_id=chunk_id,
        document_id="doc",
        filename=filename,
        page_number=page,
        excerpt="...",
    )


# ---- recall_at_k -----------------------------------------------------------------------------


class TestRecallAtK:
    def test_all_expected_in_top_k(self) -> None:
        assert recall_at_k([A1, B1, A2], [A1, B1], k=2) == 1.0

    def test_partial_recall(self) -> None:
        assert recall_at_k([A1, C9, A2], [A1, B1], k=3) == 0.5

    def test_expected_page_beyond_k_is_missed(self) -> None:
        assert recall_at_k([A1, C9, B1], [B1], k=2) == 0.0

    def test_duplicates_in_retrieved_do_not_consume_slots(self) -> None:
        # Two chunks from the same page count as one page, so B1 is still within the top 2.
        assert recall_at_k([A1, A1, A1, B1], [B1], k=2) == 1.0

    def test_duplicate_expected_pairs_are_counted_once(self) -> None:
        assert recall_at_k([A1], [A1, A1, B1], k=5) == 0.5

    def test_empty_retrieved(self) -> None:
        assert recall_at_k([], [A1], k=5) == 0.0

    def test_empty_expected_is_vacuously_one(self) -> None:
        assert recall_at_k([A1, B1], [], k=5) == 1.0

    def test_k_larger_than_retrieved(self) -> None:
        assert recall_at_k([A1], [A1, B1], k=100) == 0.5

    @pytest.mark.parametrize("k", [0, -1])
    def test_invalid_k_raises(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            recall_at_k([A1], [A1], k=k)


# ---- hit_at_k ---------------------------------------------------------------------------------


class TestHitAtK:
    def test_hit_in_top_k(self) -> None:
        assert hit_at_k([C9, A1], [A1, B1], k=2) is True

    def test_expected_only_beyond_k(self) -> None:
        assert hit_at_k([C9, A2, A1], [A1], k=2) is False

    def test_duplicates_dedupe_before_cutoff(self) -> None:
        assert hit_at_k([C9, C9, A1], [A1], k=2) is True

    def test_no_expected_is_never_a_hit(self) -> None:
        assert hit_at_k([A1], [], k=3) is False

    def test_empty_retrieved(self) -> None:
        assert hit_at_k([], [A1], k=3) is False

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            hit_at_k([A1], [A1], k=0)


# ---- mrr --------------------------------------------------------------------------------------


class TestMrr:
    def test_first_position(self) -> None:
        assert mrr([A1, B1], [A1]) == 1.0

    def test_third_position(self) -> None:
        assert mrr([C9, A2, B1], [B1]) == pytest.approx(1 / 3)

    def test_first_matching_expected_wins(self) -> None:
        assert mrr([C9, B1, A1], [A1, B1]) == 0.5

    def test_duplicates_collapse_ranks(self) -> None:
        # Without de-duplication B1 would sit at rank 3.
        assert mrr([A1, A1, B1], [B1]) == 0.5

    def test_not_found(self) -> None:
        assert mrr([A1, A2], [B1]) == 0.0

    def test_empty_inputs(self) -> None:
        assert mrr([], [A1]) == 0.0
        assert mrr([A1], []) == 0.0


# ---- citation_validity ------------------------------------------------------------------------


class TestCitationValidity:
    def test_all_valid(self) -> None:
        cites = [cite("d:p1:c0"), cite("d:p1:c1")]
        assert citation_validity(cites, {"d:p1:c0", "d:p1:c1", "d:p2:c2"}) == 1.0

    def test_partially_valid(self) -> None:
        cites = [cite("d:p1:c0"), cite("d:p1:c1"), cite("bogus"), cite("also-bogus")]
        assert citation_validity(cites, {"d:p1:c0", "d:p1:c1"}) == 0.5

    def test_none_valid(self) -> None:
        assert citation_validity([cite("x")], {"y"}) == 0.0

    def test_no_citations_is_valid(self) -> None:
        assert citation_validity([], set()) == 1.0
        assert citation_validity([], {"d:p1:c0"}) == 1.0

    def test_empty_retrieved_set_with_citations(self) -> None:
        assert citation_validity([cite("x")], set()) == 0.0

    def test_accepts_any_collection(self) -> None:
        assert citation_validity([cite("x")], ["x", "y"]) == 1.0


# ---- citation_page_accuracy -------------------------------------------------------------------


class TestCitationPageAccuracy:
    def test_all_on_expected_pages(self) -> None:
        cites = [cite("c0", filename="a.pdf", page=1), cite("c1", filename="b.pdf", page=3)]
        assert citation_page_accuracy(cites, [A1, B3]) == 1.0

    def test_partial(self) -> None:
        cites = [
            cite("c0", filename="a.pdf", page=1),
            cite("c1", filename="a.pdf", page=2),
            cite("c2", filename="c.pdf", page=9),
        ]
        assert citation_page_accuracy(cites, [A1]) == pytest.approx(1 / 3)

    def test_same_page_different_file_does_not_count(self) -> None:
        cites = [cite("c0", filename="b.pdf", page=1)]
        assert citation_page_accuracy(cites, [A1]) == 0.0

    def test_duplicate_citations_each_count(self) -> None:
        cites = [cite("c0", page=1), cite("c0", page=1), cite("c1", page=2)]
        assert citation_page_accuracy(cites, [A1]) == pytest.approx(2 / 3)

    def test_no_citations_is_undefined(self) -> None:
        assert citation_page_accuracy([], [A1]) is None

    def test_no_expected_pages(self) -> None:
        assert citation_page_accuracy([cite("c0")], []) == 0.0


# ---- keyword_coverage / normalize_for_matching ------------------------------------------------


class TestNormalizeForMatching:
    def test_casefold_and_whitespace(self) -> None:
        assert normalize_for_matching("  Hello\n\tWORLD  ") == "hello world"

    def test_thousands_separators_removed(self) -> None:
        assert (
            normalize_for_matching("1,240 units and 5,870 aircraft")
            == "1240 units and 5870 aircraft"
        )

    def test_list_commas_are_kept(self) -> None:
        assert normalize_for_matching("1,2,3 and 12,34") == "1,2,3 and 12,34"

    def test_decimal_like_group_is_kept(self) -> None:
        # Four digits after the comma is not a thousands group.
        assert normalize_for_matching("1,2400") == "1,2400"

    def test_nfkc(self) -> None:
        assert normalize_for_matching("ﬁle") == "file"


class TestKeywordCoverage:
    def test_full_coverage(self) -> None:
        assert keyword_coverage("Pay is 750 dollars plus 60 per month", ["750", "60"]) == 1.0

    def test_partial_coverage(self) -> None:
        assert (
            keyword_coverage("Only the 750 stipend is mentioned", ["750", "60 US dollars"]) == 0.5
        )

    def test_case_insensitive(self) -> None:
        assert keyword_coverage("approved: halcyon authenticator", ["Halcyon Authenticator"]) == 1.0

    def test_phrase_keyword_with_irregular_whitespace(self) -> None:
        answer = "The parachute deploys in 0.8\n   seconds."
        assert keyword_coverage(answer, ["0.8 seconds"]) == 1.0

    def test_number_with_comma_matches_plain_number(self) -> None:
        assert keyword_coverage("The budget is $1500 per year", ["1,500"]) == 1.0

    def test_plain_number_keyword_matches_comma_number(self) -> None:
        assert keyword_coverage("Shipped 1,240 units", ["1240"]) == 1.0

    def test_missing_keyword(self) -> None:
        assert keyword_coverage("No relevant content", ["E-455"]) == 0.0

    def test_duplicate_keywords_count_once(self) -> None:
        assert keyword_coverage("E-401 only", ["E-401", "E-401", "E-503"]) == 0.5

    def test_blank_keywords_are_ignored(self) -> None:
        assert keyword_coverage("E-401 only", ["E-401", "", "   "]) == 1.0

    def test_no_keywords_is_vacuously_one(self) -> None:
        assert keyword_coverage("anything", []) == 1.0

    def test_empty_answer(self) -> None:
        assert keyword_coverage("", ["x"]) == 0.0


# ---- answer_correctness -----------------------------------------------------------------------


class TestAnswerCorrectness:
    def test_answerable_answered_with_full_coverage(self) -> None:
        score = answer_correctness(
            "Birthing parents get 16 weeks, others 8 weeks",
            ["16 weeks", "8 weeks"],
            abstained=False,
            answerable=True,
        )
        assert score == 1.0

    def test_answerable_answered_with_partial_coverage(self) -> None:
        score = answer_correctness(
            "Birthing parents get 16 weeks",
            ["16 weeks", "8 weeks"],
            abstained=False,
            answerable=True,
        )
        assert score == 0.5

    def test_answerable_but_abstained_scores_zero(self) -> None:
        # Even if the no-answer text happens to contain a keyword.
        score = answer_correctness(
            "16 weeks? I don't know.", ["16 weeks"], abstained=True, answerable=True
        )
        assert score == 0.0

    def test_unanswerable_abstained_scores_one(self) -> None:
        assert answer_correctness("", [], abstained=True, answerable=False) == 1.0

    def test_unanswerable_answered_scores_zero(self) -> None:
        assert (
            answer_correctness("The CEO is Jane Doe", [], abstained=False, answerable=False) == 0.0
        )

    def test_answerable_without_keywords_and_answered(self) -> None:
        assert answer_correctness("some answer", [], abstained=False, answerable=True) == 1.0


# ---- forbidden_terms_found --------------------------------------------------------------------


class TestForbiddenTermsFound:
    def test_detects_injected_token_case_insensitively(self) -> None:
        assert forbidden_terms_found("hacked", ["HACKED"]) == ["HACKED"]

    def test_clean_answer(self) -> None:
        answer = "Red flags: a look-alike domain, urgency, a request for a password and MFA code."
        assert forbidden_terms_found(answer, ["HACKED"]) == []

    def test_returns_only_present_terms_in_order(self) -> None:
        assert forbidden_terms_found("b then a", ["a", "zzz", "b"]) == ["a", "b"]

    def test_blank_terms_never_match(self) -> None:
        assert forbidden_terms_found("anything", ["", " "]) == []


# ---- abstention_metrics -----------------------------------------------------------------------


class TestAbstentionMetrics:
    def test_perfect(self) -> None:
        outcomes = [
            AbstentionOutcome(abstained=True, answerable=False),
            AbstentionOutcome(abstained=False, answerable=True),
        ]
        assert abstention_metrics(outcomes) == {"precision": 1.0, "recall": 1.0}

    def test_hand_computed_mix(self) -> None:
        outcomes = [
            (True, False),  # TP
            (True, False),  # TP
            (True, True),  # FP: abstained on an answerable question
            (False, False),  # FN: answered an unanswerable question
            (False, True),  # TN
            (False, True),  # TN
        ]
        result = abstention_metrics(outcomes)
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == pytest.approx(2 / 3)

    def test_never_abstains(self) -> None:
        result = abstention_metrics([(False, False), (False, True)])
        assert result == {"precision": None, "recall": 0.0}

    def test_no_unanswerable_questions(self) -> None:
        result = abstention_metrics([(True, True), (False, True)])
        assert result == {"precision": 0.0, "recall": None}

    def test_empty(self) -> None:
        assert abstention_metrics([]) == {"precision": None, "recall": None}


# ---- latency_summary / percentile -------------------------------------------------------------


class TestLatency:
    def test_percentile_nearest_rank(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0]
        assert percentile(values, 50) == 20.0  # ceil(0.5 * 4) = 2 -> second value
        assert percentile(values, 95) == 40.0  # ceil(0.95 * 4) = 4 -> fourth value
        assert percentile(values, 0) == 10.0  # rank clamped to 1
        assert percentile(values, 100) == 40.0

    def test_percentile_rejects_empty_or_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            percentile([], 50)
        with pytest.raises(ValueError, match="within"):
            percentile([1.0], 101)

    def test_summary_hand_computed(self) -> None:
        values = [50.0, 10.0, 30.0, 20.0, 40.0]  # unsorted on purpose
        summary = latency_summary(values)
        assert summary == {"p50": 30.0, "p95": 50.0, "mean": 30.0, "max": 50.0, "n": 5}

    def test_summary_twenty_values_p95_is_nineteenth(self) -> None:
        values = [float(i) for i in range(1, 21)]
        summary = latency_summary(values)
        assert summary["p95"] == 19.0  # ceil(0.95 * 20) = 19
        assert summary["p50"] == 10.0  # ceil(0.5 * 20) = 10
        assert summary["mean"] == 10.5
        assert summary["max"] == 20.0
        assert summary["n"] == 20

    def test_single_value(self) -> None:
        assert latency_summary([7.5]) == {"p50": 7.5, "p95": 7.5, "mean": 7.5, "max": 7.5, "n": 1}

    def test_empty(self) -> None:
        assert latency_summary([]) == {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0, "n": 0}

    def test_accepts_ints(self) -> None:
        summary = latency_summary([1, 2, 3])
        assert summary["mean"] == 2.0
        assert isinstance(summary["max"], float)


# ---- evaluation/dataset.json integrity ------------------------------------------------------


@pytest.fixture(scope="module")
def dataset() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        pytest.skip("evaluation/dataset.json missing")
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest_pages() -> dict[str, int]:
    if not MANIFEST_PATH.exists():
        pytest.skip("sample_data/manifest.json missing")
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {name: info["pages"] for name, info in data["documents"].items()}


@pytest.fixture(scope="module")
def page_texts(dataset: dict[str, Any]) -> dict[tuple[str, int], str]:
    """Casefolded normalized text of every page, keyed by (filename, page), via the real parser."""
    from app.ingestion.parser import parse_pdf

    texts: dict[tuple[str, int], str] = {}
    for rel in dataset["documents"]:
        path = REPO_ROOT / rel
        if not path.exists():
            pytest.skip(f"{rel} missing; run scripts/generate_sample_data.py")
        parsed = parse_pdf(path.read_bytes(), path.name)
        for page in parsed.pages:
            texts[(path.name, page.page_number)] = page.text.casefold()
    return texts


class TestDataset:
    REQUIRED_FIELDS = {
        "id",
        "question",
        "type",
        "expected_sources",
        "expected_keywords",
        "reference_answer",
        "notes",
    }

    def test_top_level_schema(self, dataset: dict[str, Any]) -> None:
        assert dataset["version"] == 1
        assert len(dataset["documents"]) == 5
        assert all(
            d.startswith("sample_data/") and d.endswith(".pdf") for d in dataset["documents"]
        )

    def test_question_fields_and_unique_ids(self, dataset: dict[str, Any]) -> None:
        ids = [q["id"] for q in dataset["questions"]]
        assert len(ids) == len(set(ids))
        for q in dataset["questions"]:
            assert set(q) >= self.REQUIRED_FIELDS, q["id"]
            assert re.fullmatch(r"q\d{2}", q["id"])
            assert q["type"] in {"answerable", "unanswerable", "cross_document"}
            assert q["question"].strip() and q["reference_answer"].strip()

    def test_composition(self, dataset: dict[str, Any]) -> None:
        counts: dict[str, int] = {}
        for q in dataset["questions"]:
            counts[q["type"]] = counts.get(q["type"], 0) + 1
        assert len(dataset["questions"]) >= 26
        assert counts["answerable"] >= 15
        assert counts["cross_document"] >= 5
        assert counts["unanswerable"] >= 5

    def test_unanswerable_have_no_sources_or_keywords(self, dataset: dict[str, Any]) -> None:
        for q in dataset["questions"]:
            if q["type"] == "unanswerable":
                assert q["expected_sources"] == [] and q["expected_keywords"] == [], q["id"]

    def test_answerable_questions_span_documents(self, dataset: dict[str, Any]) -> None:
        for q in dataset["questions"]:
            files = {s["filename"] for s in q["expected_sources"]}
            if q["type"] == "answerable":
                assert len(files) == 1 and q["expected_keywords"], q["id"]
            elif q["type"] == "cross_document":
                assert len(files) >= 2 and q["expected_keywords"], q["id"]
        covered = {
            s["filename"]
            for q in dataset["questions"]
            if q["type"] == "answerable"
            for s in q["expected_sources"]
        }
        assert covered == {Path(d).name for d in dataset["documents"]}

    def test_expected_pages_exist_in_manifest(
        self, dataset: dict[str, Any], manifest_pages: dict[str, int]
    ) -> None:
        for q in dataset["questions"]:
            for src in q["expected_sources"]:
                assert src["filename"] in manifest_pages, q["id"]
                assert 1 <= src["page"] <= manifest_pages[src["filename"]], q["id"]

    def test_injection_probe_present(self, dataset: dict[str, Any]) -> None:
        probes = [q for q in dataset["questions"] if q.get("must_not_contain")]
        assert len(probes) == 1
        probe = probes[0]
        assert probe["type"] == "answerable"
        assert probe["must_not_contain"] == ["HACKED"]
        assert probe["expected_sources"] == [
            {"filename": "halcyon_information_security_policy.pdf", "page": 6}
        ]

    def test_keywords_appear_on_expected_pages(
        self, dataset: dict[str, Any], page_texts: dict[tuple[str, int], str]
    ) -> None:
        for q in dataset["questions"]:
            pages = [page_texts[(s["filename"], s["page"])] for s in q["expected_sources"]]
            for keyword in q["expected_keywords"]:
                needle = keyword.casefold()
                assert any(needle in text for text in pages), f"{q['id']}: {keyword!r} not on pages"

    def test_injected_instruction_is_on_probe_page(
        self, page_texts: dict[tuple[str, int], str]
    ) -> None:
        assert "hacked" in page_texts[("halcyon_information_security_policy.pdf", 6)]
