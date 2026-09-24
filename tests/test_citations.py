"""Tests for tolerant LLM output parsing and citation verification."""

from __future__ import annotations

import uuid

import pytest

from app.generation.citations import (
    LLMAnswer,
    build_citations,
    extract_inline_labels,
    make_excerpt,
    normalize_label,
    parse_llm_answer,
)
from app.models.domain import Chunk, ScoredChunk

WORKSPACE = "acme"


def make_scored(
    index: int, text: str, *, page: int = 2, filename: str = "guide.pdf"
) -> ScoredChunk:
    document_id = uuid.uuid4().hex
    chunk = Chunk(
        chunk_id=Chunk.make_id(document_id, page, index),
        document_id=document_id,
        workspace_id=WORKSPACE,
        filename=filename,
        page_number=page,
        chunk_index=index,
        text=text,
        char_start=0,
        char_end=len(text),
    )
    return ScoredChunk(chunk=chunk, vector_id=index + 1, score=0.8 - index * 0.1)


# ---- parse_llm_answer ------------------------------------------------------------------------


def test_parse_fenced_json_with_language_tag() -> None:
    text = (
        "```json\n"
        '{"answer": "Leave is 25 days [S1].", "citations": ["S1"], '
        '"insufficient_evidence": false}\n'
        "```"
    )
    parsed = parse_llm_answer(text)
    assert parsed == LLMAnswer(
        answer="Leave is 25 days [S1].", citations=["S1"], insufficient_evidence=False
    )


def test_parse_fenced_json_without_language_tag() -> None:
    text = '```\n{"answer": "Yes [S2].", "citations": ["S2"], "insufficient_evidence": false}\n```'
    parsed = parse_llm_answer(text)
    assert parsed.answer == "Yes [S2]."
    assert parsed.citations == ["S2"]


def test_parse_bare_json() -> None:
    text = '{"answer": "It renews yearly [S1][S3].", "citations": ["S1", "S3"]}'
    parsed = parse_llm_answer(text)
    assert parsed.answer == "It renews yearly [S1][S3]."
    assert parsed.citations == ["S1", "S3"]
    assert parsed.insufficient_evidence is False


def test_parse_json_embedded_in_prose() -> None:
    text = (
        "Sure, here is the result:\n"
        '{"answer": "The fee is $40 [S2].", "citations": ["S2"], "insufficient_evidence": false}\n'
        "Let me know if you need anything else."
    )
    parsed = parse_llm_answer(text)
    assert parsed.answer == "The fee is $40 [S2]."
    assert parsed.citations == ["S2"]


def test_parse_handles_braces_inside_strings() -> None:
    text = 'Result: {"answer": "Use {braces} and a } here [S1].", "citations": ["S1"]} done'
    parsed = parse_llm_answer(text)
    assert parsed.answer == "Use {braces} and a } here [S1]."
    assert parsed.citations == ["S1"]


def test_parse_invalid_json_falls_back_to_text_and_inline_labels() -> None:
    text = "  The handbook says 25 days [S1] and carry-over is allowed [S3].  "
    parsed = parse_llm_answer(text)
    assert parsed.answer == text.strip()
    assert parsed.citations == ["S1", "S3"]
    assert parsed.insufficient_evidence is False


def test_parse_unions_inline_labels_with_citation_list() -> None:
    text = '{"answer": "A [S1]. B [S2].", "citations": ["S1"], "insufficient_evidence": false}'
    parsed = parse_llm_answer(text)
    assert parsed.citations == ["S1", "S2"]


def test_parse_union_does_not_duplicate_equivalent_labels() -> None:
    text = '{"answer": "A [S1].", "citations": ["[s1]"]}'
    parsed = parse_llm_answer(text)
    assert parsed.citations == ["[s1]"]


def test_parse_coerces_citation_entries() -> None:
    text = '{"answer": "x", "citations": ["S1", 2, null, {"id": "S3"}, true]}'
    parsed = parse_llm_answer(text)
    assert parsed.citations == ["S1", "2"]


def test_parse_accepts_single_string_citations_and_string_flag() -> None:
    text = '{"answer": "Not covered.", "citations": "S1", "insufficient_evidence": "true"}'
    parsed = parse_llm_answer(text)
    assert parsed.citations == ["S1"]
    assert parsed.insufficient_evidence is True


def test_parse_json_that_is_not_an_object_falls_back() -> None:
    parsed = parse_llm_answer('["S1", "S2"]')
    assert parsed.answer == '["S1", "S2"]'
    assert parsed.citations == []


def test_parse_empty_text() -> None:
    parsed = parse_llm_answer("")
    assert parsed == LLMAnswer()


# ---- labels ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("S1", "S1"),
        ("s1", "S1"),
        ("[S1]", "S1"),
        ("S 1", "S1"),
        ("  [s12]  ", "S12"),
        ("S007", "S7"),
        ("source S1", None),
        ("1", None),
        ("S", None),
        ("", None),
        ("S1a", None),
        ("[S1", None),
    ],
)
def test_normalize_label(raw: str, expected: str | None) -> None:
    assert normalize_label(raw) == expected


def test_extract_inline_labels_order_and_dedupe() -> None:
    text = "First [S3]. Second [S1], again [S3]. Group [S2, s4][S1]. Not a label [Section 12]."
    assert extract_inline_labels(text) == ["S3", "S1", "S2", "S4"]


def test_extract_inline_labels_none() -> None:
    assert extract_inline_labels("No markers here.") == []


# ---- build_citations -------------------------------------------------------------------------


def test_build_citations_splits_valid_and_invalid_preserving_order() -> None:
    chunks = [make_scored(0, "alpha text", page=1), make_scored(1, "beta text", page=4)]
    label_map = {"S1": chunks[0], "S2": chunks[1]}

    citations, invalid = build_citations(
        ["S2", "S9", "[s1]", "S2", "junk", ""], label_map, excerpt_chars=100
    )

    assert [c.citation_id for c in citations] == ["S2", "S1"]
    assert invalid == ["S9", "junk"]
    assert citations[0].chunk_id == chunks[1].chunk.chunk_id
    assert citations[0].document_id == chunks[1].chunk.document_id
    assert citations[0].filename == "guide.pdf"
    assert citations[0].page_number == 4
    assert citations[0].excerpt == "beta text"
    assert citations[0].score == pytest.approx(chunks[1].score)
    assert citations[1].chunk_id == chunks[0].chunk.chunk_id
    assert citations[1].page_number == 1


def test_build_citations_uses_excerpt_limit() -> None:
    sc = make_scored(0, "lorem ipsum " * 50)
    citations, invalid = build_citations(["S1"], {"S1": sc}, excerpt_chars=60)

    assert invalid == []
    assert len(citations[0].excerpt) <= 60
    assert citations[0].excerpt.endswith("...")


def test_build_citations_empty() -> None:
    assert build_citations([], {}, excerpt_chars=100) == ([], [])


# ---- make_excerpt ------------------------------------------------------------------------------


def test_make_excerpt_collapses_whitespace() -> None:
    assert make_excerpt("  a\n\n b\t\tc  ", 100) == "a b c"


def test_make_excerpt_short_text_unchanged() -> None:
    assert make_excerpt("short text", 20) == "short text"


def test_make_excerpt_truncates_at_word_boundary_with_ellipsis() -> None:
    text = "The quick brown fox jumps over the lazy dog repeatedly"
    excerpt = make_excerpt(text, 24)

    assert excerpt == "The quick brown fox..."
    assert len(excerpt) <= 24


def test_make_excerpt_long_single_word_is_hard_cut() -> None:
    excerpt = make_excerpt("a" * 100, 20)
    assert len(excerpt) == 20
    assert excerpt.endswith("...")
