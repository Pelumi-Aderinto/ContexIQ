"""Tests for app.ingestion.chunker."""

from __future__ import annotations

import hashlib
import re
from itertools import pairwise

import pytest

from app.ingestion.chunker import MIN_CHUNK_CHARS, chunk_document
from app.ingestion.parser import parse_pdf
from app.models.domain import Chunk, PageText, ParsedDocument

DOC_ID = "0123456789abcdef0123456789abcdef"
WS_ID = "alpha"
CHUNK_ID_RE = re.compile(rf"^{DOC_ID}:p(\d+):c(\d+)$")


def _parsed(pages: list[str], filename: str = "doc.pdf") -> ParsedDocument:
    joined = "\n".join(pages).encode()
    return ParsedDocument(
        filename=filename,
        sha256=hashlib.sha256(joined).hexdigest(),
        size_bytes=len(joined),
        page_count=len(pages),
        pages=[PageText(page_number=i + 1, text=t) for i, t in enumerate(pages)],
    )


def _long_page(seed: str, sentences: int) -> str:
    return " ".join(
        f"{seed} sentence {i} adds a few more words to the page." for i in range(sentences)
    )


def _chunk(parsed: ParsedDocument, *, size: int = 200, overlap: int = 50) -> list[Chunk]:
    return chunk_document(
        parsed, document_id=DOC_ID, workspace_id=WS_ID, chunk_size=size, chunk_overlap=overlap
    )


def test_chunks_never_span_pages_and_offsets_slice_back() -> None:
    parsed = _parsed([_long_page("Alpha", 25), _long_page("Beta", 25), _long_page("Gamma", 5)])
    chunks = _chunk(parsed)

    assert len(chunks) > 3
    for chunk in chunks:
        page = parsed.pages[chunk.page_number - 1]
        assert page.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.text.strip()
        assert 0 <= chunk.char_start < chunk.char_end <= len(page.text)
        assert chunk.text.split()[0] in {"Alpha", "Beta", "Gamma"}
        assert chunk.text.startswith(("Alpha", "Beta", "Gamma")[chunk.page_number - 1])


def test_chunk_ids_follow_pattern() -> None:
    chunks = _chunk(_parsed([_long_page("Alpha", 20), _long_page("Beta", 20)]))
    for chunk in chunks:
        match = CHUNK_ID_RE.match(chunk.chunk_id)
        assert match, chunk.chunk_id
        assert int(match.group(1)) == chunk.page_number
        assert int(match.group(2)) == chunk.chunk_index
        assert chunk.chunk_id == Chunk.make_id(DOC_ID, chunk.page_number, chunk.chunk_index)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_consecutive_chunks_on_a_page_overlap() -> None:
    parsed = _parsed([_long_page("Alpha", 40)])
    chunks = _chunk(parsed, size=200, overlap=60)

    assert len(chunks) >= 4
    for previous, current in pairwise(chunks):
        assert current.char_start > previous.char_start
        assert current.char_start < previous.char_end, "expected overlapping character ranges"
        shared = parsed.pages[0].text[current.char_start : previous.char_end]
        assert shared and previous.text.endswith(shared) and current.text.startswith(shared)


def test_empty_pages_yield_no_chunks() -> None:
    chunks = _chunk(_parsed(["", "   \n\n ", _long_page("Gamma", 10)]))
    assert chunks
    assert {c.page_number for c in chunks} == {3}
    assert _chunk(_parsed(["", ""])) == []


def test_chunk_index_is_global_and_increasing() -> None:
    chunks = _chunk(_parsed([_long_page("Alpha", 15), "", _long_page("Gamma", 15)]))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    page_of_first_gamma = next(c for c in chunks if c.page_number == 3)
    assert page_of_first_gamma.chunk_index > 0


def test_ids_and_filename_propagate() -> None:
    chunks = _chunk(_parsed([_long_page("Alpha", 5)], filename="handbook.pdf"))
    assert chunks
    for chunk in chunks:
        assert chunk.document_id == DOC_ID
        assert chunk.workspace_id == WS_ID
        assert chunk.filename == "handbook.pdf"


def test_tiny_fragment_is_dropped_when_page_has_other_chunks() -> None:
    page = "A" * 98 + "\n\nok."
    chunks = _chunk(_parsed([page]), size=100, overlap=0)
    assert [c.text for c in chunks] == ["A" * 98]


def test_tiny_page_keeps_its_only_chunk() -> None:
    chunks = _chunk(_parsed(["Short."]), size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == "Short."
    assert len(chunks[0].text) < MIN_CHUNK_CHARS


def test_chunks_do_not_start_with_sentence_separator() -> None:
    chunks = _chunk(_parsed([_long_page("Alpha", 40)]), size=150, overlap=40)
    assert len(chunks) > 2
    for chunk in chunks:
        assert not chunk.text.startswith(". ")
        assert chunk.text == chunk.text.strip()


def test_paragraph_breaks_are_preferred_split_points() -> None:
    paragraphs = [f"Paragraph {i} " + "word " * 15 + "ends here." for i in range(6)]
    parsed = _parsed(["\n\n".join(paragraphs)])
    chunks = _chunk(parsed, size=120, overlap=0)
    assert len(chunks) == 6
    assert [c.text for c in chunks] == paragraphs


@pytest.mark.parametrize(("size", "overlap"), [(100, 100), (100, 150), (0, 0), (100, -1)])
def test_invalid_sizes_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        _chunk(_parsed(["text"]), size=size, overlap=overlap)


def test_end_to_end_with_real_pdf(make_pdf) -> None:
    pages = [_long_page("Alpha", 30), "", _long_page("Gamma", 8)]
    parsed = parse_pdf(make_pdf(pages), "e2e.pdf")
    chunks = _chunk(parsed, size=300, overlap=50)

    assert {c.page_number for c in chunks} == {1, 3}
    for chunk in chunks:
        page = parsed.pages[chunk.page_number - 1]
        assert page.text[chunk.char_start : chunk.char_end] == chunk.text
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
