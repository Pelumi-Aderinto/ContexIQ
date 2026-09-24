"""Chunking: split a :class:`ParsedDocument` into page-bounded :class:`Chunk` objects.

Each page is split independently with LangChain's ``RecursiveCharacterTextSplitter`` so a
chunk never spans two pages (page citations stay exact). ``char_start``/``char_end`` are
offsets into the page's normalized text, and ``page.text[char_start:char_end] == chunk.text``
always holds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.domain import Chunk, ParsedDocument

logger = logging.getLogger(__name__)

SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]
MIN_CHUNK_CHARS = 20
_SENTENCE_SEPARATOR = ". "


@dataclass(frozen=True, slots=True)
class _Piece:
    text: str
    start: int
    end: int


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        keep_separator=True,
    )


def _clean_piece(raw: str) -> str:
    """Strip whitespace and the leading ``". "`` the splitter keeps from the previous sentence."""
    piece = raw.strip()
    if piece.startswith(_SENTENCE_SEPARATOR):
        piece = piece[len(_SENTENCE_SEPARATOR) :].lstrip()
    return piece


def _locate_pieces(page_text: str, raw_pieces: list[str]) -> list[_Piece]:
    """Map each split piece back to its offsets in ``page_text`` using a moving cursor.

    The cursor only advances past the *start* of the previous piece because consecutive
    pieces overlap by design.
    """
    located: list[_Piece] = []
    cursor = 0
    for raw in raw_pieces:
        text = _clean_piece(raw)
        if not text:
            continue
        start = page_text.find(text, cursor)
        if start == -1:
            start = page_text.find(text)
        if start == -1:
            logger.warning("chunk_piece_not_found", extra={"piece_chars": len(text)})
            continue
        located.append(_Piece(text=text, start=start, end=start + len(text)))
        cursor = max(start + 1, cursor)
    return located


def _drop_tiny_pieces(pieces: list[_Piece]) -> list[_Piece]:
    """Skip fragments shorter than ``MIN_CHUNK_CHARS`` unless they are all a page has."""
    if len(pieces) <= 1:
        return pieces
    kept = [p for p in pieces if len(p.text.strip()) >= MIN_CHUNK_CHARS]
    return kept or pieces


def _split_page(page_text: str, splitter: RecursiveCharacterTextSplitter) -> list[_Piece]:
    if not page_text.strip():
        return []
    return _drop_tiny_pieces(_locate_pieces(page_text, splitter.split_text(page_text)))


def chunk_document(
    parsed: ParsedDocument,
    *,
    document_id: str,
    workspace_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split every page of ``parsed`` into overlapping chunks with exact provenance.

    ``chunk_index`` is 0-based and increases across the whole document; empty pages yield no
    chunks. Raises ``ValueError`` when ``chunk_overlap`` is not smaller than ``chunk_size``.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    splitter = _build_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for page in parsed.pages:
        for piece in _split_page(page.text, splitter):
            chunk_index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(document_id, page.page_number, chunk_index),
                    document_id=document_id,
                    workspace_id=workspace_id,
                    filename=parsed.filename,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=piece.text,
                    char_start=piece.start,
                    char_end=piece.end,
                )
            )
    return chunks
