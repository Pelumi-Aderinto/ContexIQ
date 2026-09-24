"""PDF parsing: raw upload bytes -> :class:`ParsedDocument` with normalized per-page text.

Text is extracted with PyMuPDF (``page.get_text("text")``) and normalized so that downstream
chunking, embedding and citation excerpts work on clean prose. Every failure mode is mapped to
a :class:`PdfParseError` with a stable ``code`` so the API can report *why* a file was rejected.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata

import pymupdf

from app.models.domain import PageText, ParsedDocument

logger = logging.getLogger(__name__)

# MuPDF prints low-level warnings for damaged files straight to stderr; we report failures
# through exceptions and structured logs instead, so silence that channel.
pymupdf.TOOLS.mupdf_display_errors(False)

PDF_MAGIC = b"%PDF-"

# C0/C1 control characters except tab (0x09) and newline (0x0A), plus a few zero-width
# format characters (soft hyphen, zero-width space/joiners, BOM) that break words in PDFs.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u00ad\u200b-\u200d\ufeff]")
_HYPHEN_BREAK_RE = re.compile(r"(\S+)-[ \t]*\n[ \t]*(\S+)")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")


class PdfParseError(Exception):
    """Raised when an uploaded file cannot be parsed into text.

    ``code`` is one of: ``not_pdf``, ``encrypted``, ``corrupt``, ``too_many_pages``,
    ``no_text``, ``parse_error``.
    """

    def __init__(self, message: str, *, code: str = "parse_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return self.message


def is_pdf(data: bytes) -> bool:
    """Return True when ``data`` starts with the PDF magic bytes ``%PDF-``."""
    return data.startswith(PDF_MAGIC)


# --------------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------------


def _join_hyphenated(match: re.Match[str]) -> str:
    """Rejoin a word that was hyphenated across a line break.

    ``infor-\\nmation`` becomes ``information``. Compound words (``state-of-the-\\nart``),
    capitalised continuations (``Wi-\\nFi``) and non-alphabetic parts (``2024-\\n2025``) keep
    their hyphen because it is most likely a real one.
    """
    head, tail = match.group(1), match.group(2)
    is_syllable_break = head[-1].isalpha() and tail[0].isalpha() and tail[0].islower()
    if "-" in head or not is_syllable_break:
        return f"{head}-{tail}"
    return f"{head}{tail}"


def _collapse_paragraphs(text: str) -> str:
    """Join wrapped lines with spaces; keep blank-line paragraph breaks as one ``\\n\\n``."""
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in text.split("\n"):
        line = _INLINE_WHITESPACE_RE.sub(" ", raw_line).strip()
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def normalize_text(text: str) -> str:
    """Normalize extracted PDF text into clean prose.

    Steps: Unicode NFKC, unify newlines, drop control characters (keeping newline and tab),
    rejoin words hyphenated across line breaks, turn single newlines inside a paragraph into
    spaces while preserving blank-line paragraph breaks, and collapse runs of whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\t", " ")
    text = _HYPHEN_BREAK_RE.sub(_join_hyphenated, text)
    return _collapse_paragraphs(text).strip()


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------


def _open_document(data: bytes) -> pymupdf.Document:
    try:
        return pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # pymupdf raises FileDataError/RuntimeError/ValueError variants
        raise PdfParseError(
            "The file could not be opened as a PDF; it may be corrupt or truncated.",
            code="corrupt",
        ) from exc


def _extract_page_text(doc: pymupdf.Document, index: int) -> str:
    """Extract and normalize one page; a damaged page degrades to empty text."""
    try:
        raw = doc[index].get_text("text")
    except Exception as exc:  # a single broken page must not sink the whole document
        logger.warning(
            "pdf_page_extraction_failed",
            extra={"page_number": index + 1, "error_type": type(exc).__name__},
        )
        return ""
    return normalize_text(raw)


def _extract_title(doc: pymupdf.Document) -> str | None:
    metadata = doc.metadata or {}
    title = (metadata.get("title") or "").strip()
    return normalize_text(title) or None


def parse_pdf(data: bytes, filename: str, *, max_pages: int = 500) -> ParsedDocument:
    """Parse raw PDF bytes into a :class:`ParsedDocument`.

    Raises :class:`PdfParseError` with ``code`` ``not_pdf``, ``corrupt``, ``encrypted``,
    ``too_many_pages``, ``no_text`` or ``parse_error``.
    """
    if not is_pdf(data):
        raise PdfParseError("The file is not a PDF (missing %PDF- header).", code="not_pdf")

    sha256 = hashlib.sha256(data).hexdigest()
    doc = _open_document(data)
    try:
        if doc.needs_pass:
            raise PdfParseError("The PDF is password protected.", code="encrypted")
        page_count = doc.page_count
        if page_count > max_pages:
            raise PdfParseError(
                f"The PDF has {page_count} pages; the limit is {max_pages}.",
                code="too_many_pages",
            )
        pages = [
            PageText(page_number=i + 1, text=_extract_page_text(doc, i)) for i in range(page_count)
        ]
        title = _extract_title(doc)
    except PdfParseError:
        raise
    except Exception as exc:
        raise PdfParseError("Failed to extract text from the PDF.") from exc
    finally:
        doc.close()

    parsed = ParsedDocument(
        filename=filename,
        sha256=sha256,
        size_bytes=len(data),
        page_count=page_count,
        pages=pages,
        title=title,
    )
    if not parsed.has_text:
        raise PdfParseError(
            "The PDF contains no extractable text (scanned or image-only?).", code="no_text"
        )
    return parsed
