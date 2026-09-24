"""Tests for app.ingestion.parser."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.ingestion.parser import PdfParseError, is_pdf, normalize_text, parse_pdf

PAGES = [
    "First page: alpha beta gamma.",
    "Second page: delta epsilon zeta.",
    "Third page: eta theta iota.",
]


def _encrypted_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), "top secret")
        return doc.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="user-pw", owner_pw="owner-pw"
        )
    finally:
        doc.close()


# --------------------------------------------------------------------------------------------
# parse_pdf
# --------------------------------------------------------------------------------------------


def test_parse_pdf_extracts_pages_hash_and_title(make_pdf) -> None:
    data = make_pdf(PAGES, title="Probe Title")
    parsed = parse_pdf(data, "probe.pdf")

    assert parsed.filename == "probe.pdf"
    assert parsed.page_count == 3
    assert [p.page_number for p in parsed.pages] == [1, 2, 3]
    assert [p.text for p in parsed.pages] == PAGES
    assert parsed.sha256 == hashlib.sha256(data).hexdigest()
    assert parsed.size_bytes == len(data)
    assert parsed.title == "Probe Title"
    assert parsed.has_text
    assert parsed.total_chars == sum(len(p) for p in PAGES)


def test_parse_pdf_title_is_none_without_metadata(make_pdf) -> None:
    parsed = parse_pdf(make_pdf(["Some text."]), "untitled.pdf")
    assert parsed.title is None


def test_parse_pdf_normalizes_wrapped_lines(make_pdf) -> None:
    long_paragraph = " ".join(
        f"Sentence number {i} keeps the paragraph flowing." for i in range(30)
    )
    parsed = parse_pdf(make_pdf([long_paragraph]), "wrapped.pdf")
    assert parsed.pages[0].text == long_paragraph


def test_parse_pdf_keeps_blank_pages_as_empty_text(make_pdf) -> None:
    parsed = parse_pdf(make_pdf(["Text on page one.", ""]), "mixed.pdf")
    assert parsed.page_count == 2
    assert parsed.pages[0].text == "Text on page one."
    assert parsed.pages[1].text == ""


@pytest.mark.parametrize("data", [b"hello", b"", b"PDF-1.7 without percent"])
def test_parse_pdf_rejects_non_pdf(data: bytes) -> None:
    with pytest.raises(PdfParseError) as excinfo:
        parse_pdf(data, "not.pdf")
    assert excinfo.value.code == "not_pdf"


def test_parse_pdf_rejects_corrupt_file() -> None:
    with pytest.raises(PdfParseError) as excinfo:
        parse_pdf(b"%PDF-1.7 garbage", "corrupt.pdf")
    assert excinfo.value.code == "corrupt"


def test_parse_pdf_rejects_encrypted_file() -> None:
    with pytest.raises(PdfParseError) as excinfo:
        parse_pdf(_encrypted_pdf(), "locked.pdf")
    assert excinfo.value.code == "encrypted"


def test_parse_pdf_rejects_too_many_pages(make_pdf) -> None:
    with pytest.raises(PdfParseError) as excinfo:
        parse_pdf(make_pdf(PAGES), "long.pdf", max_pages=2)
    assert excinfo.value.code == "too_many_pages"
    assert "3 pages" in str(excinfo.value)


def test_parse_pdf_allows_exactly_max_pages(make_pdf) -> None:
    parsed = parse_pdf(make_pdf(PAGES), "exact.pdf", max_pages=3)
    assert parsed.page_count == 3


def test_parse_pdf_rejects_document_without_text(make_pdf) -> None:
    with pytest.raises(PdfParseError) as excinfo:
        parse_pdf(make_pdf([""]), "blank.pdf")
    assert excinfo.value.code == "no_text"


def test_pdf_parse_error_defaults() -> None:
    err = PdfParseError("boom")
    assert err.code == "parse_error"
    assert str(err) == "boom"
    assert isinstance(err, Exception)


@pytest.mark.parametrize(
    ("data", "expected"),
    [(b"%PDF-1.4\n%...", True), (b"hello", False), (b"", False), (b" %PDF-1.4", False)],
)
def test_is_pdf(data: bytes, expected: bool) -> None:
    assert is_pdf(data) is expected


# --------------------------------------------------------------------------------------------
# normalize_text
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("   \n\n  ", ""),
        ("infor-\nmation retrieval", "information retrieval"),
        ("retrieval-\naugmented generation", "retrievalaugmented generation"),
        ("a state-of-the-art system", "a state-of-the-art system"),
        ("a state-of-the-\nart system", "a state-of-the-art system"),
        ("uses Wi-\nFi links", "uses Wi-Fi links"),
        ("fiscal 2025-\n2026 budget", "fiscal 2025-2026 budget"),
        ("line one\nline two", "line one line two"),
        ("para one\n\npara two", "para one\n\npara two"),
        ("para one\n\n\n\n   \npara two", "para one\n\npara two"),
        ("windows\r\nline\r\n\r\nnext", "windows line\n\nnext"),
        ("too   many\t\tspaces", "too many spaces"),
        ("  leading and trailing  ", "leading and trailing"),
        ("\ufb01nancial \ufb02ow", "financial flow"),
        ("\uff26\uff35\uff2c\uff2c width", "FULL width"),
        ("non\u00a0breaking", "non breaking"),
        ("ctrl\x00chars\x07here\x1f", "ctrlcharshere"),
        ("soft\u00adhyphen zero\u200bwidth", "softhyphen zerowidth"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_normalize_text_is_idempotent() -> None:
    raw = "Infor-\nmation\n\nsecond   para\nwraps here.\n"
    once = normalize_text(raw)
    assert normalize_text(once) == once


# --------------------------------------------------------------------------------------------
# Generated sample documents
# --------------------------------------------------------------------------------------------


def test_sample_pdfs_parse_and_match_manifest(
    sample_data_dir: Path, sample_pdf_paths: list[Path]
) -> None:
    manifest = json.loads((sample_data_dir / "manifest.json").read_text(encoding="utf-8"))
    documents = manifest["documents"]

    assert len(sample_pdf_paths) == 5
    assert sorted(documents) == [p.name for p in sample_pdf_paths]

    for path in sample_pdf_paths:
        parsed = parse_pdf(path.read_bytes(), path.name)
        entry = documents[path.name]
        assert parsed.page_count == entry["pages"], path.name
        assert all(page.text.strip() for page in parsed.pages), f"{path.name} has an empty page"
        for page_number, headings in entry["sections"].items():
            page_text = parsed.pages[int(page_number) - 1].text
            for heading in headings:
                assert heading in page_text, f"{path.name} p{page_number}: {heading!r}"
