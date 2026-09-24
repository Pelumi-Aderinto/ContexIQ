"""Ingest PDF files into a ContextIQ workspace from the command line.

Runs the same ``IngestionPipeline`` the API uses (parse -> chunk -> embed -> SQLite + FAISS)
against the data directory from ``.env`` / ``CONTEXTIQ_DATA_DIR`` or ``--data-dir`` and prints
one line per file with its outcome. Exit code is 1 when any file failed.

Example::

    .venv/bin/python scripts/ingest.py --workspace alpha sample_data/*.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.core.security import validate_workspace_id  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.models.schemas import DocumentUploadResult, UploadOutcome  # noqa: E402
from app.retrieval.embeddings import create_embedder  # noqa: E402
from app.retrieval.vector_store import VectorStore  # noqa: E402
from app.storage.metadata_store import MetadataStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", type=Path, help="PDF files to ingest.")
    parser.add_argument("--workspace", default=None, help="Workspace id (default: settings).")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def make_settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {"log_json": False, "log_level": args.log_level}
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    return Settings(**overrides)


def format_row(result: DocumentUploadResult) -> str:
    doc = result.document
    document_id = result.duplicate_of or (doc.document_id if doc else "-")
    pages = str(doc.page_count) if doc else "-"
    chunks = str(doc.chunk_count) if doc else "-"
    ms = f"{result.processing_ms:.0f}" if result.processing_ms is not None else "-"
    return (
        f"{result.filename:<55} {result.outcome.value:<9} {document_id:<32} "
        f"{pages:>5} {chunks:>6} {ms:>7}  {result.message}"
    )


def ingest_files(
    pipeline: IngestionPipeline, files: list[Path], workspace_id: str
) -> list[DocumentUploadResult]:
    results: list[DocumentUploadResult] = []
    for path in files:
        if not path.is_file():
            print(f"error: not a file: {path}", file=sys.stderr)
            results.append(
                DocumentUploadResult(
                    filename=path.name, outcome=UploadOutcome.FAILED, message="file not found"
                )
            )
            continue
        results.append(pipeline.ingest(path.read_bytes(), path.name, workspace_id))
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = make_settings(args)
    configure_logging(settings)
    try:
        workspace_id = validate_workspace_id(args.workspace or settings.default_workspace_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    embedder = create_embedder(settings)
    store = MetadataStore(settings.db_path)
    try:
        vector_store = VectorStore(settings.index_dir, embedder.dimension)
        pipeline = IngestionPipeline(
            settings=settings, store=store, vector_store=vector_store, embedder=embedder
        )
        results = ingest_files(pipeline, args.files, workspace_id)
    finally:
        store.close()

    header = f"{'file':<55} {'outcome':<9} {'document_id':<32} {'pages':>5} {'chunks':>6}"
    print(f"{header} {'ms':>7}  message")
    for result in results:
        print(format_row(result))
    indexed = sum(1 for r in results if r.outcome is UploadOutcome.INDEXED)
    failed = sum(1 for r in results if r.outcome is UploadOutcome.FAILED)
    print(
        f"\nWorkspace {workspace_id!r}: {indexed} indexed, "
        f"{len(results) - indexed - failed} duplicate, {failed} failed -> {settings.data_dir}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
