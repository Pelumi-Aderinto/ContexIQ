"""Ask a ContextIQ workspace a question from the command line.

Builds the retrieval and answer services in-process (no API server needed), runs the question
and prints the answer with its verified citations. With ``--debug`` every retrieved chunk and
the timings are printed too. The LLM provider comes from ``.env`` unless ``--provider`` is set;
``--provider extractive`` returns the top passages without calling any LLM.

Example::

    .venv/bin/python scripts/query.py --workspace alpha "How many days of PTO do employees get?"
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.core.security import validate_workspace_id  # noqa: E402
from app.generation.chain import AnswerService, GenerationError  # noqa: E402
from app.generation.llm import ConfigurationError, create_chat_model  # noqa: E402
from app.models.schemas import QueryRequest, QueryResponse, RetrievalMode  # noqa: E402
from app.retrieval.embeddings import create_embedder  # noqa: E402
from app.retrieval.retriever import RetrievalService  # noqa: E402
from app.retrieval.vector_store import VectorStore  # noqa: E402
from app.storage.metadata_store import MetadataStore  # noqa: E402

_DEBUG_TEXT_CHARS = 160


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument("--workspace", default=None, help="Workspace id (default: settings).")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--mode", choices=("dense", "hybrid"), default=None)
    parser.add_argument(
        "--provider",
        choices=("auto", "anthropic", "openai", "groq", "extractive"),
        default=None,
        help="LLM provider override; 'extractive' skips the LLM.",
    )
    parser.add_argument("--debug", action="store_true", help="Print retrieved chunks and timings.")
    parser.add_argument("--log-level", default="WARNING")
    return parser


def make_settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {"log_json": False, "log_level": args.log_level}
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.provider is not None:
        overrides["llm_provider"] = args.provider
    return Settings(**overrides)


def print_response(response: QueryResponse, *, debug: bool) -> None:
    mode = response.answer_mode.value + (f" ({response.model})" if response.model else "")
    print(f"[{mode}] {'ABSTAINED' if response.abstained else 'answer'}\n")
    print(response.answer)
    if response.citations:
        print("\nCitations:")
        for c in response.citations:
            score = f" score={c.score:.3f}" if c.score is not None else ""
            print(f"  [{c.citation_id}] {c.filename} p.{c.page_number}{score}\n      {c.excerpt}")
    if response.invalid_citation_ids:
        print(f"\nDropped invalid citation labels: {', '.join(response.invalid_citation_ids)}")
    if debug:
        print_debug(response)


def print_debug(response: QueryResponse) -> None:
    print("\nRetrieved chunks:")
    for rank, chunk in enumerate(response.retrieved or [], start=1):
        signals = f"score={chunk.score:.4f}"
        if chunk.dense_score is not None:
            signals += f" dense={chunk.dense_score:.4f}"
        if chunk.sparse_rank is not None:
            signals += f" sparse_rank={chunk.sparse_rank}"
        if chunk.rerank_score is not None:
            signals += f" rerank={chunk.rerank_score:.4f}"
        text = " ".join(chunk.text.split())[:_DEBUG_TEXT_CHARS]
        print(f"  {rank:>2}. {chunk.filename} p.{chunk.page_number} ({signals})\n      {text}")
    t = response.timings
    print(
        f"\nTimings: retrieval {t.retrieval_ms:.1f} ms, generation {t.generation_ms:.1f} ms, "
        f"total {t.total_ms:.1f} ms"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = make_settings(args)
    configure_logging(settings)
    try:
        workspace_id = validate_workspace_id(args.workspace or settings.default_workspace_id)
        request = QueryRequest(
            question=args.question,
            top_k=args.top_k,
            mode=RetrievalMode(args.mode) if args.mode else None,
            include_debug=args.debug,
        )
        llm = create_chat_model(settings)
    except (ValueError, ConfigurationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    embedder = create_embedder(settings)
    store = MetadataStore(settings.db_path)
    try:
        vector_store = VectorStore(settings.index_dir, embedder.dimension)
        retrieval = RetrievalService(
            settings=settings, store=store, vector_store=vector_store, embedder=embedder
        )
        service = AnswerService(settings=settings, retrieval=retrieval, llm=llm)
        if store.count_chunks(workspace_id) == 0:
            print(
                f"warning: workspace {workspace_id!r} has no indexed chunks in {settings.data_dir}",
                file=sys.stderr,
            )
        response = service.answer(workspace_id, request, request_id=uuid.uuid4().hex)
    except GenerationError as exc:
        print(f"error: the LLM call failed ({exc})", file=sys.stderr)
        return 1
    finally:
        store.close()
    print_response(response, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
