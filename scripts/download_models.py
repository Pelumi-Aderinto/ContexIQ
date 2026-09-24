"""Pre-download the embedding model (and optionally the reranker) into the Hugging Face cache.

Intended for Docker builds and offline deployments: run it once with network access so the
application never downloads at startup. Model names default to the application settings
(``CONTEXTIQ_EMBEDDING_MODEL``, ``CONTEXTIQ_RERANK_MODEL``); ``--cache-dir`` overrides
``CONTEXTIQ_EMBEDDING_CACHE_DIR`` and the Hugging Face default.

Example::

    .venv/bin/python scripts/download_models.py --rerank
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--rerank", action="store_true", help="Also download the reranker.")
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def make_settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {"log_json": False, "log_level": args.log_level}
    if args.embedding_model:
        overrides["embedding_model"] = args.embedding_model
    if args.rerank_model:
        overrides["rerank_model"] = args.rerank_model
    if args.cache_dir is not None:
        overrides["embedding_cache_dir"] = args.cache_dir
    return Settings(**overrides)


def download_embedding_model(model_name: str, cache_dir: Path | None) -> int:
    """Load the sentence-transformers model (downloading if needed); returns its dimension."""
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(
        model_name, device="cpu", cache_folder=str(cache_dir) if cache_dir else None
    )
    dimension = model.get_sentence_embedding_dimension() or 0
    print(f"embedding model {model_name}: {dimension}-d, {time.perf_counter() - started:.1f} s")
    return int(dimension)


def download_reranker(model_name: str, cache_dir: Path | None) -> None:
    """Load the cross-encoder reranker (downloading if needed)."""
    from sentence_transformers import CrossEncoder

    started = time.perf_counter()
    CrossEncoder(model_name, device="cpu", cache_folder=str(cache_dir) if cache_dir else None)
    print(f"reranker {model_name}: ready, {time.perf_counter() - started:.1f} s")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = make_settings(args)
    configure_logging(settings)
    cache_dir = settings.embedding_cache_dir
    try:
        download_embedding_model(settings.embedding_model, cache_dir)
        if args.rerank:
            download_reranker(settings.rerank_model, cache_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: model download failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    print(f"Models cached in {cache_dir or 'the default Hugging Face cache'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
