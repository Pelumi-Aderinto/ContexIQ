"""Rebuild FAISS indexes from the chunks stored in SQLite.

Re-embeds every chunk of a workspace with the configured embedding model and replaces the
workspace's ``{index_dir}/{workspace}.faiss`` file. Use it when an index file was lost or
corrupted, or after changing ``CONTEXTIQ_EMBEDDING_MODEL``.

Changing the embedding model invalidates *every* workspace index (vectors from different
models are not comparable), so in that case run this without ``--workspace`` to rebuild all.

Example::

    .venv/bin/python scripts/rebuild_index.py --workspace alpha
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.core.security import validate_workspace_id  # noqa: E402
from app.retrieval.embeddings import Embedder, create_embedder  # noqa: E402
from app.retrieval.vector_store import VectorStore  # noqa: E402
from app.storage.metadata_store import MetadataStore  # noqa: E402

MODEL_CHANGE_WARNING = (
    "warning: if the embedding model changed, every workspace must be rebuilt; indexes built "
    "with another model return meaningless results (run without --workspace to rebuild all)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--workspace", default=None, help="Workspace to rebuild (default: every workspace)."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Chunks embedded per batch (default 256)."
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def make_settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {"log_json": False, "log_level": args.log_level}
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    return Settings(**overrides)


def embed_workspace(
    store: MetadataStore, embedder: Embedder, workspace_id: str, batch_size: int
) -> tuple[list[int], np.ndarray]:
    """Re-embed all chunks of ``workspace_id`` in batches; returns ``(vector_ids, vectors)``."""
    vector_ids: list[int] = []
    batches: list[np.ndarray] = []
    pending_ids: list[int] = []
    pending_texts: list[str] = []

    def flush() -> None:
        if pending_texts:
            batches.append(embedder.embed_documents(pending_texts))
            vector_ids.extend(pending_ids)
            pending_ids.clear()
            pending_texts.clear()

    for vector_id, chunk in store.iter_chunks(workspace_id):
        pending_ids.append(vector_id)
        pending_texts.append(chunk.text)
        if len(pending_texts) >= batch_size:
            flush()
    flush()
    if not batches:
        return [], np.empty((0, embedder.dimension), dtype=np.float32)
    return vector_ids, np.vstack(batches)


def rebuild(
    store: MetadataStore,
    vector_store: VectorStore,
    embedder: Embedder,
    workspace_id: str,
    batch_size: int,
) -> int:
    """Rebuild one workspace index and return the number of vectors written."""
    started = time.perf_counter()
    vector_ids, vectors = embed_workspace(store, embedder, workspace_id, batch_size)
    vector_store.rebuild(workspace_id, vector_ids, vectors)
    elapsed = time.perf_counter() - started
    print(f"{workspace_id:<30} {len(vector_ids):>7} vectors  {elapsed:7.1f} s")
    return len(vector_ids)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        print("error: --batch-size must be >= 1", file=sys.stderr)
        return 2
    settings = make_settings(args)
    configure_logging(settings)

    store = MetadataStore(settings.db_path)
    try:
        if args.workspace is not None:
            try:
                workspaces = [validate_workspace_id(args.workspace)]
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print(MODEL_CHANGE_WARNING, file=sys.stderr)
        else:
            workspaces = store.list_workspaces()
        if not workspaces:
            print(f"No workspaces found in {settings.db_path}; nothing to rebuild.")
            return 0

        embedder = create_embedder(settings)
        vector_store = VectorStore(settings.index_dir, embedder.dimension)
        print(f"Embedding model: {embedder.model_name} ({embedder.dimension}-d)")
        total = sum(
            rebuild(store, vector_store, embedder, workspace_id, args.batch_size)
            for workspace_id in workspaces
        )
    finally:
        store.close()
    print(f"Rebuilt {len(workspaces)} workspace index(es), {total} vectors -> {settings.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
