"""Run the ContextIQ evaluation dataset in-process and write JSON + Markdown reports.

Ingests the dataset's PDFs into a throw-away ``eval`` workspace (in a temporary data
directory unless ``--data-dir`` is given), asks every question through the same retrieval and
answer services the API uses, scores the results with ``app.evaluation.metrics`` and writes
``<output-dir>/<name>.json`` and ``<output-dir>/<name>.md`` (default ``latest``).

Examples::

    .venv/bin/python scripts/evaluate.py --provider extractive
    .venv/bin/python scripts/evaluate.py --mode dense --name dense
    .venv/bin/python scripts/evaluate.py --provider anthropic --limit 5 --json-only
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.evaluation.runner import (  # noqa: E402
    DEFAULT_K_VALUES,
    EvaluationError,
    EvaluationRunner,
    QuestionResult,
    load_dataset,
    render_markdown,
    write_report,
)

DEFAULT_DATASET = REPO_ROOT / "evaluation" / "dataset.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--name", default="latest", help="Basename of the JSON/MD files (default: latest)."
    )
    parser.add_argument("--top-k", type=int, default=None, help="top_k for the answer path.")
    parser.add_argument("--mode", choices=("dense", "hybrid"), default=None)
    parser.add_argument("--rerank", action="store_true", help="Enable the cross-encoder reranker.")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--provider",
        choices=("auto", "anthropic", "openai", "groq", "extractive"),
        default=None,
        help="LLM provider; 'extractive' runs without an LLM.",
    )
    parser.add_argument("--model", default=None, help="LLM model name override.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Persistent data directory (default: a temporary directory).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N questions."
    )
    parser.add_argument(
        "--k-values",
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated k values for recall@k / hit@k (default: 1,3,5,10).",
    )
    parser.add_argument(
        "--json-only", action="store_true", help="Do not print the Markdown report."
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def parse_k_values(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--k-values must be integers: {raw!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("--k-values must not be empty")
    return values


def settings_overrides(args: argparse.Namespace, data_dir: Path) -> dict[str, Any]:
    """Translate CLI flags into ``Settings`` fields; unset flags keep the .env / defaults."""
    overrides: dict[str, Any] = {
        "data_dir": data_dir,
        "log_json": False,
        "log_level": args.log_level,
    }
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.mode is not None:
        overrides["retrieval_mode"] = args.mode
    if args.rerank:
        overrides["rerank_enabled"] = True
    if args.embedding_model:
        overrides["embedding_model"] = args.embedding_model
    if args.provider is not None:
        overrides["llm_provider"] = args.provider
    if args.model:
        overrides["llm_model"] = args.model
    return overrides


def make_settings(args: argparse.Namespace, data_dir: Path, k_values: tuple[int, ...]) -> Settings:
    settings = Settings(**settings_overrides(args, data_dir))
    # Retrieval depth must cover the largest k and the requested top_k.
    needed = max(max(k_values), settings.top_k)
    if settings.max_top_k < needed:
        settings = settings.model_copy(update={"max_top_k": needed})
    return settings


def _hit_label(result: QuestionResult) -> str:
    if result.hit_at_k is None:
        return "n/a"
    return "hit" if result.hit_at_k[max(result.hit_at_k)] else "MISS"


def _progress(result: QuestionResult) -> None:
    hit = _hit_label(result)
    status = "abstained" if result.abstained else f"correctness={result.answer_correctness:.2f}"
    flags = " INJECTION-FAIL" if result.injection_check_passed is False else ""
    print(f"  {result.id:<5} {result.type:<15} {hit:<4} {status}{flags}", file=sys.stderr)


def run(args: argparse.Namespace, data_dir: Path) -> int:
    k_values = parse_k_values(args.k_values)
    settings = make_settings(args, data_dir, k_values)
    configure_logging(settings)
    dataset = load_dataset(args.dataset)
    runner = EvaluationRunner(settings, k_values=k_values)
    total = len(dataset.questions)
    count = total if args.limit is None else min(args.limit, total)
    print(
        f"Evaluating {count} of {total} questions over {len(dataset.documents)} documents "
        f"(mode={settings.retrieval_mode}, top_k={settings.top_k}, "
        f"provider={settings.resolved_llm_provider()}, embedding={settings.embedding_model})",
        file=sys.stderr,
    )
    report = runner.run(dataset, documents_root=REPO_ROOT, limit=args.limit, on_question=_progress)
    json_path, md_path = write_report(report, args.output_dir, stem=args.name)
    if not args.json_only:
        print(render_markdown(report))
    print(f"Wrote {json_path} and {md_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.data_dir is not None:
            return run(args, args.data_dir)
        with tempfile.TemporaryDirectory(prefix="contextiq-eval-") as tmp:
            return run(args, Path(tmp))
    except (EvaluationError, FileNotFoundError, ValueError, argparse.ArgumentTypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
