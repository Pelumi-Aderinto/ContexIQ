# syntax=docker/dockerfile:1
# ContextIQ: FastAPI backend and Streamlit UI in one image. See docs/DEPLOYMENT.md.
#
# Stage "builder" installs the locked dependencies with uv and pre-downloads the embedding
# model. Stage "runtime" copies only the finished virtualenv, the code and the model cache onto
# a clean slim base, so the final image carries no uv binary, package cache or build tooling
# and runs as a non-root user with a read-only code tree.
#
# Build args:
#   PYTHON_VERSION  base image tag (default 3.14; every wheel in uv.lock exists for cp314)
#   UV_VERSION      tag of ghcr.io/astral-sh/uv to copy the uv binary from (default latest)

ARG PYTHON_VERSION=3.14
ARG UV_VERSION=latest

# BuildKit does not expand variables inside `COPY --from=<image>`, so the uv image gets a
# named stage of its own that the builder copies the binaries from.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first so code changes do not invalidate this (large) layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Project sources. Tests, docs, local state and secrets are excluded by .dockerignore.
COPY README.md ./
COPY app ./app
COPY scripts ./scripts
COPY sample_data ./sample_data
COPY evaluation ./evaluation
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Pre-download the embedding model so the runtime image works without network access.
ENV HF_HOME=/app/.hf_cache
RUN .venv/bin/python scripts/download_models.py

# ---------------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="ContextIQ" \
      org.opencontainers.image.description="Document-grounded QA over PDFs with verified citations"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/app/.hf_cache \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    CONTEXTIQ_DATA_DIR=/app/data \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN groupadd --system --gid 1001 contextiq \
    && useradd --system --uid 1001 --gid contextiq --create-home --home-dir /home/contextiq contextiq

WORKDIR /app

# Code and virtualenv stay root-owned (read-only for the service user).
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/pyproject.toml /app/uv.lock /app/README.md ./
COPY --from=builder /app/app ./app
COPY --from=builder /app/scripts ./scripts
COPY --from=builder /app/sample_data ./sample_data
COPY --from=builder /app/evaluation ./evaluation
# The model cache is owned by the service user so `scripts/download_models.py` can add models.
COPY --from=builder --chown=contextiq:contextiq /app/.hf_cache ./.hf_cache

# Writable locations: SQLite + FAISS data (volume) and evaluation reports.
RUN mkdir -p /app/data /app/evaluation/results \
    && chown contextiq:contextiq /app/data /app/evaluation/results

USER contextiq

EXPOSE 8000 8501

# python:3.x-slim ships no curl; the health probe uses the standard library instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["/app/.venv/bin/uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
