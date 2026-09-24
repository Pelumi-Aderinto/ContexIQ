# Deploying ContextIQ

ContextIQ is two processes that share one code base and one image:

| Process | Entrypoint | Port | Purpose |
|---------|-----------|------|---------|
| API | `uvicorn app.api.main:app` | 8000 | FastAPI: ingestion, retrieval, grounded answers (`/docs` for OpenAPI) |
| UI | `streamlit run app/ui/streamlit_app.py` | 8501 | Streamlit chat client that talks to the API over HTTP |

Both read their configuration from `CONTEXTIQ_*` environment variables (or a `.env` file in the
working directory). The embedding model runs locally on CPU; an LLM provider is optional.

## 1. Local run (development)

Requirements: Python 3.11+ (the lock file targets 3.14, see `.python-version`) and
[uv](https://docs.astral.sh/uv/).

```bash
git clone <repo> && cd ContexIQ
make setup                 # uv sync --frozen: creates .venv with runtime + dev dependencies
cp .env.example .env       # edit: API keys, provider key (optional)
make models                # one-time download of BAAI/bge-small-en-v1.5 (~130 MB)
make dev                   # API on http://127.0.0.1:8000, UI on http://localhost:8501
```

`make dev` runs `scripts/dev.sh`, which starts uvicorn (auto-reload on `app/`) and Streamlit
together and stops both on Ctrl-C. `make api` and `make ui` start them individually.
Other targets: `make test`, `make test-all` (includes the slow real-model tests), `make lint`,
`make format`, `make eval`, `make sample-data`; `make help` lists everything.

Plain pip instead of uv:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # runtime only, CPU-only PyTorch via the extra index
pip install -r requirements-dev.txt      # adds pytest / ruff (also pulls requirements.txt)
```

`requirements.txt` is generated from `uv.lock` (`make requirements`); do not edit it by hand.
It mixes PyPI with the PyTorch CPU index, which pip handles as is; uv's `uv pip install` needs
`--index-strategy unsafe-best-match` for such files (or simply use `uv sync`).

Smoke test without the UI:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/documents -H "X-API-Key: dev-key-alpha-0001" \
     -F "files=@sample_data/halcyon_employee_handbook.pdf"
curl -s -X POST localhost:8000/query -H "X-API-Key: dev-key-alpha-0001" \
     -H "Content-Type: application/json" \
     -d '{"question": "How many days of paid time off do employees get?"}'
```

## 2. Docker

### Image

`Dockerfile` is a two-stage build on `python:3.14-slim`:

1. **builder**: installs the locked dependencies with `uv sync --frozen --no-dev`
   (bytecode pre-compiled), then runs `scripts/download_models.py` so the embedding model is
   baked into the image under `/app/.hf_cache`.
2. **runtime**: a clean slim base with only the virtualenv, code, sample data, evaluation
   dataset and model cache; no uv, no package cache, no tests. Runs as the non-root user
   `contextiq` (uid 1001) with a read-only code tree. `HF_HUB_OFFLINE=1` is set so startup
   never contacts Hugging Face.

Measured size (linux/amd64, Python 3.14): **3.02 GB on disk, 689 MB compressed**
(`docker image ls contextiq:local`). The virtualenv is 2.0 GB of that: PyTorch CPU 776 MB,
pre-compiled bytecode 335 MB, transformers 120 MB, PyMuPDF 65 MB, FAISS 66 MB, Streamlit 35 MB;
the embedding model cache is 129 MB and the Python base image about 120 MB. Dropping
`UV_COMPILE_BYTECODE=1` saves ~335 MB at the cost of slower cold starts.

Build arguments: `PYTHON_VERSION` (default `3.14`) and `UV_VERSION` (default `latest`), e.g.
`docker build --build-arg PYTHON_VERSION=3.12 -t contextiq:local .`

### Compose (API + UI)

```bash
cp .env.example .env            # optional but recommended: sets CONTEXTIQ_API_KEYS
docker compose up -d --build    # or: make docker-build && make docker-up
docker compose logs -f api
open http://localhost:8501      # UI;  API docs: http://localhost:8000/docs
docker compose down             # stop; `down -v` also deletes the indexed data
```

`docker-compose.yml` defines:

* `api`: builds `contextiq:local`, publishes `8000`, mounts the named volume
  `contextiq-data` at `/app/data`, loads `./.env` when present (`required: false`), and has a
  health check on `GET /health`.
* `ui`: same image, `streamlit run ... --server.address 0.0.0.0`, publishes `8501`, talks to
  the API at `http://api:8000` over the compose network, starts only after `api` is healthy.

Without a `.env` the API starts with **no API keys configured** and rejects every
authenticated request with 401 (health still works). Provide keys either in `.env` or
inline, e.g. `CONTEXTIQ_API_KEYS=mykey-0123456789:alpha docker compose up -d` will *not*
work because compose only forwards variables that are listed; use an override file instead:

```yaml
# docker-compose.override.yml (picked up automatically)
services:
  api:
    environment:
      CONTEXTIQ_API_KEYS: mykey-0123456789:alpha
  ui:
    environment:
      CONTEXTIQ_UI_API_KEY: mykey-0123456789
```

If 8000 or 8501 is taken on the host, move the published ports (container ports stay fixed):

```bash
CONTEXTIQ_API_PORT=18000 CONTEXTIQ_UI_PORT=18501 docker compose -p contextiq-alt up -d
```

### Single container (API only)

```bash
docker build -t contextiq:local .
docker run -d --name contextiq-api -p 8000:8000 \
  -e CONTEXTIQ_API_KEYS=mykey-0123456789:alpha \
  -v contextiq-data:/app/data contextiq:local
```

Run the UI against it with `docker run -p 8501:8501 -e CONTEXTIQ_API_URL=http://<api-host>:8000
contextiq:local streamlit run app/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0`.

CLI scripts work inside the image too (same volume, API stopped for index rebuilds):

```bash
docker compose run --rm api python scripts/ingest.py --workspace alpha sample_data/*.pdf
docker compose run --rm api python scripts/query.py --workspace alpha "What is the PTO policy?"
docker compose run --rm api python scripts/evaluate.py --provider extractive --output-dir /app/data/eval
```

## 3. Environment variables

All application settings are prefixed `CONTEXTIQ_` (see `app/core/config.py` for the full
list and validation rules). Provider keys use their conventional names. Defaults below are the
application defaults; `.env.example` documents the same set.

| Variable | Default | Meaning |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GROQ_API_KEY` | unset | LLM provider credentials. With none set the API runs in **extractive** mode (verbatim cited passages, no synthesis). |
| `CONTEXTIQ_LLM_PROVIDER` | `auto` | `auto` picks the first provider with a key; or `anthropic`, `openai`, `groq`, `extractive`. |
| `CONTEXTIQ_LLM_MODEL` | provider default | `claude-opus-5` / `gpt-5-mini` / `openai/gpt-oss-120b`. |
| `CONTEXTIQ_LLM_TEMPERATURE` | `0.0` | Sampling temperature (ignored for Anthropic models). |
| `CONTEXTIQ_LLM_MAX_TOKENS` | `1024` | Max completion tokens. |
| `CONTEXTIQ_LLM_TIMEOUT_SECONDS` | `60` | Per-call LLM timeout. |
| `CONTEXTIQ_AUTH_MODE` | `api_key` | `api_key` or `disabled` (single shared workspace; local development only). |
| `CONTEXTIQ_API_KEYS` | empty | Comma-separated `key:workspace` pairs. Keys must be at least 8 characters; each key is scoped to exactly one workspace (`^[a-z0-9][a-z0-9_-]{0,63}$`). |
| `CONTEXTIQ_UI_API_KEY` | unset | Key the Streamlit UI pre-fills. |
| `CONTEXTIQ_DATA_DIR` | `data` (`/app/data` in the image) | Root for `contextiq.db` (SQLite) and `indexes/<workspace>.faiss`. |
| `CONTEXTIQ_MAX_UPLOAD_MB` | `25` | Per-file upload limit (1-500). |
| `CONTEXTIQ_MAX_PAGES` | `500` | PDFs with more pages are rejected. |
| `CONTEXTIQ_MAX_FILES_PER_UPLOAD` | `10` | Files per `POST /documents`. |
| `CONTEXTIQ_CHUNK_SIZE` / `CONTEXTIQ_CHUNK_OVERLAP` | `1000` / `150` | Chunking in characters; overlap must be smaller than size. |
| `CONTEXTIQ_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-Transformers model (384-d). Changing it requires an index rebuild and, in Docker, a model download (see below). |
| `CONTEXTIQ_EMBEDDING_DEVICE` | `cpu` | `cpu`, `cuda`, `mps` or `auto`. The image ships CPU-only PyTorch. |
| `CONTEXTIQ_EMBEDDING_BATCH_SIZE` | `32` | Passages per embedding batch. |
| `CONTEXTIQ_EMBEDDING_CACHE_DIR` | HF default | Model cache folder override (the image uses `HF_HOME=/app/.hf_cache`). |
| `CONTEXTIQ_RETRIEVAL_MODE` | `hybrid` | `dense` (FAISS cosine) or `hybrid` (dense + BM25 fused with RRF). |
| `CONTEXTIQ_TOP_K` / `CONTEXTIQ_MAX_TOP_K` | `5` / `20` | Default and maximum number of chunks per query. |
| `CONTEXTIQ_RERANK_ENABLED` | `false` | Cross-encoder reranking of the candidates. |
| `CONTEXTIQ_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model (not pre-downloaded in the image; run `scripts/download_models.py --rerank` with `HF_HUB_OFFLINE=0`). |
| `CONTEXTIQ_LOG_LEVEL` / `CONTEXTIQ_LOG_JSON` | `INFO` / `true` | Log level; JSON lines (`true`) or a console renderer. |
| `CONTEXTIQ_CORS_ORIGINS` | `http://localhost:8501` | Comma-separated allowed browser origins. |
| `CONTEXTIQ_API_URL` | `http://localhost:8000` | Backend URL used by the UI (compose sets `http://api:8000`). |
| `CONTEXTIQ_HOST` / `CONTEXTIQ_PORT` | `0.0.0.0` / `8000` | Bind address for the `contextiq-api` console script and `scripts/dev.sh`. |
| `HF_HUB_OFFLINE` | `1` in the image | Never contact Hugging Face at runtime. Set to `0` when downloading a new model. |
| `CONTEXTIQ_API_PORT` / `CONTEXTIQ_UI_PORT` | `8000` / `8501` | **Compose only**: host ports to publish. |

## 4. Volumes and persistence

Everything the service writes lives under `CONTEXTIQ_DATA_DIR`:

```
data/
  contextiq.db          # SQLite: documents, chunks (+ FTS5 index for BM25), WAL files
  indexes/
    <workspace>.faiss   # one exact-cosine FAISS index per workspace
```

* In compose this is the named volume `contextiq_contextiq-data`; `docker compose down` keeps
  it, `docker compose down -v` deletes it. To use a host directory instead, replace the volume
  with a bind mount (`./data:/app/data`) and make it writable by uid 1001.
* The SQLite file is the source of truth; FAISS files can always be regenerated from it
  (next section). Back up the whole directory while the API is stopped, or copy
  `contextiq.db` with `sqlite3 contextiq.db ".backup backup.db"` while it runs.
* Uploaded PDFs are **not** stored; only their extracted text, chunk metadata and SHA-256.
* Run exactly one API process per data directory. SQLite uses WAL mode and FAISS files are
  replaced atomically, but two writers on the same FAISS index would overwrite each other.

## 5. Rebuilding indexes

`scripts/rebuild_index.py` re-embeds every chunk stored in SQLite and atomically replaces the
workspace's `.faiss` file. Use it when an index file was lost or corrupted, or after changing
`CONTEXTIQ_EMBEDDING_MODEL` (vectors from different models are not comparable, so rebuild
**every** workspace by omitting `--workspace`). Stop the API first.

```bash
# locally
.venv/bin/python scripts/rebuild_index.py                    # all workspaces
.venv/bin/python scripts/rebuild_index.py --workspace alpha

# in compose (same volume)
docker compose stop api ui
docker compose run --rm api python scripts/rebuild_index.py
docker compose start api ui
```

Switching to a model that is not in the image requires a download at build or run time:
either rebuild the image with `CONTEXTIQ_EMBEDDING_MODEL` set as an environment variable for
the `download_models.py` step, or run
`docker compose run --rm -e HF_HUB_OFFLINE=0 api python scripts/download_models.py` (the
cache at `/app/.hf_cache` is writable by the service user but lives in the container layer,
so mount a volume there to keep it).

## 6. Resource expectations

Measured with the default configuration (bge-small-en-v1.5 on CPU, hybrid retrieval, five
sample PDFs, 82 chunks):

| Item | Value |
|------|-------|
| Embedding model on disk | ~130 MB (`BAAI/bge-small-en-v1.5`, safetensors) |
| Docker image | 3.02 GB on disk (689 MB compressed) |
| API RAM | 1.36 GiB measured (`docker stats`) after loading the model, indexing one 7-page PDF and answering one query; expect spikes while embedding large uploads |
| UI RAM | ~70 MiB measured idle; grows with open sessions |
| API cold start | ~3 s to `app.started` in the container (model load); the compose health check allows 20 s start-up plus 12 retries |
| Retrieval latency | p50 ~17 ms, p95 ~24 ms for hybrid search on the sample corpus (see `evaluation/results/latest.md`) |
| Ingestion | dominated by embedding: tens of pages per second on a modern CPU |

Storage grows with the text of the indexed documents (SQLite) plus `4 * 384` bytes per chunk
for the FAISS index. Give the API container at least 2 GB of RAM; CPU-only inference
benefits from several cores (`OMP_NUM_THREADS` can pin PyTorch to fewer threads on shared hosts).

## 7. Security caveats before exposing the API

* **Transport**: the API speaks plain HTTP. Put it behind a TLS-terminating reverse proxy
  (nginx, Caddy, a cloud load balancer) and never publish port 8000 directly to the internet.
* **API keys are bearer secrets**: anyone holding a key owns that workspace (read, upload,
  delete). Generate long random keys (`python -c "import secrets; print(secrets.token_urlsafe(32))"`),
  pass them through the environment or a secrets manager, never commit `.env`, and rotate by
  editing `CONTEXTIQ_API_KEYS` and restarting.
* **`CONTEXTIQ_AUTH_MODE=disabled`** removes authentication entirely and maps every caller to
  one workspace. Local development only.
* **Workspace isolation** is enforced by the credential (one FAISS index per workspace, every
  SQLite query filtered by workspace), but the process is shared: a tenant that needs a hard
  boundary should get its own container and data directory.
* **`GET /health` is unauthenticated** and reports the configured models and modes (never
  secrets or document data). Restrict it at the proxy if that is too much information.
* **The Streamlit UI has no authentication of its own** and keeps the API key in the browser
  session. Do not expose port 8501 publicly; front it with SSO or restrict it to a private
  network.
* **CORS**: `CONTEXTIQ_CORS_ORIGINS` defaults to the local UI origin. Widen it deliberately;
  do not use `*` together with credentials.
* **LLM providers receive document text**: retrieved passages are sent to the configured
  provider as part of the prompt. Use extractive mode or a provider you are allowed to send
  the documents to. Document text is treated as untrusted data (delimited source blocks,
  instructions inside documents are ignored by prompt design), but no prompt is a security
  boundary; the verified-citation check is what keeps answers grounded.
* **Uploads**: only PDFs are accepted, size and page limits apply (`CONTEXTIQ_MAX_UPLOAD_MB`,
  `CONTEXTIQ_MAX_PAGES`, `CONTEXTIQ_MAX_FILES_PER_UPLOAD`), and files are parsed with PyMuPDF
  in the API process. There is no rate limiting; add it at the proxy.
* **Container**: runs as uid 1001 with a root-owned code tree; the only writable paths are
  `/app/data`, `/app/evaluation/results` and the model cache. `HF_HUB_OFFLINE=1` keeps the
  runtime from making outbound calls except to the configured LLM provider.
* **Logs** contain request ids, counts, durations and error types, never document text,
  queries or keys; they are safe to ship to a central log system.
