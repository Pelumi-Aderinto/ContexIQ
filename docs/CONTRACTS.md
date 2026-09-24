# ContextIQ module contracts

This file is the source of truth for how modules fit together. Every module MUST expose
exactly these names and signatures (extra private helpers are fine). Public API schemas live in
`app/models/schemas.py`; internal domain models in `app/models/domain.py`; settings in
`app/core/config.py`. **Read those three files before writing any code.** Do not change them
without a very good reason; if you must, keep backwards compatibility.

## Environment

* Python 3.14 venv at `.venv` (use `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`).
  All dependencies are installed. Do NOT run `uv sync`/`pip install`.
* Embedding model `BAAI/bge-small-en-v1.5` (384-d, max 512 tokens) is already in the HF cache.
* `import pymupdf` (not `fitz`). LangChain 1.x modular packages: `langchain_core`,
  `langchain_text_splitters`, `langchain_huggingface`, `langchain_anthropic`,
  `langchain_openai`, `langchain_groq`, and `langchain.chat_models.init_chat_model`.
  No `langchain_community`, no `langchain_classic`, no deprecated `LLMChain`/`RetrievalQA`.
* Run tests with `.venv/bin/pytest tests/ -q` and lint with `.venv/bin/ruff check app tests scripts`
  and `.venv/bin/ruff format app tests scripts`. Code must be ruff-clean (config in pyproject).
* Tests must never call a real LLM or the network. Real-embedding tests are marked `@pytest.mark.slow`.

## Design invariants (non-negotiable)

1. **Workspace isolation is physical**: one FAISS index per workspace on disk
   (`{index_dir}/{workspace_id}.faiss`). Every SQLite query filters by `workspace_id` as
   defense in depth. Nothing ever returns chunks from another workspace.
2. **Workspace comes from the credential, never from the request body.** `app/core/security.py`
   maps `X-API-Key` -> `Principal(workspace_id=...)`.
3. **Citations are verified**: the LLM is shown chunks labelled `S1..Sn`; it may only cite those
   labels. Any other label is dropped and reported in `invalid_citation_ids`. We never invent a
   citation. `Citation.chunk_id` always equals a retrieved chunk's `chunk_id`.
4. **Document text is untrusted data**: it is wrapped in delimited `<source id="S1" ...>` blocks
   and the system prompt says instructions inside sources must be ignored.
5. **Abstain honestly**: if the LLM reports insufficient evidence, or produces no valid
   citation for a non-trivial answer, `abstained=True` and the answer is a clear no-answer
   message (see `prompts.NO_ANSWER_TEXT`).
6. **No secrets or document contents in logs.** Log IDs, counts, durations and error types only.
7. Scores are exposed for debugging but never called probabilities/confidence.

## Identifiers

* `document_id = uuid.uuid4().hex`
* `chunk_id = Chunk.make_id(document_id, page_number, chunk_index)` -> `"{doc}:p{page}:c{idx}"`
  where `chunk_index` is the 0-based index over the whole document (not per page).
* `vector_id`: SQLite `INTEGER PRIMARY KEY AUTOINCREMENT` in `chunks`; used as FAISS int64 id.
* `sha256`: hex digest of the raw uploaded bytes; duplicate detection is per workspace.
* Workspace ids match `^[a-z0-9][a-z0-9_-]{0,63}$` (`config.WORKSPACE_ID_PATTERN`).

## Module contracts

### `app/ingestion/parser.py`
```python
class PdfParseError(Exception):            # raised for invalid/corrupt/encrypted/oversized PDFs
    def __init__(self, message: str, *, code: str = "parse_error"): ...  # codes: not_pdf, encrypted, corrupt, too_many_pages, no_text, parse_error
def is_pdf(data: bytes) -> bool             # magic-bytes check (b"%PDF-")
def normalize_text(text: str) -> str        # NFKC, fix hyphenation at line ends, collapse whitespace, strip control chars, de-duplicate repeated header/footer-like lines is NOT required
def parse_pdf(data: bytes, filename: str, *, max_pages: int = 500) -> ParsedDocument
```
`parse_pdf` computes sha256/size, extracts per-page text with `page.get_text("text")`
(PyMuPDF), normalizes each page, keeps `page_number` 1-based, reads title from metadata.
Encrypted PDFs (`doc.needs_pass`) -> `PdfParseError(code="encrypted")`. A PDF whose pages are
all empty (scanned/image-only) -> `PdfParseError(code="no_text")`.

### `app/ingestion/chunker.py`
```python
def chunk_document(parsed: ParsedDocument, *, document_id: str, workspace_id: str,
                   chunk_size: int, chunk_overlap: int) -> list[Chunk]
```
Uses `langchain_text_splitters.RecursiveCharacterTextSplitter` **per page** so a chunk never
spans pages (page citations stay exact). `char_start`/`char_end` are offsets into the page's
normalized text (use `splitter.split_text` and locate each piece with `str.find` from a moving
cursor). Empty pages produce no chunks. `chunk_index` increments across the document.

### `app/storage/metadata_store.py` (SQLite, thread-safe via one connection + RLock)
```python
class MetadataStore:
    def __init__(self, db_path: Path) -> None            # creates schema; ":memory:" allowed
    def close(self) -> None
    # documents
    def create_document(self, doc: DocumentInfo) -> None
    def get_document(self, workspace_id: str, document_id: str) -> DocumentInfo | None
    def list_documents(self, workspace_id: str) -> list[DocumentInfo]      # newest first
    def find_by_sha256(self, workspace_id: str, sha256: str) -> DocumentInfo | None
    def update_document(self, workspace_id: str, document_id: str, *, status: DocumentStatus,
                        error: str | None = None, chunk_count: int | None = None,
                        page_count: int | None = None) -> None
    def delete_document(self, workspace_id: str, document_id: str) -> list[int]   # deletes doc + its chunks, returns removed vector_ids ([] if doc missing)
    # chunks
    def add_chunks(self, chunks: list[Chunk]) -> list[int]            # returns vector_ids in input order
    def get_chunks(self, workspace_id: str, vector_ids: list[int]) -> dict[int, Chunk]
    def get_chunk_by_id(self, workspace_id: str, chunk_id: str) -> Chunk | None
    def list_vector_ids(self, workspace_id: str, document_ids: list[str] | None = None) -> list[int]
    def keyword_search(self, workspace_id: str, query: str, k: int,
                       document_ids: list[str] | None = None) -> list[tuple[int, float]]   # FTS5 bm25; returns (vector_id, bm25_score) best first; sanitizes the query into a safe OR-of-terms MATCH expression; never raises on odd input (returns [])
    def iter_chunks(self, workspace_id: str) -> Iterator[tuple[int, Chunk]]     # for index rebuild
    def count_chunks(self, workspace_id: str) -> int
    def list_workspaces(self) -> list[str]
```
Schema: `documents(document_id TEXT PK, workspace_id, filename, sha256, size_bytes, page_count,
chunk_count, status, error, created_at TEXT ISO)`, indexes on (workspace_id), (workspace_id, sha256);
`chunks(vector_id INTEGER PK AUTOINCREMENT, chunk_id TEXT UNIQUE, document_id, workspace_id,
filename, page_number, chunk_index, text, char_start, char_end)`, index on (workspace_id, document_id);
`chunks_fts` = `fts5(text, content='chunks', content_rowid='vector_id')` kept in sync with
triggers. Use `PRAGMA journal_mode=WAL` for file DBs. Filter every chunk query by workspace_id.

### `app/retrieval/embeddings.py`
```python
class Embedder(Protocol):
    model_name: str
    dimension: int
    def embed_documents(self, texts: list[str]) -> np.ndarray   # shape (n, dim) float32, L2-normalized
    def embed_query(self, text: str) -> np.ndarray              # shape (dim,) float32, L2-normalized

class SentenceTransformerEmbedder:   # wraps langchain_huggingface.HuggingFaceEmbeddings
    def __init__(self, model_name: str, *, device: str = "cpu", batch_size: int = 32,
                 query_prefix: str = "", cache_dir: Path | None = None) -> None
class HashingEmbedder:               # deterministic, dependency-free fake for tests (token hashing -> dim 64, normalized)
    def __init__(self, dimension: int = 64) -> None
def create_embedder(settings: Settings) -> Embedder
```
`embed_documents([])` returns an empty `(0, dim)` array. Device `"auto"` picks cuda/mps/cpu.

### `app/retrieval/vector_store.py` (FAISS)
```python
class VectorStore:
    def __init__(self, index_dir: Path, dimension: int) -> None      # loads any existing *.faiss on init
    def add(self, workspace_id: str, vector_ids: list[int], vectors: np.ndarray) -> None   # persists
    def remove(self, workspace_id: str, vector_ids: list[int]) -> int                       # persists, returns removed count
    def search(self, workspace_id: str, query: np.ndarray, k: int,
               allowed_vector_ids: set[int] | None = None) -> list[tuple[int, float]]      # (vector_id, cosine) best first; [] for unknown workspace; uses IDSelectorBatch when allowed_vector_ids given
    def count(self, workspace_id: str) -> int
    def workspaces(self) -> list[str]
    def drop_workspace(self, workspace_id: str) -> None
    def save(self, workspace_id: str) -> None                          # atomic: write tmp then os.replace
    def rebuild(self, workspace_id: str, vector_ids: list[int], vectors: np.ndarray) -> None
    load_errors: dict[str, str]   # property: index files found at start-up that could not be loaded ({workspace: reason}); /health reports 'degraded' with a warning per entry
```
Index type: `faiss.IndexIDMap2(faiss.IndexFlatIP(dim))` per workspace (exact cosine on
normalized vectors). File name = `f"{workspace_id}.faiss"`. A per-workspace `threading.RLock`.
Vectors are `float32`, C-contiguous. Validate `vectors.shape[1] == dimension`.

### `app/ingestion/pipeline.py`
```python
class IngestionPipeline:
    def __init__(self, *, settings: Settings, store: MetadataStore, vector_store: VectorStore,
                 embedder: Embedder) -> None
    def ingest(self, data: bytes, filename: str, workspace_id: str) -> DocumentUploadResult
    def delete_document(self, workspace_id: str, document_id: str) -> DeleteDocumentResponse | None  # None if not found; request_id filled by the route (pass "" here)
    def reconcile_interrupted(self) -> int   # start-up repair: documents still 'processing' are marked failed (INTERRUPTED_ERROR), orphan chunks/vectors removed; returns count
```
`ingest` flow: size check (`settings.max_upload_bytes`) -> `is_pdf` -> sha256 duplicate check
(`outcome=duplicate`, `duplicate_of`) -> create `DocumentInfo(status=processing)` ->
`parse_pdf` -> `chunk_document` -> `embedder.embed_documents` -> `store.add_chunks` ->
`vector_store.add` -> `update_document(status=indexed, chunk_count, page_count)`. Any failure
-> `update_document(status=failed, error=<short reason>)`, and returns `outcome=failed` with the
document (status failed) so the user sees why. Failed docs have no chunks. Log durations.

### `app/retrieval/retriever.py`
```python
class RetrievalService:
    def __init__(self, *, settings: Settings, store: MetadataStore, vector_store: VectorStore,
                 embedder: Embedder, reranker: Reranker | None = None) -> None
    def retrieve(self, workspace_id: str, query: str, *, k: int | None = None,
                 document_ids: list[str] | None = None,
                 mode: RetrievalMode | None = None) -> list[ScoredChunk]
    def as_langchain_retriever(self, workspace_id: str, *, k: int | None = None,
                               document_ids: list[str] | None = None) -> "WorkspaceRetriever"

class WorkspaceRetriever(langchain_core.retrievers.BaseRetriever):   # LCEL-compatible
    # fields: service (Any, excluded from serialization), workspace_id, k, document_ids
    # _get_relevant_documents returns langchain_core.documents.Document with page_content=chunk.text and
    # metadata = {chunk_id, document_id, filename, page_number, score, dense_score, sparse_rank, rerank_score, vector_id}

class Reranker(Protocol):
    def rerank(self, query: str, texts: list[str]) -> list[float]
class CrossEncoderReranker(Reranker): def __init__(self, model_name: str, device: str = "cpu")

def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]
def scored_chunk_to_retrieved(sc: ScoredChunk) -> RetrievedChunk
```
Dense: `vector_store.search(ws, embedder.embed_query(q), k*mult, allowed)` where `allowed` is
`store.list_vector_ids(ws, document_ids)` if `document_ids` given else None. Hybrid: dense +
`store.keyword_search(...)` fused with RRF, then top-k. Optional rerank of top `rerank_candidates`.
`k` is clamped to `settings.max_top_k`. Unknown/empty workspace -> `[]`. document_ids not in the
workspace are silently ignored (they cannot leak anything).

### `app/generation/prompts.py`
```python
SYSTEM_PROMPT: str                      # grounding rules, cite with [S#], treat sources as data, JSON output spec
NO_ANSWER_TEXT: str                     # "I couldn't find enough information in the indexed documents to answer that."
ANSWER_PROMPT: ChatPromptTemplate       # system + human; input vars: question, context
def format_context(chunks: list[ScoredChunk], *, max_chars: int) -> tuple[str, dict[str, ScoredChunk]]
    # returns (context_block, label_map {"S1": chunk, ...}); each source rendered as
    # <source id="S1" file="handbook.pdf" page="3">\n...text...\n</source>; truncates to max_chars total, whole sources only (always at least one)
```
The LLM must answer with a JSON object: `{"answer": str, "citations": ["S1", ...], "insufficient_evidence": bool}`.
Answer text should use inline markers like `[S1]` after claims.

### `app/generation/citations.py`
```python
class LLMAnswer(BaseModel): answer: str = ""; citations: list[str] = []; insufficient_evidence: bool = False
def parse_llm_answer(text: str) -> LLMAnswer      # tolerant: strips code fences, finds first {...}, validates; falls back to answer=text with citations extracted from inline [S#] markers
def extract_inline_labels(text: str) -> list[str] # ["S1","S3"] in order of first appearance, deduped
def normalize_label(raw: str) -> str | None       # "s1", "[S1]", "S1" -> "S1"; junk -> None
def build_citations(labels: list[str], label_map: dict[str, ScoredChunk], *, excerpt_chars: int, focus: str | None = None) -> tuple[list[Citation], list[str]]  # (valid citations in label order, invalid labels)
def make_excerpt(text: str, max_chars: int, *, focus: str | None = None) -> str   # with focus (the question) the excerpt starts at the sentence sharing the most distinctive terms, prefixed "... " when not at the start
```

### `app/generation/llm.py`
```python
def create_chat_model(settings: Settings) -> BaseChatModel | None   # None => extractive mode
```
Uses `init_chat_model(model, model_provider=provider, ...)`. **Do NOT pass `temperature` for
Anthropic models** (Claude 5-family rejects sampling params); pass it for openai/groq. Pass
`timeout`/`max_retries` where the provider supports them; `max_tokens` for all.

### `app/generation/chain.py`
```python
class AnswerService:
    def __init__(self, *, settings: Settings, retrieval: RetrievalService,
                 llm: BaseChatModel | None) -> None
    def answer(self, workspace_id: str, request: QueryRequest, *, request_id: str) -> QueryResponse
    def search(self, workspace_id: str, request: SearchRequest, *, request_id: str) -> SearchResponse
```
LCEL chain: `ANSWER_PROMPT | llm | StrOutputParser() | RunnableLambda(parse_llm_answer)`.
Post-processing: `build_citations` -> if `insufficient_evidence` or (no valid citations) ->
`abstained=True`, `answer=NO_ANSWER_TEXT`, citations=[]. No retrieved chunks -> abstain without
calling the LLM. LLM errors -> raise `GenerationError` (the API maps it to 502). Extractive
mode (`llm is None`): answer is a short header + the top 3 excerpts, each cited, `answer_mode=extractive`,
`abstained` only if nothing retrieved. Record `retrieval_ms`, `generation_ms`, `total_ms`.

### `app/core/security.py`
```python
class Principal(BaseModel): workspace_id: str; key_label: str   # key_label = first 4 chars + "..." (safe for logs)
class AuthError(Exception)
def validate_workspace_id(workspace_id: str) -> str          # raises ValueError
def authenticate(api_key: str | None, key_map: dict[str, str]) -> Principal   # constant-time compare over all keys; raises AuthError
def get_principal(request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> Principal   # FastAPI dependency; auth_mode disabled -> Principal(default_workspace_id, "disabled"); raises HTTPException 401
```

### `app/core/logging.py`
```python
def configure_logging(settings: Settings) -> None       # structlog JSON or console
def get_logger(name: str) -> structlog.stdlib.BoundLogger
class RequestContextMiddleware(BaseHTTPMiddleware)       # sets request_id (X-Request-ID header or uuid4 hex), binds contextvars, logs method/path/status/duration_ms, adds X-Request-ID response header
```

### `app/api/main.py`
```python
class Services:   # simple container stored on app.state.services
    settings, store, vector_store, embedder, retrieval, ingestion, answer, llm_model_name: str | None
def build_services(settings: Settings, *, embedder: Embedder | None = None, llm: BaseChatModel | None | Literal["auto"] = "auto") -> Services
def create_app(settings: Settings | None = None, *, embedder=None, llm="auto") -> FastAPI
app = create_app()   # module-level for `uvicorn app.api.main:app`
def run() -> None    # uvicorn entrypoint
```
Lifespan: `build_services` (loads embedder, opens store, loads FAISS indexes). Exception
handlers map `AuthError`->401, `PdfParseError`/`ValueError`->422/400, `GenerationError`->502,
unexpected->500 with `ErrorResponse` bodies including `request_id`. CORS from settings.

### `app/api/routes/documents.py`  (prefix `/documents`, all require `get_principal`)
* `POST /documents` multipart `files: list[UploadFile]` -> `201 DocumentUploadResponse` (207-like semantics: per-file outcomes; HTTP 201 if any indexed, 200 if only duplicates, 422 if all failed). Enforce `max_files_per_upload` and `max_upload_bytes` per file (read in chunks, stop early -> 413).
* `GET /documents` -> `DocumentListResponse`
* `GET /documents/{document_id}` -> `DocumentInfo` (404 if not in caller's workspace)
* `DELETE /documents/{document_id}` -> `DeleteDocumentResponse` (404 if not in caller's workspace)

### `app/api/routes/query.py`
* `POST /query` -> `QueryResponse`
* `POST /search` -> `SearchResponse`

### `app/api/routes/health.py`
* `GET /health` -> `HealthResponse` (no auth). `status="degraded"` and `warnings=[...]` when a persisted index could not be loaded (e.g. embedding model changed); otherwise `ok` with `warnings=[]`.

### `app/evaluation/metrics.py`
```python
def recall_at_k(retrieved: list[tuple[str,int]], expected: list[tuple[str,int]], k: int) -> float   # (filename, page) pairs; fraction of expected pages found in top-k
def hit_at_k(...) -> bool; def mrr(...) -> float
def citation_validity(citations: list[Citation], retrieved_chunk_ids: set[str]) -> float          # 1.0 if all valid (or no citations), else fraction
def citation_page_accuracy(citations, expected_pages: list[tuple[str,int]]) -> float
def answer_correctness(answer: str, expected_keywords: list[str], *, abstained: bool, answerable: bool) -> float   # keyword coverage for answerable; 1.0 if abstained on unanswerable, 0.0 if answered an unanswerable question
def latency_summary(values_ms: list[float]) -> dict[str, float]   # p50, p95, mean, max
```
### `evaluation/dataset.json`
```json
{"version": 1, "documents": ["sample_data/<file>.pdf", ...],
 "questions": [{"id": "q01", "question": "...", "type": "answerable|unanswerable|cross_document",
                "expected_sources": [{"filename": "x.pdf", "page": 3}], "expected_keywords": ["..."],
                "reference_answer": "...", "notes": "..."}]}
```
### `scripts/`
`generate_sample_data.py` (creates PDFs + `sample_data/manifest.json` with per-page section titles),
`ingest.py` (CLI: ingest files into a workspace), `query.py` (CLI question), `evaluate.py`
(runs dataset in-process, writes `evaluation/results/latest.json` + prints a markdown table),
`rebuild_index.py` (re-embed a workspace from SQLite), `download_models.py`, `dev.sh` (api + ui).
