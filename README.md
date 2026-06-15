# Proxy Pointer RAG Template

A FastAPI template for a Proxy-Pointer RAG service.

The project indexes Markdown documents by their heading structure, embeds chunks with breadcrumb
context, retrieves candidate nodes, fetches full sections as pointer context, and synthesizes an
answer through a configurable LLM provider.

## Requirements

- Python 3.11+
- `uv`

## Install

Create the environment and install project dependencies:

```bash
uv sync --python 3.11 --extra dev --extra faiss --extra chroma
```

The `chroma` extra is needed for the full pre-push test suite. For API runtime only, `--extra faiss` is enough.

Copy the example environment file:

```bash
cp .env.example .env
```

The default configuration works without external API keys:

```bash
LLM_PROVIDER=echo
EMBEDDING_PROVIDER=hash
EMBEDDING_DIMENSION=8
VECTOR_STORE_PROVIDER=faiss
```

## Quick demo

```bash
bash scripts/demo.sh
```

On Windows:

```powershell
./scripts/demo.ps1
```

## Run the API

Start the FastAPI service:

```bash
uv run --python 3.11 uvicorn app.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Check health:

```bash
curl http://127.0.0.1:8000/v1/health
```

## Run with Docker

Build and run the API container:

```bash
docker build -t rag-template .
docker run --env-file .env -p 8000:8000 rag-template
```

Docker Compose profiles:

```bash
# FAISS only (default)
docker compose up --build rag-api

# ChromaDB sidecar + API wired to Chroma
docker compose --profile chroma up --build
```

Service ports:

- API: `http://127.0.0.1:8000`
- ChromaDB (chroma profile): `http://127.0.0.1:8001`

## Index Markdown

```bash
curl -X POST http://127.0.0.1:8000/v1/index/markdown \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "manual",
    "markdown": "# Intro\nWelcome\n\n## Install\nInstall with uv\n\n## Query\nAsk questions"
  }'
```

The indexing pipeline:

1. Builds a Markdown heading tree.
2. Chunks content within section boundaries.
3. Injects breadcrumb context into embedding text.
4. Filters noisy sections with a heuristic by default.
5. Stores vectors and full section text via atomic `replace_document` (safe re-indexing).

## Query

```bash
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I install it?",
    "k_recall": 10,
    "k_candidates": 5,
    "k_final": 1
  }'
```

Optional query filters:

- `doc_id`: scope search to one document
- `min_score`: drop low-scoring Stage 1 hits

Responses include an answer and traceable sources with `doc_id`, `node_id`, breadcrumbs, score, and
the full section text used for synthesis. With `ENABLE_QUERY_TRACING=true`, responses also include
latency metadata per pipeline stage.

Streaming synthesis (SSE) is available at `POST /v1/query/stream` when
`ENABLE_STREAMING_QUERY=true`.

## Configuration

Common settings in `.env`:

```bash
LLM_PROVIDER=echo
LLM_ROUTING_PROVIDER=
LLM_SYNTHESIS_PROVIDER=
EMBEDDING_PROVIDER=hash
EMBEDDING_DIMENSION=8

VECTOR_STORE_PROVIDER=faiss
SECTION_STORE_PROVIDER=json
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=rag_template

ENABLE_LLM_NOISE_FILTER=false
ENABLE_LLM_RERANKER=false
ENABLE_LLAMAPARSE=true
ENABLE_PDF_INDEXING=true
ENABLE_BENCHMARK_JUDGE=false
ENABLE_INDEX_ADMIN=false
ENABLE_QUERY_TRACING=false
ENABLE_STREAMING_QUERY=false
ENABLE_HYBRID_SEARCH=false
ENABLE_PUBLIC_HEALTH=true

INDEX_DIR=.index
MAX_UPLOAD_MB=5
MAX_QUERY_CHARS=4000
API_KEY=
```

FAISS vectors are persisted under `INDEX_DIR` as `vectors.faiss` (with companion
`vectors.records.json` metadata). Sections are stored in `sections.json` when
`SECTION_STORE_PROVIDER=json`.

### Provider matrix

| Provider ID | Type | Required env | Typical `EMBEDDING_DIMENSION` |
|-------------|------|--------------|--------------------------------|
| `echo` | LLM | none | n/a |
| `ollama` | LLM / embeddings | `OLLAMA_BASE_URL` | 768 (`nomic-embed-text`) |
| `openai_compatible` | LLM / embeddings | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | 1536 (`text-embedding-3-small`) |
| `anthropic` | LLM | `ANTHROPIC_API_KEY` | n/a |
| `hash` | embeddings | none | 8 (demo default) |
| `faiss` | vector store | `INDEX_DIR` | matches embedding dim |
| `chroma` | vector store | `CHROMA_HOST`, `CHROMA_PORT` | matches embedding dim |
| `memory` | vector store | none | in-memory only |
| `json` | section store | `INDEX_DIR` | persisted sections |
| `memory` | section store | none | in-memory only |

When `API_KEY` is set, protected routes require:

```bash
X-API-Key: your-key
```

`/v1/health` is public by default (`ENABLE_PUBLIC_HEALTH=true`). Set
`ENABLE_PUBLIC_HEALTH=false` to require the API key on health as well.

## Provider Examples

Local Ollama:

```bash
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

OpenAI-compatible endpoint:

```bash
LLM_PROVIDER=openai_compatible
EMBEDDING_PROVIDER=openai_compatible
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

Anthropic for synthesis/reranking with local hash embeddings:

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key
ANTHROPIC_MODEL=claude-3-5-haiku-latest
EMBEDDING_PROVIDER=hash
EMBEDDING_DIMENSION=8
```

ChromaDB vector store:

```bash
VECTOR_STORE_PROVIDER=chroma
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

Hybrid BM25 + vector retrieval (Reciprocal Rank Fusion):

```bash
ENABLE_HYBRID_SEARCH=true
uv sync --python 3.11 --extra dev --extra hybrid
```

## PDF Indexing

`POST /v1/index/pdf` is controlled by `ENABLE_PDF_INDEXING=true` (default).

The route accepts multipart form data and runs: PDF bytes → Markdown → existing indexing pipeline.
When `ENABLE_LLAMAPARSE=true` and `LLAMA_CLOUD_API_KEY` is set (with the `llamaparse` extra
installed), extraction uses LlamaParse for richer content such as math, tables, images, and complex
layout. Otherwise it falls back to local text extraction with `pypdf`.

Uploads are validated with a `%PDF-` header and `MAX_UPLOAD_MB`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/index/pdf \
  -F "doc_id=manual" \
  -F "file=@manual.pdf;type=application/pdf"
```

## Index administration

When `ENABLE_INDEX_ADMIN=true`:

- `GET /v1/index` lists indexed documents with section/chunk counts
- `DELETE /v1/index/{doc_id}` removes vectors and sections for a document

## Infrastructure Integrations

The template is built around application ports, so infrastructure can be replaced without changing
the domain or use cases.

### ChromaDB

Install the optional dependency:

```bash
uv sync --python 3.11 --extra dev --extra chroma
```

Set `VECTOR_STORE_PROVIDER=chroma` and point `CHROMA_HOST` / `CHROMA_PORT` at your Chroma instance.

### AI Agents via MCP

Install the optional dependency:

```bash
uv sync --python 3.11 --extra dev --extra mcp
```

Run over stdio:

```bash
uv run --python 3.11 --extra mcp python -m core.mcp_server \
  --base-url http://127.0.0.1:8000 \
  --transport stdio
```

The MCP server verifies API health on startup unless `--skip-health-check` is passed.

Tools exposed:

- `rag_health`: checks `GET /v1/health`.
- `rag_index_markdown`: indexes Markdown through `POST /v1/index/markdown`.
- `rag_index_pdf`: indexes base64-encoded PDF bytes through `POST /v1/index/pdf`.
- `rag_query`: queries `POST /v1/query` and returns answer plus traceable sources.
- `rag_delete_index`: deletes a document when `ENABLE_INDEX_ADMIN=true`.

## Benchmarks

Seed fixture documents and run the model matrix:

```bash
python benchmarks/seed_index.py
python benchmarks/run_matrix.py --live --index-dir benchmarks/.benchmark-index
python benchmarks/run_matrix.py --mock
```

Set `ENABLE_BENCHMARK_JUDGE=true` to score answers with the routing LLM during benchmark runs.

Artifacts are written to `benchmarks/out/` as JSON and CSV.

## Test, format and local CI

### pre-commit (anbefalt)

Installer hooks én gang:

```bash
uv sync --python 3.11 --extra dev --extra faiss --extra chroma
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

| Hook stage | Checks |
|------------|--------|
| `pre-commit` | ruff (lint + format), import-linter |
| `pre-push` | pytest (`not live_models`, 70 % coverage), benchmark mock CLI |

Kjør alle sjekker manuelt:

```bash
uv run pre-commit run --all-files
uv run pre-commit run --all-files --hook-stage pre-push
```

Konfigurasjon: [`.pre-commit-config.yaml`](.pre-commit-config.yaml)

### Full CI-script

Samme sjekker som pre-commit (inkl. tester), samlet i ett script:

```bash
bash scripts/ci.sh
```

On Windows:

```powershell
./scripts/ci.ps1
```

**Merk:** En lokal `.env` med f.eks. `LLM_PROVIDER=ollama` kan påvirke tester som forventer
default `echo`/`hash`. Bruk eksplisitte `Settings(...)` i tester eller midlertidig `.env` uten
provider-overrides ved feilsøking.

Live provider tests (optional, not in pre-commit):

```bash
RUN_LIVE_MODELS=1 uv run --python 3.11 --extra dev pytest -m live_models
```

## CI on GitHub

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs **lint only** on push/PR:

- ruff check
- ruff format
- import-linter

**Tester og mock-benchmark kjøres ikke i GitHub Actions** — de ligger i pre-commit pre-push hooks
og `scripts/ci.sh`. Dette er bevisst: full verifikasjon skjer lokalt ved push.

If Actions is unavailable in your GitHub org, use pre-commit or `scripts/ci.sh` locally instead.
