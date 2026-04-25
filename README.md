# rag-template

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
uv sync --python 3.11 --extra dev
```

Copy the example environment file:

```bash
cp .env.example .env
```

The default configuration works without external API keys:

```bash
LLM_PROVIDER=echo
EMBEDDING_PROVIDER=hash
EMBEDDING_DIMENSION=8
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
5. Stores vectors and full section text for pointer retrieval.

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

Responses include an answer and traceable sources with `doc_id`, `node_id`, breadcrumbs, score, and
the full section text used for synthesis.

## Configuration

Common settings in `.env`:

```bash
LLM_PROVIDER=echo
LLM_ROUTING_PROVIDER=
LLM_SYNTHESIS_PROVIDER=
EMBEDDING_PROVIDER=hash
EMBEDDING_DIMENSION=8

ENABLE_LLM_NOISE_FILTER=false
ENABLE_LLM_RERANKER=false
ENABLE_LLAMAPARSE=false
ENABLE_BENCHMARK_JUDGE=false

INDEX_DIR=.index
MAX_UPLOAD_MB=5
API_KEY=
```

Provider IDs currently registered:

- LLM: `echo`, `ollama`, `openai_compatible`, `anthropic`
- Embeddings: `hash`, `ollama`, `openai_compatible`

When `API_KEY` is set, requests must include:

```bash
X-API-Key: your-key
```

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

## PDF Indexing

`POST /v1/index/pdf` is behind `ENABLE_LLAMAPARSE`.

The route is intentionally disabled by default. Enabling it requires a configured `PdfExtractorPort`
adapter and `LLAMA_CLOUD_API_KEY`.

## Benchmarks

The benchmark module compares model profiles against the same query pipeline and can write JSON and
CSV artifacts.

Minimal Python usage:

```python
from pathlib import Path

from rag_template_core.application.benchmarking.models import BenchmarkQuestion, ModelProfile
from rag_template_core.application.benchmarking.runner import BenchmarkRunner

# Provide a QueryUseCase instance wired with the providers you want to compare.
runner = BenchmarkRunner(query_use_case_factory=lambda profile: query_use_case)
result = runner.run(
    profiles=[
        ModelProfile(
            name="local-default",
            llm_routing_provider="echo",
            llm_synthesis_provider="echo",
            embedding_provider="hash",
        )
    ],
    questions=[BenchmarkQuestion(id="q1", question="How do I install it?")],
    k_recall=10,
    k_candidates=5,
    k_final=1,
)
runner.write_artifacts(result=result, output_dir=Path("benchmarks/out"))
```

## Test and Format

```bash
uv run --python 3.11 --extra dev pytest
uv run --python 3.11 --extra dev ruff check .
uv run --python 3.11 --extra dev ruff format --check .
```
