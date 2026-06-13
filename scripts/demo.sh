#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv sync --python 3.11 --extra dev --extra faiss

uv run --python 3.11 --extra dev uvicorn app.main:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!
trap 'kill "$SERVER_PID"' EXIT

sleep 2

curl -s -X POST http://127.0.0.1:8000/v1/index/markdown \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "demo",
    "markdown": "# Demo\nWelcome\n\n## Install\nInstall with uv sync"
  }'

echo
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I install it?", "k_recall": 10, "k_candidates": 5, "k_final": 1}' | python -m json.tool
