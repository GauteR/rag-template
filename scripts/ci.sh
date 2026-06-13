#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-3.11}"

echo "==> uv sync"
uv sync --python "$PYTHON" --extra dev --extra faiss

echo "==> pre-commit (commit hooks)"
uv run pre-commit run --all-files

echo "==> pre-commit (pre-push hooks)"
uv run pre-commit run --all-files --hook-stage pre-push

echo
echo "CI passed."
