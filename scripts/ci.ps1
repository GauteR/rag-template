$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = if ($env:PYTHON) { $env:PYTHON } else { "3.11" }

Write-Host "==> uv sync"
uv sync --python $Python --extra dev --extra faiss

Write-Host "==> pre-commit (commit hooks)"
uv run pre-commit run --all-files

Write-Host "==> pre-commit (pre-push hooks)"
uv run pre-commit run --all-files --hook-stage pre-push

Write-Host ""
Write-Host "CI passed."
