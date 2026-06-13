$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

uv sync --python 3.11 --extra dev --extra faiss

$server = Start-Process -PassThru -NoNewWindow `
  uv run --python 3.11 --extra dev uvicorn app.main:app --host 127.0.0.1 --port 8000

try {
    Start-Sleep -Seconds 2

    $indexBody = @{
        doc_id = "demo"
        markdown = "# Demo`nWelcome`n`n## Install`nInstall with uv sync"
    } | ConvertTo-Json

    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/index/markdown" `
      -ContentType "application/json" -Body $indexBody

    $queryBody = @{
        question = "How do I install it?"
        k_recall = 10
        k_candidates = 5
        k_final = 1
    } | ConvertTo-Json

    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/query" `
      -ContentType "application/json" -Body $queryBody | ConvertTo-Json -Depth 6
}
finally {
    Stop-Process -Id $server.Id -Force
}
