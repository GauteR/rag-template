# Install

Install the template with uv:

```bash
uv sync --python 3.11 --extra dev
```

## Query

Ask questions through the API:

```bash
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I install the template?"}'
```
