from __future__ import annotations

import json


def parse_json_object(content: str) -> dict[str, object]:
    """Extract the first JSON object from *content* and return it as a dict.

    Returns an empty dict when no valid JSON object is found.
    """
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
