import json


def extract_json_object(raw: str) -> dict:
    """Extract the first valid JSON object from a raw string (e.g. an LLM response).

    Tries direct parse first; falls back to bracket-scanning so extra prose around
    the JSON block doesn't break extraction.
    Returns an empty dict if no valid JSON object is found.
    """
    if not raw:
        return {}
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = raw.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = raw[start : i + 1]
                try:
                    obj = json.loads(block)
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    return {}
    return {}
