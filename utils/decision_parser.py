import ast
import json
import re
from typing import Dict, Iterable, Optional


DECISION_WORDS = ("LONG", "SHORT", "HOLD", "CANCEL_AND_CLOSE")
NUMERIC_FIELDS = (
    "allocation_pct",
    "leverage",
    "entry_price",
    "stop_loss",
    "take_profit",
    "win_probability_pct",
    "expected_profit_pct",
    "expected_loss_pct",
)
REQUIRED_HINTS = ("decision", "reasoning", "monitoring_playbook", "daily_dual_plan")


def extract_decision_from_response(response: str) -> Optional[Dict]:
    text = _normalize_text(response)
    if not text:
        return None

    for candidate in _iter_candidate_strings(text):
        parsed = _parse_candidate(candidate)
        if _looks_like_decision(parsed):
            return parsed

    salvaged = _salvage_partial_decision(text)
    if _looks_like_decision(salvaged, allow_partial=True):
        return salvaged
    return None


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\ufeff": "",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
    }
    cleaned = str(text)
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    return cleaned.strip()


def _iter_candidate_strings(text: str) -> Iterable[str]:
    seen = set()

    def _push(value: str):
        norm = _normalize_text(value)
        if not norm or norm in seen:
            return
        seen.add(norm)
        yield norm

    for value in _push(text):
        yield value

    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        for value in _push(block):
            yield value

    candidates = []
    brace_positions = [idx for idx, char in enumerate(text) if char == "{"][:40]
    for start_idx in brace_positions:
        depth = 0
        in_string = False
        escape_next = False
        for idx in range(start_idx, len(text)):
            char = text[idx]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start_idx:idx + 1])
                    break

    candidates.sort(key=_candidate_priority, reverse=True)
    for candidate in candidates:
        for value in _push(candidate):
            yield value


def _candidate_priority(candidate: str) -> tuple[int, int]:
    lower = candidate.lower()
    hints = sum(1 for hint in REQUIRED_HINTS if hint in lower)
    return (hints, len(candidate))


def _parse_candidate(candidate: str) -> Optional[Dict]:
    attempts = [
        candidate,
        _strip_code_fence(candidate),
        _sanitize_candidate(candidate),
    ]
    expanded_attempts = []
    for attempt in attempts:
        if attempt:
            expanded_attempts.append(attempt)
            quoted = _quote_unquoted_keys(attempt)
            if quoted != attempt:
                expanded_attempts.append(quoted)

    seen = set()
    for attempt in expanded_attempts:
        normalized = _normalize_text(attempt)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        parsed = _try_json_loads(normalized)
        if isinstance(parsed, dict):
            return parsed

        parsed = _try_literal_eval(normalized)
        if isinstance(parsed, dict):
            return parsed
    return None


def _strip_code_fence(candidate: str) -> str:
    stripped = candidate.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.lower().startswith("json\n"):
        stripped = stripped[5:]
    return stripped.strip()


def _sanitize_candidate(candidate: str) -> str:
    sanitized = _strip_code_fence(candidate)
    start = sanitized.find("{")
    end = sanitized.rfind("}")
    if start != -1 and end != -1 and end > start:
        sanitized = sanitized[start:end + 1]
    sanitized = re.sub(r",(\s*[}\]])", r"\1", sanitized)
    sanitized = re.sub(r":\s*NaN\b", ": null", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r":\s*Infinity\b", ": null", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r":\s*-Infinity\b", ": null", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


def _quote_unquoted_keys(candidate: str) -> str:
    return re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', candidate)


def _try_json_loads(candidate: str) -> Optional[Dict]:
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _try_literal_eval(candidate: str) -> Optional[Dict]:
    try:
        parsed = ast.literal_eval(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _looks_like_decision(payload: Optional[Dict], allow_partial: bool = False) -> bool:
    if not isinstance(payload, dict):
        return False
    if str(payload.get("decision", "")).upper() in DECISION_WORDS:
        return True
    if allow_partial:
        return any(key in payload for key in ("allocation_pct", "entry_price", "reasoning", "final_logic"))
    return False


def _salvage_partial_decision(text: str) -> Optional[Dict]:
    payload: Dict = {}
    upper_text = text.upper()

    decision_match = re.search(
        r"(?:FINAL\s+DECISION|DECISION)\s*[:=]\s*['\"]?(LONG|SHORT|HOLD|CANCEL_AND_CLOSE)\b",
        upper_text,
    )
    if not decision_match:
        # Require JSON key context to avoid matching decision words in reasoning text
        decision_match = re.search(r'"decision"\s*:\s*"(LONG|SHORT|HOLD|CANCEL_AND_CLOSE)"', upper_text)
    if decision_match:
        payload["decision"] = decision_match.group(1).upper()

    for field in NUMERIC_FIELDS:
        value = _extract_numeric_field(text, field)
        if value is not None:
            payload[field] = value

    hold_duration = _extract_string_field(text, "hold_duration")
    if hold_duration:
        payload["hold_duration"] = hold_duration

    final_logic = _extract_string_field(text, "final_logic")
    if not final_logic:
        final_logic = _extract_reasoning_summary(text)
    if final_logic:
        payload["reasoning"] = {"final_logic": final_logic}

    key_factors = _extract_key_factors(text)
    if key_factors:
        payload["key_factors"] = key_factors

    if not payload:
        return None
    return payload


def _extract_numeric_field(text: str, field: str) -> Optional[float]:
    patterns = [
        rf'"{field}"\s*:\s*(-?\d+(?:\.\d+)?)',
        rf"'{field}'\s*:\s*(-?\d+(?:\.\d+)?)",
        rf"\b{field}\b\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
    return None


def _extract_string_field(text: str, field: str) -> str:
    patterns = [
        rf'"{field}"\s*:\s*"([^"]+)"',
        rf"'{field}'\s*:\s*'([^']+)'",
        rf"\b{field}\b\s*[:=]\s*([^\n,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip().strip('"').strip("'")
            if value:
                return value
    return ""


def _extract_reasoning_summary(text: str) -> str:
    without_fence = _strip_code_fence(text)
    compact = re.sub(r"\s+", " ", without_fence).strip()
    if not compact:
        return ""
    # Skip if the text looks like raw JSON — avoid leaking judge output into summary
    if compact.lstrip().startswith("{"):
        return ""
    if len(compact) > 280:
        return compact[:277] + "..."
    return compact


def _extract_key_factors(text: str) -> list[str]:
    factors = []
    list_field_patterns = [
        r'"key_factors"\s*:\s*\[(.*?)\]',
        r"'key_factors'\s*:\s*\[(.*?)\]",
        r"\bkey_factors\b\s*[:=]\s*\[(.*?)\]",
    ]
    for pattern in list_field_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        raw_items = re.findall(r"""['"]([^'"]+)['"]""", match.group(1))
        for item in raw_items:
            value = item.strip()
            if value and value not in factors:
                factors.append(value[:120])
            if len(factors) >= 5:
                return factors

    for match in re.finditer(r"^\s*[\-\*\u2022]\s+(.+)$", text, flags=re.MULTILINE):
        value = match.group(1).strip()
        if value and value not in factors:
            factors.append(value[:120])
        if len(factors) >= 5:
            break
    return factors
