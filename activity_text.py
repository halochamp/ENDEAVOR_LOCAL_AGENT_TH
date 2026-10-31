"""Bounded, redacted text for the Activity log.

The Activity panel shows each tool call's arguments and a preview of its result. Both
can contain credentials or binary payloads, so everything passes through here first.
Redaction runs BEFORE truncation: a private-key block cut in half would otherwise
still leak its head.
"""
from __future__ import annotations

import json
import re

_SECRET_FIELD = re.compile(r"(?i)(password|passwd|secret|token|api.?key|authorization|cookie|credential)")
_PAYLOAD_FIELDS = {"image_url", "base64", "pixels", "bytes", "headers"}
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?(?:-----END [^-\n]*PRIVATE KEY-----|$)", re.I | re.S
)
_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|PASSWORD|SECRET|API_KEY|ACCESS_KEY)[A-Z0-9_]*)(\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_CLI_FLAG = re.compile(r"(?i)(--(?:password|token|secret|api-key)(?:=|\s+))(?:\"[^\"]*\"|'[^']*'|[^\s;]+)")
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s,\"']+")
_URL_SECRET = re.compile(r"(?i)([?&](?:[^=&]*(?:token|key|signature|secret|password)[^=&]*)=)[^&\s]+")
_URL_CREDENTIALS = re.compile(r"https?://[^/\s:@]+:[^@\s]+@")
_DATA_URI = re.compile(r"data:[^\s,]+;base64,[A-Za-z0-9+/=]+")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_TRUNCATED = "… [truncated]"


def safe_text(value, limit: int = 1800) -> str:
    text = "" if value is None else str(value)
    text = _PRIVATE_KEY.sub("[REDACTED_PRIVATE_KEY]", text)
    text = _ASSIGNMENT.sub(r"\1\2[REDACTED]", text)
    text = _CLI_FLAG.sub(r"\1[REDACTED]", text)
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _URL_SECRET.sub(r"\1[REDACTED]", text)
    text = _URL_CREDENTIALS.sub("https://[REDACTED]@", text)
    text = _DATA_URI.sub("[image payload omitted]", text)
    text = _CONTROL.sub("", text)
    return text[:limit] + (_TRUNCATED if len(text) > limit else "")


def _safe_value(value, *, key: str = "", depth: int = 0, budget: list[int] | None = None):
    if budget is None:
        budget = [160]
    budget[0] -= 1
    if budget[0] < 0:
        return "[metadata budget exceeded]"
    if _SECRET_FIELD.search(str(key)):
        return "[REDACTED]"
    if str(key).lower() in _PAYLOAD_FIELDS:
        return "[payload omitted]"
    if depth > 4:
        return "[nested value omitted]"
    if isinstance(value, dict):
        result = {
            safe_text(k, 80): _safe_value(v, key=str(k), depth=depth + 1, budget=budget)
            for k, v in list(value.items())[:24]
        }
        if len(value) > 24:
            result["_truncated_fields"] = len(value) - 24
        return result
    if isinstance(value, (list, tuple)):
        items = [_safe_value(v, key=key, depth=depth + 1, budget=budget) for v in value[:12]]
        if len(value) > 12:
            items.append(f"… [truncated: {len(value) - 12} items]")
        return items
    if isinstance(value, (bytes, bytearray)):
        return "[bytes omitted]"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            nested = json.loads(value)
            if isinstance(nested, (dict, list)):
                return json.dumps(
                    _safe_value(nested, depth=depth + 1, budget=budget), ensure_ascii=False
                )
        except (ValueError, TypeError, RecursionError):
            pass
    return safe_text(value, 1200)


def arguments_preview(args) -> str:
    """Redacted, bounded JSON of a tool call's arguments ("" when there are none)."""
    if not isinstance(args, dict) or not args:
        return ""
    return safe_text(json.dumps(_safe_value(args), ensure_ascii=False), 2400)


def result_preview(output, *, error: bool = False) -> dict:
    """Status plus the first three non-empty lines of a tool result, redacted."""
    text = str(getattr(output, "content", output) or "")
    head = text.lstrip().lower()
    status = "failed" if error or head.startswith("[error]") else "returned"
    if head.startswith("[blocked]"):
        status = "blocked"
    lines = [line.strip() for line in safe_text(text, 1600).splitlines() if line.strip()]
    preview = "\n".join(lines[:3])[:700] or "[empty return]"
    truncated = len(text) > len(preview) and preview != text.strip()
    return {"status": status, "preview": preview, "truncated": truncated}


__all__ = ["arguments_preview", "result_preview", "safe_text"]
