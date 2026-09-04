# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Process-local capability detection for the configured image model.

The cache is deliberately ephemeral and keyed by the effective endpoint/model
pair.  It is shared by the read-image fallback and the computer-use safety gate,
but it never persists a capability decision to the workspace or to a file.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Literal


UNKNOWN = "unknown"
VISION = "vision"
TEXT_ONLY = "text_only"
Capability = Literal["unknown", "vision", "text_only"]

_CACHE: dict[tuple[str, str], Capability] = {}
_CACHE_LOCK = threading.RLock()
_PROBE_LOCK = threading.Lock()
_PROBE_ATTEMPTED: set[tuple[str, str]] = set()

# The marker is deliberately absent from the probe prompt. A response containing
# it is semantic evidence that the model actually read the pixels, not merely
# that the OpenAI-compatible endpoint accepted an image-shaped payload.
_PROBE_MARKER = "VISION 742"
_PROBE_IMAGE_URL = ""

_UNSUPPORTED_IMAGE_RE = re.compile(
    r"(?:"
    r"(?:image(?:_url)?|multimodal|vision|pixel)[^\n]{0,120}"
    r"(?:unsupported|not[\s_-]+supported|not[\s_-]+accepted|does[\s_-]+not[\s_-]+support|"
    r"doesn't[\s_-]+support|isn't[\s_-]+supported|only[\s_-]+supports[\s_-]+text)"
    r"|(?:unsupported|not[\s_-]+supported|not[\s_-]+accepted|does[\s_-]+not[\s_-]+support|"
    r"doesn't[\s_-]+support|isn't[\s_-]+supported|only[\s_-]+supports[\s_-]+text)[^\n]{0,120}"
    r"(?:image(?:_url)?|multimodal|vision|pixel)"
    r"|\btext[- ]only\b[^\n]{0,120}(?:image|multimodal|vision)"
    r"|(?:image|multimodal|vision)[^\n]{0,120}\btext[- ]only\b"
    r")",
    re.IGNORECASE,
)

_NO_VISION_RESPONSE_RE = re.compile(
    r"(?:"
    r"\bno[_ -]?image[_ -]?access\b"
    r"|"
    r"\b(?:cannot|can't|can\s+not|unable\s+to|not\s+able\s+to)\b[^\n]{0,100}"
    r"\b(?:view|see|look\s+at|analy[sz]e|interpret|access|understand)\b[^\n]{0,100}"
    r"\b(?:image|images|picture|pictures|photo|photos|visual|pixel|pixels)\b"
    r"|\b(?:do\s+not|don't|does\s+not|doesn't)\s+have\b[^\n]{0,80}"
    r"\b(?:vision|visual\s+capabilit(?:y|ies)|image\s+access)\b"
    r")",
    re.IGNORECASE,
)


def capability_key(endpoint: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Return the normalized process-local key for an endpoint/model pair."""
    if endpoint is None or model is None:
        from config import MLX_BASE_URL, MODEL

        endpoint = MLX_BASE_URL if endpoint is None else endpoint
        model = MODEL if model is None else model
    return (str(endpoint).strip().rstrip("/"), str(model).strip())


def get_capability(endpoint: str | None = None, model: str | None = None) -> Capability:
    key = capability_key(endpoint, model)
    with _CACHE_LOCK:
        return _CACHE.get(key, UNKNOWN)


def mark_vision(endpoint: str | None = None, model: str | None = None) -> Capability:
    key = capability_key(endpoint, model)
    with _CACHE_LOCK:
        _CACHE[key] = VISION
    return VISION


def mark_text_only(endpoint: str | None = None, model: str | None = None) -> Capability:
    key = capability_key(endpoint, model)
    with _CACHE_LOCK:
        _CACHE[key] = TEXT_ONLY
    return TEXT_ONLY


def reset_for_tests() -> None:
    """Clear only the in-memory cache; intended for deterministic tests."""
    with _CACHE_LOCK:
        _CACHE.clear()
        _PROBE_ATTEMPTED.clear()


def _build_probe_image_url() -> str:
    """Build a small high-contrast marker image without touching user data."""
    global _PROBE_IMAGE_URL
    if _PROBE_IMAGE_URL:
        return _PROBE_IMAGE_URL

    import base64
    from io import BytesIO
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    font = None
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            font = ImageFont.truetype(candidate, 48)
            break

    if font is not None:
        image = Image.new("RGB", (420, 128), "white")
        draw = ImageDraw.Draw(image)
        bounds = draw.textbbox((0, 0), _PROBE_MARKER, font=font, stroke_width=1)
        x = (image.width - (bounds[2] - bounds[0])) // 2
        y = (image.height - (bounds[3] - bounds[1])) // 2 - bounds[1]
        draw.rectangle((4, 4, image.width - 5, image.height - 5), outline="black", width=4)
        draw.text((x, y), _PROBE_MARKER, fill="black", font=font, stroke_width=1)
    else:
        # Pillow's fallback font is rendered large so minimal test hosts still
        # get a useful marker without a system TrueType font.
        small = Image.new("RGB", (96, 18), "white")
        ImageDraw.Draw(small).text((2, 2), _PROBE_MARKER, fill="black")
        image = small.resize((384, 108), Image.Resampling.NEAREST)

    encoded = BytesIO()
    image.save(encoded, format="PNG", optimize=True)
    _PROBE_IMAGE_URL = "data:image/png;base64," + base64.b64encode(
        encoded.getvalue()
    ).decode("ascii")
    return _PROBE_IMAGE_URL


def _response_text(response: object) -> str:
    """Extract textual content from LangChain/OpenAI response shapes."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content or "")


def _semantic_probe_result(response: object) -> Capability:
    """Classify only a marker proof or an explicit no-vision refusal."""
    text = _response_text(response)
    compact = re.sub(r"[^a-z0-9]+", "", text.casefold())
    expected = re.sub(r"[^a-z0-9]+", "", _PROBE_MARKER.casefold())
    if expected in compact:
        return VISION
    if _NO_VISION_RESPONSE_RE.search(text):
        return TEXT_ONLY
    return UNKNOWN


def _exception_text(error: BaseException) -> str:
    parts = [str(error)]
    for attr in ("message", "body"):
        value = getattr(error, attr, None)
        if value:
            parts.append(json.dumps(value, ensure_ascii=False, default=str)
                         if isinstance(value, (dict, list)) else str(value))
    response = getattr(error, "response", None)
    if response is not None:
        for attr in ("text", "reason"):
            value = getattr(response, attr, None)
            if value:
                parts.append(str(value))
        value = getattr(response, "_content", None)
        if value:
            parts.append(value.decode(errors="replace") if isinstance(value, bytes) else str(value))
    return " ".join(parts).casefold()


def _status_code(error: BaseException) -> int | None:
    candidates = [getattr(error, "status_code", None)]
    response = getattr(error, "response", None)
    if response is not None:
        candidates.extend((getattr(response, "status_code", None), getattr(response, "status", None)))
    for value in candidates:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def is_unsupported_image_error(error: BaseException) -> bool:
    """Conservatively recognize a backend's explicit text-only/image rejection.

    Authentication, transport, timeout, server, context-window, malformed-tool,
    and generic bad-request errors intentionally return False.  A status code is
    helpful but not required when an exception carries an unambiguous backend
    message (as several OpenAI-compatible servers do).
    """
    text = _exception_text(error)
    error_type = type(error).__name__.casefold()
    status = _status_code(error)

    if any(token in error_type for token in ("timeout", "connection", "connecterror", "readerror")):
        return False
    if re.search(r"\b(?:timed?\s*out|timeout|connection\s+(?:refused|reset|error)|network\s+error)\b", text):
        return False
    if re.search(r"context[^\n]{0,50}(?:length|window|limit)|(?:maximum|max)[^\n]{0,20}context", text):
        return False
    if "tool" in text and re.search(r"(?:tool|function)[^\n]{0,80}(?:invalid|malformed|schema|argument|json|call)", text):
        return False
    if status is not None and status not in {400, 415, 422}:
        return False
    if status is None:
        textual_status = re.search(r"\b([1-5]\d{2})\b", text)
        if textual_status and int(textual_status.group(1)) not in {400, 415, 422}:
            return False
    return bool(_UNSUPPORTED_IMAGE_RE.search(text))


def probe_vision_capability(
    endpoint: str | None = None,
    model: str | None = None,
    *,
    timeout: float = 5.0,
) -> Capability:
    """Send one tiny, non-mutating image request when capability is unknown.

    The probe disables the read-image fallback so an unsupported response cannot
    be mistaken for a successful text-only retry.  Inconclusive failures remain
    UNKNOWN and are therefore fail-closed by computer_use.py.
    """
    key = capability_key(endpoint, model)
    with _CACHE_LOCK:
        cached = _CACHE.get(key, UNKNOWN)
        attempted = key in _PROBE_ATTEMPTED
    if cached != UNKNOWN or attempted:
        return cached

    with _PROBE_LOCK:
        with _CACHE_LOCK:
            cached = _CACHE.get(key, UNKNOWN)
            attempted = key in _PROBE_ATTEMPTED
            if cached != UNKNOWN or attempted:
                return cached
            _PROBE_ATTEMPTED.add(key)
        try:
            from langchain_core.messages import HumanMessage
            from llm import build_llm

            overrides = {
                "vision_fallback": False,
                "max_tokens": 8,
                "temperature": 0,
                "streaming": False,
                "timeout": timeout,
                "max_retries": 0,
                "extra_body": {"enable_thinking": False},
            }
            if endpoint is not None:
                overrides["base_url"] = endpoint
            if model is not None:
                overrides["model"] = model
            client = build_llm(**overrides)
            response = client.invoke([HumanMessage(content=[
                {
                    "type": "text",
                    "text": "Read the visible alphanumeric marker from this image. "
                    "Reply with only the marker, exactly as seen. "
                    "If you cannot inspect image pixels, reply exactly NO_IMAGE_ACCESS.",
                },
                {"type": "image_url", "image_url": {"url": _build_probe_image_url()}},
            ])])
        except Exception as error:
            if is_unsupported_image_error(error):
                return mark_text_only(*key)
            return UNKNOWN
        semantic = _semantic_probe_result(response)
        if semantic == VISION:
            return mark_vision(*key)
        if semantic == TEXT_ONLY:
            return mark_text_only(*key)
        return UNKNOWN
