# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Structured, text-only screen observations for the ``computer`` tool,
forked byte-identical from ENDEAVOR_LOCAL_AGENT_MAX's tools/_screen_state.py.

The main model never receives pixels.  This module turns OCR and optional
macOS Accessibility (AX) records into a compact, versioned element list that the
model can reference on its next tool call.  It contains no capture, input, LLM,
or LangChain code, so the geometry/fingerprinting rules stay deterministic and
cheap to unit-test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any


_VOLATILE_TEXT = re.compile(r"^(?:\d{1,2}:\d{2}|\d+\s*(?:s|m|h)\s+ago)$", re.IGNORECASE)
_ACTIONABLE_ROLES = {
    "AXButton", "AXCheckBox", "AXComboBox", "AXLink", "AXMenuButton",
    "AXMenuItem", "AXPopUpButton", "AXRadioButton", "AXSearchField",
    "AXSecureTextField", "AXSlider", "AXTextArea", "AXTextField",
}


@dataclass
class ScreenElement:
    element_id: str
    text: str
    role: str
    source: str
    point: tuple[float, float]
    bounds: tuple[float, float, float, float]
    region: str
    enabled: bool | None = None
    focused: bool | None = None
    actions: tuple[str, ...] = ()
    confidence: float = 1.0

    @property
    def actionable(self) -> bool:
        return self.role in _ACTIONABLE_ROLES or bool(self.actions)


@dataclass
class ScreenSnapshot:
    observation_id: str
    app: str
    window: str
    elements: list[ScreenElement]
    ocr_boxes: list[dict]
    image_path: str
    capture_size: tuple[int, int]
    screen_size: tuple[float, float]
    accessibility_status: str = "unavailable"
    description: str = ""
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        self.fingerprint = snapshot_fingerprint(self)

    def get(self, element_id: str) -> ScreenElement | None:
        needle = element_id.strip().casefold()
        return next((e for e in self.elements if e.element_id.casefold() == needle), None)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _region(point: tuple[float, float], screen_size: tuple[float, float]) -> str:
    width, height = screen_size
    if width <= 0 or height <= 0:
        return "center"
    x, y = point[0] / width, point[1] / height
    row = "top" if y < 1 / 3 else ("bottom" if y > 2 / 3 else "")
    col = "left" if x < 1 / 3 else ("right" if x > 2 / 3 else "")
    return f"{row}-{col}" if row and col else (row or col or "center")


def _inside_window(point: tuple[float, float], bounds: tuple[float, float, float, float] | None) -> bool:
    if not bounds:
        return True
    x, y, w, h = bounds
    # A little margin keeps sheet shadows/window chrome and menu-adjacent labels.
    margin = 8.0
    return x - margin <= point[0] <= x + w + margin and y - margin <= point[1] <= y + h + margin


def _same_element(a: ScreenElement, b: ScreenElement) -> bool:
    if not a.text or not b.text or a.text.casefold() != b.text.casefold():
        return False
    dx, dy = a.point[0] - b.point[0], a.point[1] - b.point[1]
    return dx * dx + dy * dy <= 30.0 * 30.0


def build_snapshot(
    observation_id: str,
    *,
    app: str,
    ocr_boxes: list[dict],
    ax_state: dict | None,
    image_path: str,
    capture_size: tuple[int, int],
    screen_size: tuple[float, float],
    description: str = "",
) -> ScreenSnapshot:
    """Fuse AX records with OCR boxes and assign stable IDs for this snapshot.

    AX is preferred because it supplies roles, state and icon labels. OCR fills
    visible-text gaps. When AX reports a focused-window bound, whole-desktop OCR
    outside that window is dropped to prevent host IDE/chat text from polluting
    the agent's view.
    """
    ax_state = ax_state if isinstance(ax_state, dict) else {}
    window = _clean_text(ax_state.get("window"))
    status = _clean_text(ax_state.get("status")) or "unavailable"
    raw_window_bounds = ax_state.get("window_bounds")
    window_bounds: tuple[float, float, float, float] | None = None
    if isinstance(raw_window_bounds, (list, tuple)) and len(raw_window_bounds) == 4:
        try:
            window_bounds = tuple(float(v) for v in raw_window_bounds)  # type: ignore[assignment]
        except (TypeError, ValueError):
            window_bounds = None

    elements: list[ScreenElement] = []
    for raw in ax_state.get("elements", []) if isinstance(ax_state.get("elements"), list) else []:
        try:
            bounds = tuple(float(v) for v in raw.get("bounds", []))
            if len(bounds) != 4 or bounds[2] <= 0 or bounds[3] <= 0:
                continue
            point = (bounds[0] + bounds[2] / 2, bounds[1] + bounds[3] / 2)
            text = _clean_text(raw.get("name") or raw.get("value"))
            role = _clean_text(raw.get("role")) or "AXUnknown"
            actions = tuple(_clean_text(v) for v in raw.get("actions", []) if _clean_text(v))
            # Keep actionable unnamed controls (their role/location is still useful)
            # and named semantic/static elements. Drop invisible structural groups.
            if not text and role not in _ACTIONABLE_ROLES and not actions:
                continue
            elements.append(ScreenElement(
                element_id="",
                text=text or f"<{role}>",
                role=role,
                source="ax",
                point=point,
                bounds=bounds,  # type: ignore[arg-type]
                region=_region(point, screen_size),
                enabled=raw.get("enabled") if isinstance(raw.get("enabled"), bool) else None,
                focused=raw.get("focused") if isinstance(raw.get("focused"), bool) else None,
                actions=actions,
                confidence=1.0,
            ))
        except (TypeError, ValueError, KeyError):
            continue

    screen_width, screen_height = screen_size
    for box in ocr_boxes:
        try:
            cx = (float(box["x"]) + float(box["w"]) / 2) * screen_width
            cy = (1.0 - (float(box["y"]) + float(box["h"]) / 2)) * screen_height
            point = (cx, cy)
            if not _inside_window(point, window_bounds):
                continue
            text = _clean_text(box.get("text"))
            if not text:
                continue
            w, h = float(box["w"]) * screen_width, float(box["h"]) * screen_height
            candidate = ScreenElement(
                element_id="",
                text=text,
                role="OCRText",
                source="ocr",
                point=point,
                bounds=(cx - w / 2, cy - h / 2, w, h),
                region=_region(point, screen_size),
                confidence=0.98,
            )
            if not any(_same_element(candidate, existing) for existing in elements):
                elements.append(candidate)
        except (TypeError, ValueError, KeyError):
            continue

    # Actionable/focused AX nodes first, then other AX, then OCR in visual order.
    elements.sort(key=lambda e: (
        0 if e.focused else 1,
        0 if e.actionable else 1,
        0 if e.source == "ax" else 1,
        round(e.point[1], 2), round(e.point[0], 2), e.text.casefold(),
    ))
    for index, element in enumerate(elements, 1):
        element.element_id = f"e{index}"

    return ScreenSnapshot(
        observation_id=observation_id,
        app=app,
        window=window,
        elements=elements,
        ocr_boxes=list(ocr_boxes),
        image_path=image_path,
        capture_size=capture_size,
        screen_size=screen_size,
        accessibility_status=status,
        description=description,
    )


def snapshot_fingerprint(snapshot: ScreenSnapshot) -> str:
    semantic = []
    for e in snapshot.elements:
        text = _clean_text(e.text).casefold()
        if _VOLATILE_TEXT.fullmatch(text):
            continue
        semantic.append((
            e.source, e.role, text, e.region, e.enabled, e.focused,
            tuple(action.casefold() for action in e.actions),
        ))
    payload = json.dumps(
        {"app": snapshot.app.casefold(), "window": snapshot.window.casefold(), "elements": semantic},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def format_snapshot(snapshot: ScreenSnapshot, max_chars: int = 900) -> str:
    header = f'[OBS {snapshot.observation_id} app={snapshot.app or "unknown"!r}'
    if snapshot.window:
        header += f' window={snapshot.window!r}'
    header += f' ax={snapshot.accessibility_status}]'
    lines = [header]
    for e in snapshot.elements:
        state = []
        if e.enabled is not None:
            state.append(f"enabled={str(e.enabled).lower()}")
        if e.focused:
            state.append("focused=true")
        if e.actions:
            state.append("actions=" + ",".join(action.removeprefix("AX") for action in e.actions[:3]))
        suffix = (" " + " ".join(state)) if state else ""
        line = (
            f'{e.element_id} role={e.role.removeprefix("AX")} text={e.text!r} '
            f'region={e.region} source={e.source}{suffix}'
        )
        if sum(len(v) + 1 for v in lines) + len(line) > max_chars:
            remaining = len(snapshot.elements) - (len(lines) - 1)
            lines.append(f"… {remaining} more elements omitted; use inspect on a visible element or scroll")
            break
        lines.append(line)
    if len(lines) == 1:
        lines.append("(no semantic/OCR elements found)")
    return "\n".join(lines)
