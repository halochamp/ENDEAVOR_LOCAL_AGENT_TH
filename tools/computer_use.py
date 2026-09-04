# ENDEAVOR_LOCAL_AGENT_TH — public project source
# Licensed under the repository's public project terms. Keep local-only runtime behavior.

"""Direct-vision computer interaction tool.

This is the public project's fork of the current direct-vision interaction
lifecycle. Computer observation/action state is intentionally independent from
read_image state; the model boundary composes their published image data.
"""

from __future__ import annotations

import base64
import difflib
import re
import tempfile
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

try:
    from langchain_core.tools import tool
except ImportError:  # lets deterministic tests run without the agent environment
    def tool(fn):
        fn.func = fn
        return fn

from . import _mac_input as _backend
from ._ocr import read_layout as _ocr_layout
from .read_file import _sample_coverage
from ._screen_state import ScreenElement, ScreenSnapshot, build_snapshot, format_snapshot
from tools._progress import progress as _progress, phase as _phase

_ACTION_ATTEMPTS = [0]
_ACTION_MAX = 15
# Per-turn overrides — set by graph.py for turns that run with no human watching
# (e.g. an AWAKE-fired background turn). Both reset to defaults every real turn
# by reset_computer_guards() so a tightened scope never leaks into a later,
# normal (human-supervised) turn.
_ACTION_MAX_OVERRIDE = [None]
_DESTRUCTIVE_GUARD = [False]
# Two tiers, not one — ported from reference 2026-07-21 (live-reproduced there as
# audit F1): a single broad list wrongly catches "Format" (a menu-bar text-
# formatting item, not disk-erase) and "Trash" (Finder sidebar item — opening
# it to LOOK is not emptying it) the moment the destructive guard becomes the
# DEFAULT for ordinary human-supervised turns. Those words are genuinely
# destructive only in narrow contexts (Disk Utility's "Format..." button,
# "Empty Trash") that plain substring matching can't distinguish from the
# common benign case. _DESTRUCTIVE_MARKERS_SUPERVISED (default-guard turns,
# where over-blocking directly breaks ordinary work) omits them;
# _DESTRUCTIVE_MARKERS_UNSUPERVISED (awake-fired turns, where nobody is
# watching and maximum caution is worth the false-positive rate) keeps the
# full original list. Selected in the check itself via whether action_max
# was set (awake-fired always sets it; the normal-turn default guard never
# does) — no new per-turn flag needed, that distinction already exists.
_DESTRUCTIVE_MARKERS_SUPERVISED = (
    "delete", "ลบ", "remove", "เอาออก",
    "uninstall", "ถอนการติดตั้ง", "ลบล้าง", "ล้างข้อมูล", "empty trash", "move to trash",
)
_DESTRUCTIVE_MARKERS_UNSUPERVISED = _DESTRUCTIVE_MARKERS_SUPERVISED + (
    "trash", "ถังขยะ", "erase", "format", "wipe",
)
_LAST_SIGNATURE = [""]
# OCR text as of the end of the last action — stands in for "the screen right
# now, before this call" (nothing else touches the display between agent
# turns), so a repeat check needs no extra pre-action capture on the common path.
_LAST_SCREEN_TEXT = [""]
# Frontmost app as of the end of the last action. Live-reproduced failure mode
# (2026-07-17): something outside the agent's control (another app, the host
# IDE re-activating itself) can steal frontmost focus between two computer()
# calls. click/double_click/right_click self-correct for this by construction
# (they re-locate the target via a fresh OCR pass every time — worst case they
# error "text not found" or, more subtly, click a same-labeled element in the
# WRONG app), but type/key have no such check at all: they blindly send
# keystrokes to whatever currently has focus, so a silent focus change means
# keystrokes land in an unintended app with no error. Checked for every
# context-dependent action, not just type/key, since a click on a wrong-app
# element with a matching label is exactly as silent a failure.
_LAST_FRONTMOST_APP = [""]
# Target text used to locate the most recently clicked element (audit F3) —
# e.g. target="Password" when the model clicks a password field via its
# visible placeholder/label. A whole-screen scan for password markers would
# refuse ordinary typing on any login-adjacent screen (a visible "Password"
# label next to the username field you're legitimately filling) since OCR
# can't tell which field is actually focused; the click target the model
# itself chose is a much more precise proxy for "the field about to receive
# text" than anything scanned from the screen at large.
_LAST_CLICK_TARGET = [""]
_CONTEXT_DEPENDENT_ACTIONS = {"click", "double_click", "triple_click", "right_click", "type", "key", "scroll", "drag", "hover"}
# A visible clock changes the OCR text every minute even though nothing about
# the screen actually changed — without filtering it out, the repeat-guard's
# raw-text comparison below would only ever catch a repeat within the same
# wall-clock minute (audit F7). Same pattern as awake_engine.py's
# _VOLATILE_LINE/_screen_hash for the same reason; duplicated rather than
# imported since tools/ and the host-level awake_engine.py are different tiers.
_VOLATILE_LINE = re.compile(r"\b\d{1,2}:\d{2}\b")
_ACTION_LOG = Path(__file__).resolve().parents[1] / "logs" / "computer_actions.jsonl"
# Screen-facing text is OCR-dense/low-signal-per-char (menu bars, file trees) —
# capped well under CLAUDE.md's Tool Output Contract ceiling since this tool
# fires far more often per task than a single document/web sensor call.
_SUMMARY_MAX_CHARS = 500
_PASSWORD_MARKERS = ("password", "รหัสผ่าน")
_OBSERVATION_SEQ = [0]
_LATEST_OBSERVATION: list[ScreenSnapshot | None] = [None]
# Computer owns its own direct-vision context and lifecycle. It does not import or
# mutate another image tool's queues, guards, or snapshots; graph.py is the only
# model-boundary composition layer.
_COMPUTER_PENDING_IMAGES: list[str] = []
_COMPUTER_ACTIVE_IMAGES: list[str] = []
_COMPUTER_VLM_MAX_SIDE = 1024
_ACTION_LOCK = threading.Lock()


def _clear_computer_vision_state(*, remove_snapshot: bool) -> None:
    _COMPUTER_PENDING_IMAGES.clear()
    _COMPUTER_ACTIVE_IMAGES.clear()
    if remove_snapshot:
        previous = _LATEST_OBSERVATION[0]
        if previous:
            Path(previous.image_path).unlink(missing_ok=True)
        _LATEST_OBSERVATION[0] = None


def active_computer_turn_images() -> list[str]:
    """Return only computer's newest direct-vision frame for the current outer turn."""
    if _COMPUTER_PENDING_IMAGES:
        _COMPUTER_ACTIVE_IMAGES[:] = _COMPUTER_PENDING_IMAGES[-1:]
        _COMPUTER_PENDING_IMAGES.clear()
    return list(_COMPUTER_ACTIVE_IMAGES)


def end_computer_turn() -> None:
    """Release computer-only transient pixels/screenshot at the outer-turn boundary."""
    _clear_computer_vision_state(remove_snapshot=True)


def reset_computer_guards() -> None:
    _ACTION_ATTEMPTS[0] = 0
    _LAST_SIGNATURE[0] = ""
    _LAST_SCREEN_TEXT[0] = ""
    _LAST_FRONTMOST_APP[0] = ""
    _LAST_CLICK_TARGET[0] = ""
    _ACTION_MAX_OVERRIDE[0] = None
    _DESTRUCTIVE_GUARD[0] = False
    _clear_computer_vision_state(remove_snapshot=True)


def set_computer_turn_scope(action_max: int | None = None, block_destructive: bool = False) -> None:
    """Tighten this turn's computer-use ceiling below the normal per-turn defaults —
    called by graph.py for turns that run with no human watching the screen (e.g. an
    AWAKE-fired background turn). Takes effect immediately and lasts until the next
    reset_computer_guards() call (every real turn), so a tightened scope never leaks
    into a later, normal (human-supervised) turn."""
    _ACTION_MAX_OVERRIDE[0] = action_max
    _DESTRUCTIVE_GUARD[0] = block_destructive


def _next_observation_id() -> str:
    _OBSERVATION_SEQ[0] += 1
    return f"obs_{_OBSERVATION_SEQ[0]}"


def _overlay_cursor(snapshot: ScreenSnapshot) -> str:
    """Create a cursor-annotated presentation copy without mutating retained state pixels."""
    overlay_path = ""
    try:
        import cv2
        point_x, point_y = _backend.pointer_position()
        capture_w, capture_h = snapshot.capture_size
        screen_w, screen_h = snapshot.screen_size
        if screen_w <= 0 or screen_h <= 0:
            return ""
        x = max(0, min(capture_w - 1, round(point_x * capture_w / screen_w)))
        y = max(0, min(capture_h - 1, round(point_y * capture_h / screen_h)))
        frame = cv2.imread(snapshot.image_path)
        if frame is None:
            return ""
        center = (x, y)
        radius = max(8, round(min(capture_w, capture_h) * 0.012))
        cv2.circle(frame, center, radius + 2, (0, 0, 0), 4)
        cv2.circle(frame, center, radius, (255, 255, 255), 2)
        cv2.line(frame, (x - radius - 5, y), (x + radius + 5, y), (255, 255, 255), 2)
        cv2.line(frame, (x, y - radius - 5), (x, y + radius + 5), (255, 255, 255), 2)
        label = f"{x},{y}"
        label_x = max(4, min(capture_w - 90, x + radius + 8))
        label_y = max(18, min(capture_h - 6, y - radius - 6))
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            overlay_path = handle.name
        if not cv2.imwrite(overlay_path, frame):
            Path(overlay_path).unlink(missing_ok=True)
            return ""
        snapshot.cursor_capture = (x, y)
        return overlay_path
    except Exception:
        if overlay_path:
            Path(overlay_path).unlink(missing_ok=True)
        snapshot.cursor_capture = None
        return ""


def _publish(snapshot: ScreenSnapshot) -> ScreenSnapshot:
    previous = _LATEST_OBSERVATION[0]
    if previous is not None and previous is not snapshot:
        Path(previous.image_path).unlink(missing_ok=True)
    _LATEST_OBSERVATION[0] = snapshot
    overlay_path = _overlay_cursor(snapshot)
    if overlay_path:
        try:
            _queue_latest_capture(overlay_path)
        finally:
            Path(overlay_path).unlink(missing_ok=True)
    else:
        _queue_latest_capture(snapshot.image_path)
    return snapshot


def _normalize_combo(combo: str) -> str:
    return "+".join(sorted(part.strip().lower() for part in combo.split("+") if part.strip()))


# Dangerous hotkeys refused outright regardless of context (quit/logout).
# Stored pre-normalized so lookups are a plain set-membership check.
_FORBIDDEN_KEY_COMBOS = {_normalize_combo(c) for c in ("cmd+q", "cmd+shift+q")}

# `key`'s text is a literal hotkey combo, not a button label — a bare Backspace/
# Delete keystroke while editing text is an ordinary, extremely common edit
# action (fixing a typo), not the same risk as clicking a "Delete"/"Empty Trash"
# BUTTON. Loose substring matching against the marker lists would catch
# "delete" in a plain Backspace/Delete press and block routine text editing —
# so the destructive guard checks `key` against this separate, exact,
# normalized-combo set instead (macOS's actual "permanently remove"-flavored
# hotkeys), never by substring. click/double_click/right_click/drag still use
# the marker lists (their target/text IS a visible label, where a loose
# match is the right call).
_DESTRUCTIVE_KEY_COMBOS = {
    _normalize_combo(c) for c in ("cmd+delete", "cmd+backspace", "shift+delete", "cmd+shift+delete")
}


def _cap_for_agent(text: str) -> str:
    return _sample_coverage(text, "screen", max_chars=_SUMMARY_MAX_CHARS)


def _stable_signature(text: str) -> str:
    """`text` with volatile lines (e.g. a menu-bar clock) dropped, for the
    repeat-guard's screen-unchanged comparison — two captures a minute apart
    of a genuinely static screen must compare equal even though the clock
    line differs (audit F7)."""
    return "\n".join(line for line in text.splitlines()
                     if line.strip() and not _VOLATILE_LINE.search(line))


def _error(message: str) -> str:
    return f"[error] {message}"


# Uses the same directional-region vocabulary as the rest of the UI tooling.
# Boxes are normalized [0,1], origin BOTTOM-LEFT (Vision convention) — "top"
# means high y, not low.
def _in_region(box: dict, region: str) -> bool:
    cx = float(box["x"]) + float(box["w"]) / 2
    cy = float(box["y"]) + float(box["h"]) / 2
    if region == "top":
        return cy > 2 / 3
    if region == "bottom":
        return cy < 1 / 3
    if region == "left":
        return cx < 1 / 3
    if region == "right":
        return cx > 2 / 3
    if region == "center":
        return 1 / 3 <= cx <= 2 / 3 and 1 / 3 <= cy <= 2 / 3
    if region == "top-left":
        return cy > 2 / 3 and cx < 1 / 3
    if region == "top-right":
        return cy > 2 / 3 and cx > 2 / 3
    if region == "bottom-left":
        return cy < 1 / 3 and cx < 1 / 3
    if region == "bottom-right":
        return cy < 1 / 3 and cx > 2 / 3
    return True  # unrecognized region string — no filtering, caller sees the plain ambiguity error


def _fuzzy_suggestions(boxes: list[dict], needle: str, limit: int = 3) -> str:
    """Ranked near-miss candidates for a not-found target — suggestion text
    only, never used to resolve a click point. OCR/label text can differ from
    the model's guess by a typo or minor normalization (e.g. smart quotes,
    truncated ellipsis); difflib.get_close_matches surfaces the likely intent
    without ever silently substituting it, so a typo'd target can never
    resolve past the destructive-action guard below (that guard only sees
    the model's own `target` string, not a fuzzy match)."""
    if not needle:
        return ""
    # One canonical (first-seen) spelling per casefold value — two on-screen
    # labels differing only by case (e.g. a button "Remove" and, elsewhere,
    # OCR/a tooltip rendering "REMOVE") are the same suggestion to a model
    # choosing what to click next, and must not consume two of `limit` slots.
    by_casefold: dict[str, str] = {}
    for b in boxes:
        text = str(b.get("text", "")).strip()
        if text:
            by_casefold.setdefault(text.casefold(), text)
    close = difflib.get_close_matches(needle, list(by_casefold), n=limit, cutoff=0.6)
    return ", ".join(f"'{by_casefold[c]}'" for c in close)


def _find_target(boxes: list[dict], target: str, near: str = "") -> tuple[dict | None, str | None]:
    needle = target.strip().casefold()
    matches = [box for box in boxes if needle and needle in str(box.get("text", "")).casefold()]
    if not matches:
        suggestion = _fuzzy_suggestions(boxes, needle)
        hint = f" — did you mean: {suggestion}?" if suggestion else ""
        return None, _error(f"text not found: {target}{hint}")
    if len(matches) > 1:
        # Prefer a higher-precision match before falling back to raw substring
        # disambiguation — e.g. a "Save" button vs. a sentence that merely
        # contains "save" as a substring should resolve to the button alone,
        # with no need for the caller to supply near= at all.
        exact = [box for box in matches if str(box.get("text", "")).strip().casefold() == needle]
        if len(exact) == 1:
            return exact[0], None
        candidates = exact or [
            box for box in matches
            if re.search(rf"\b{re.escape(needle)}\b", str(box.get("text", "")).casefold())
        ]
        if len(candidates) == 1:
            return candidates[0], None
        if candidates:
            matches = candidates
    if len(matches) > 1 and near:
        # Real live failure this disambiguates (2026-07-17): two elements with
        # the literal same label/text — no target string, however precise,
        # can distinguish them, only screen position can (e.g. two identically
        # labeled buttons in different steps of a wizard-style dialog).
        narrowed = [box for box in matches if _in_region(box, near)]
        if len(narrowed) == 1:
            return narrowed[0], None
        if narrowed:
            matches = narrowed
    if len(matches) > 1:
        labels = ", ".join(str(box.get("text", "")) for box in matches[:3])
        hint = "" if near else ' — add near="top/bottom/left/right/center/top-left/top-right/bottom-left/bottom-right"'
        return None, _error(f"'{target}' matches {len(matches)} locations: {labels}{hint}")
    return matches[0], None


def _encode_computer_data_url(path: str) -> str:
    """Encode a computer-owned screenshot for direct vision."""
    import cv2
    frame = cv2.imread(path)
    if frame is None:
        raise ValueError(f"cannot read computer screenshot: {path}")
    h, w = frame.shape[:2]
    scale = _COMPUTER_VLM_MAX_SIDE / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    png_ok, png = cv2.imencode(".png", frame)
    if not png_ok:
        raise ValueError("computer screenshot PNG encode failed")
    jpg_ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    chosen, mime = (
        (jpg, "image/jpeg")
        if jpg_ok and len(jpg) * 4 <= len(png) * 3
        else (png, "image/png")
    )
    return f"data:{mime};base64,{base64.b64encode(chosen.tobytes()).decode()}"


def _queue_latest_capture(path: str) -> None:
    """Publish computer's newest frame into its own one-image direct-vision context."""
    try:
        encoded = _encode_computer_data_url(path)
        _COMPUTER_PENDING_IMAGES.clear()
        _COMPUTER_ACTIVE_IMAGES.clear()
        _COMPUTER_PENDING_IMAGES.append(encoded)
    except Exception:
        # OCR/action correctness must not fail merely because optional VLM encoding fails.
        pass


def _log_action(action: str, target: str = "", point: tuple[float, float] | None = None) -> None:
    """Append an auditable action record; never persist typed text."""
    try:
        _ACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(timezone.utc).isoformat(), "action": action, "target": target[:160]}
        if point is not None:
            record["point"] = [round(point[0], 2), round(point[1], 2)]
        with _ACTION_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _target_point(target: str, near: str) -> tuple[tuple[float, float] | None, str | None]:
    """Locate `target` via a fresh OCR pass and return its center in display points.
    Shared by click/scroll(target=)/drag so every text-targeted action resolves
    coordinates the same way — OCR box -> capture pixels -> display points."""
    boxes, _, capture_width, capture_height, screen_width, screen_height = _observe()
    box, problem = _find_target(boxes, target, near)
    if problem:
        return None, problem
    # Vision y is bottom-left normalized; CGEvent/display coordinates are top-left.
    capture_x = (float(box["x"]) + float(box["w"]) / 2) * capture_width
    capture_y = (1 - (float(box["y"]) + float(box["h"]) / 2)) * capture_height
    point = _backend.capture_px_to_points(capture_x, capture_y, capture_width, capture_height,
                                           screen_width, screen_height)
    return point, None


def _capture_snapshot(notify: bool = True) -> ScreenSnapshot:
    """Capture one retained, full-resolution state and queue it for the VLM.

    `notify=False` skips the per-call progress() emits — used by
    _post_action_snapshot's tight poll loop, where repeating "screenshot…/
    OCR…" every ~120ms is pure noise and, on Telegram, real edit_text() calls
    that can trip flood control."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = handle.name
    try:
        if notify:
            _progress("กำลัง screenshot…")
        _backend.capture(path)
        if notify:
            _progress("Apple Vision OCR…")
        boxes = _ocr_layout(path)
        capture_width, capture_height = _backend.image_dimensions(path)
        screen_width, screen_height = _backend.primary_display_size_points()
        try:
            from ._accessibility import read_frontmost
            ax_state = read_frontmost()
        except Exception:
            ax_state = {"status": "unavailable", "elements": []}
        return build_snapshot(
            _next_observation_id(), app=_backend.frontmost_app_name(), ocr_boxes=boxes,
            ax_state=ax_state, image_path=path, capture_size=(capture_width, capture_height),
            screen_size=(screen_width, screen_height),
        )
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise


def _observe() -> tuple[list[dict], str, int, int, float, float]:
    """Compatibility helper for OCR-backed target lookup; does not retain files."""
    snapshot = _capture_snapshot()
    try:
        text = "\n".join(str(box.get("text", "")) for box in snapshot.ocr_boxes if box.get("text"))
        _queue_latest_capture(snapshot.image_path)
        return (
            snapshot.ocr_boxes, text or "[no OCR text found]",
            snapshot.capture_size[0], snapshot.capture_size[1],
            snapshot.screen_size[0], snapshot.screen_size[1],
        )
    finally:
        Path(snapshot.image_path).unlink(missing_ok=True)


_POST_ACTION_TIMEOUT = 1.2
_POST_ACTION_POLL = 0.12
_VISUAL_CHANGE_MIN_FRACTION = 0.003
# Extra bounded wait applied ONLY when the caller passed expect= or an
# open_app/open_url app= — both name an async condition (a page finishing
# load, an app becoming frontmost) that routinely takes longer than the
# ordinary 1.2s settle window every other action pays. Every action still
# pays the base 1.2s; this only postpones giving up when there's a specific,
# still-unsatisfied thing to wait for, never used to cut the loop short early
# (see the break condition below — expect/app satisfied is an ADDITIONAL
# requirement to stop, not an alternative to changed+stable, so an early
# expect match still can't return a mid-animation frame).
_POST_ACTION_EXTENDED_TIMEOUT = 4.0


def _screen_change_signals(before: ScreenSnapshot, current: ScreenSnapshot) -> tuple[bool, bool]:
    """Return independent semantic and pixel-change signals for one screen transition."""
    semantic_changed = current.fingerprint != before.fingerprint
    visual_changed = current.visual_delta(before) >= _VISUAL_CHANGE_MIN_FRACTION
    return semantic_changed, visual_changed


def _app_matches(requested_app: str, actual_app: str) -> bool:
    # `open -a` can return 0 without the target ever becoming frontmost, and
    # a localized/short app name can legitimately differ from the requested
    # string in either direction (target="Visual Studio Code" vs. real name
    # "Code"; target="Chrome" vs. "Google Chrome") — checked both directions.
    t, a = requested_app.strip().casefold(), actual_app.casefold()
    return t in a or a in t


def _post_action_snapshot(
    before: ScreenSnapshot, *, expect: str = "", requested_app: str = "",
) -> tuple[ScreenSnapshot, str]:
    """Poll screen state until it changes and stabilizes, or the timeout expires.

    Ported from reference 2026-07-21: a single post-action capture (the previous
    behavior here) risked photographing a slow transition — app launch, a
    save-sheet's slide-in animation, a page still loading — mid-animation.
    On reference that's recoverable (OCR text is a second channel independent of
    the frame), but on this fork the screenshot IS the model's only view of
    the result, so a half-finished frame is a worse silent-failure class here,
    not a milder one.

    expect=/requested_app= extend (never shorten) how long this waits: a
    still-unsatisfied expect or app-frontmost check keeps polling up to
    _POST_ACTION_EXTENDED_TIMEOUT instead of giving up at 1.2s — real-usage
    reports showed `open_url`/`key(enter)` flagged app_not_frontmost/
    expectation_not_met while the very screenshot returned alongside the
    warning showed the action had, in fact, already succeeded a moment later."""
    _progress("กำลังตรวจสอบผลลัพธ์บนหน้าจอ…")
    waiting_on_condition = bool(expect) or bool(requested_app)
    deadline = time.monotonic() + max(0.0, _POST_ACTION_TIMEOUT)
    extended_deadline = (
        time.monotonic() + max(_POST_ACTION_TIMEOUT, _POST_ACTION_EXTENDED_TIMEOUT)
        if waiting_on_condition else deadline
    )
    chosen: ScreenSnapshot | None = None
    previous_state: tuple[str, str] | None = None
    semantic_changed = False
    visual_changed = False
    stable_hits = 0
    while True:
        current = _capture_snapshot(notify=False)
        semantic_now, visual_now = _screen_change_signals(before, current)
        semantic_changed = semantic_changed or semantic_now
        visual_changed = visual_changed or visual_now
        current_state = (current.fingerprint, current.visual_fingerprint)
        stable_hits = stable_hits + 1 if previous_state is not None and current_state == previous_state else 1
        previous_state = current_state
        if chosen is not None and chosen is not current:
            Path(chosen.image_path).unlink(missing_ok=True)
        chosen = current
        now = time.monotonic()
        condition_pending = (
            (bool(expect) and not _expected(current, expect))
            or (bool(requested_app) and not _app_matches(requested_app, current.app))
        )
        changed = semantic_changed or visual_changed
        if changed and stable_hits >= 2 and not condition_pending:
            break
        if now >= deadline and not condition_pending:
            break
        if now >= extended_deadline:
            break
        time.sleep(max(0.0, _POST_ACTION_POLL))
    assert chosen is not None
    return _publish(chosen), ("changed" if changed else "no_visible_change")


def _result(prefix: str, snapshot: ScreenSnapshot, effect: str = "") -> str:
    head = f"{prefix} effect={effect}" if effect else prefix
    out = f"{head}\n{format_snapshot(snapshot, max_chars=720)}"
    return out if len(out) <= _SUMMARY_MAX_CHARS else out[:_SUMMARY_MAX_CHARS - 34].rstrip() + "\n… observation truncated"


_COORD_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*$")
_COORD_ACTIONS = {"click", "double_click", "triple_click", "right_click", "hover", "scroll"}
_COORD_CLICK_ACTIONS = {"click", "double_click", "triple_click", "right_click"}
_COORD_AIM_TOLERANCE_PX = 18.0


def _coord_point_from_snapshot(snapshot: ScreenSnapshot, coord: str) -> tuple[tuple[float, float] | None, str | None]:
    """Resolve coord=x,y from capture pixels into display points for one observation."""
    match = _COORD_RE.fullmatch(coord or "")
    if not match:
        return None, _error('coord must be "x,y" in capture pixels, e.g. coord="640,420"')
    x, y = float(match.group(1)), float(match.group(2))
    width, height = snapshot.capture_size
    if not (0 <= x < width and 0 <= y < height):
        return None, _error(f"coord {x:g},{y:g} is outside observation size {width}x{height}")
    return _backend.capture_px_to_points(x, y, *snapshot.capture_size, *snapshot.screen_size), None


def _coord_geometry_matches(expected: ScreenSnapshot, current: ScreenSnapshot) -> bool:
    """Keep coordinate actions tied only to the newest screenshot geometry.

    Vision is the authority for what is at the pixel. AX/OCR/fingerprints are advisory and
    must not veto an ordinary coordinate action. Refuse only when screenshot/display geometry
    changed, because then capture-pixel coordinates no longer map to the same physical screen.
    """
    if expected.capture_size != current.capture_size:
        return False
    return all(abs(a - b) <= 0.5 for a, b in zip(expected.screen_size, current.screen_size))


def _coord_cursor_is_aimed(snapshot: ScreenSnapshot, coord: str, *, tolerance_px: float = _COORD_AIM_TOLERANCE_PX) -> bool:
    """Require the cursor visible in the latest screenshot to already be near coord."""
    match = _COORD_RE.fullmatch(coord or "")
    if not match or snapshot.cursor_capture is None:
        return False
    x, y = float(match.group(1)), float(match.group(2))
    cursor_x, cursor_y = snapshot.cursor_capture
    return (cursor_x - x) ** 2 + (cursor_y - y) ** 2 <= tolerance_px ** 2


def _point_from_snapshot(snapshot: ScreenSnapshot, target: str, near: str) -> tuple[tuple[float, float] | None, str | None]:
    """Resolve visible target text only inside the current frontmost window."""
    if snapshot.window_bounds is None:
        return None, _error(
            "target text cannot be safely localized without frontmost-window bounds — "
            "use element_id from [OBS] or coord with the latest observation_id"
        )
    wx, wy, ww, wh = snapshot.window_bounds
    screen_w, screen_h = snapshot.screen_size
    boxes: list[dict] = []
    for candidate in snapshot.ocr_boxes:
        try:
            center_x = (float(candidate["x"]) + float(candidate["w"]) / 2) * screen_w
            center_y = (1 - (float(candidate["y"]) + float(candidate["h"]) / 2)) * screen_h
        except (KeyError, TypeError, ValueError):
            continue
        if wx - 8 <= center_x <= wx + ww + 8 and wy - 8 <= center_y <= wy + wh + 8:
            boxes.append(candidate)
    box, problem = _find_target(boxes, target, near)
    if problem:
        return None, problem
    x = (float(box["x"]) + float(box["w"]) / 2) * snapshot.capture_size[0]
    y = (1 - (float(box["y"]) + float(box["h"]) / 2)) * snapshot.capture_size[1]
    return _backend.capture_px_to_points(x, y, *snapshot.capture_size, *snapshot.screen_size), None


def _inspect(observation_id: str, element_id: str, question: str) -> str:
    latest = _LATEST_OBSERVATION[0]
    if latest is None or latest.observation_id != observation_id:
        return _error("inspect requires the latest observation_id from [OBS]")
    path = latest.image_path
    crop = ""
    try:
        element = latest.get(element_id) if element_id else None
        if element_id and element is None:
            return _error(f"element not found in {observation_id}: {element_id}")
        if element:
            import cv2
            frame = cv2.imread(path)
            if frame is None: return _error("retained screenshot is unreadable")
            sx, sy = latest.capture_size[0] / latest.screen_size[0], latest.capture_size[1] / latest.screen_size[1]
            x, y, w, h = element.bounds; pad = max(w, h, 80.0)
            x0, y0 = max(0, round((x-pad)*sx)), max(0, round((y-pad)*sy)); x1, y1 = min(frame.shape[1], round((x+w+pad)*sx)), min(frame.shape[0], round((y+h+pad)*sy))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle: crop = handle.name
            if x1 <= x0 or y1 <= y0 or not cv2.imwrite(crop, frame[y0:y1, x0:x1]): return _error("could not make inspect crop")
            path = crop
        # `inspect` supersedes computer's full observation with its sharper crop.
        # No inner LLM: the next ReAct step sees this computer-
        # owned crop directly and performs the interpretation itself.
        _progress("encoding computer detail…")
        _queue_latest_capture(path)
        focus = question.strip()
        result = (
            f"[ok] inspect observation={observation_id}"
            + (f" element={element_id}" if element_id else "")
            + "\n[computer detail image attached for direct vision]"
        )
        if focus:
            result += f"\ninspection focus: {focus}"
        return result
    except Exception as exc:
        return _error(f"inspect failed: {exc}")
    finally:
        if crop: Path(crop).unlink(missing_ok=True)


_EXPECT_KINDS = {"app", "window", "text", "focus"}


def _expect_uses_recognized_kind(expect: str) -> bool:
    kind, sep, _ = expect.partition(":")
    return bool(sep) and kind.strip().casefold() in _EXPECT_KINDS


def _normalize_visible_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _expect_text_visible(snapshot: ScreenSnapshot, needle: str) -> bool | None:
    """Check text using evidence from the frontmost window only.

    Accessibility elements come from the focused window root and are safe to
    use directly. OCR is only trustworthy when Accessibility supplied focused
    window bounds; otherwise whole-screen OCR could match a background window,
    so report unknown instead of a false positive/negative.
    """
    normalized = _normalize_visible_text(needle)
    if not normalized:
        return False
    if any(
        e.source == "ax" and normalized in _normalize_visible_text(e.text)
        for e in snapshot.elements
    ):
        return True
    if snapshot.window_bounds is None:
        return None
    wx, wy, ww, wh = snapshot.window_bounds
    for box in snapshot.ocr_boxes:
        try:
            text = _normalize_visible_text(box.get("text") or "")
            if not text:
                continue
            x = (float(box["x"]) + float(box["w"]) / 2) * snapshot.screen_size[0]
            y = (1 - (float(box["y"]) + float(box["h"]) / 2)) * snapshot.screen_size[1]
            if wx - 8 <= x <= wx + ww + 8 and wy - 8 <= y <= wy + wh + 8:
                if normalized in text:
                    return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


def _expected(snapshot: ScreenSnapshot, expect: str) -> bool | None:
    # Only "app:"/"window:"/"text:"/"focus:" are recognized kind prefixes —
    # treating ANY prefix before the first colon as a kind (the previous
    # behavior) silently broke every URL- or time-shaped expect string:
    # expect="https://youtube.com" read kind="https", needle="//youtube.com"
    # (never matches); expect="3:45" read kind="3", needle="45". "text:" is
    # kept as an explicit synonym for the plain whole-screen search below
    # (same lookup, just an explicit kind) since it's an established
    # convention elsewhere in this tool's lineage; any other prefix is
    # treated as literal, whole-string text instead.
    #
    # Return is tri-state: True/False are a real verdict, None means "not
    # checkable right now" (currently only focus: under unavailable
    # Accessibility) — never silently collapsed to False, which would read
    # as a confirmed miss instead of "couldn't check".
    kind, sep, value = expect.partition(":")
    kind_norm = kind.strip().casefold()
    if sep and kind_norm in _EXPECT_KINDS:
        needle = value.strip().casefold()
        if kind_norm == "focus":
            if snapshot.accessibility_status != "ok":
                return None
            if not needle:
                return any(e.focused for e in snapshot.elements)
            return any(e.focused and needle in e.text.casefold() for e in snapshot.elements)
        if not needle: return False
        if kind_norm == "app": return needle in snapshot.app.casefold()
        if kind_norm == "window": return needle in snapshot.window.casefold()
        return _expect_text_visible(snapshot, needle)
    needle = expect.strip().casefold()
    if not needle: return False
    return _expect_text_visible(snapshot, needle)


_EDITABLE_AX_ROLES = {"AXTextField", "AXTextArea", "AXSearchField", "AXComboBox"}


def _focused_editable(snapshot: ScreenSnapshot) -> ScreenElement | None:
    if snapshot.accessibility_status != "ok":
        return None
    return next(
        (
            e for e in snapshot.elements
            if e.source == "ax" and e.focused and e.role in _EDITABLE_AX_ROLES
        ),
        None,
    )


def _type_input_verification(before: ScreenSnapshot, after: ScreenSnapshot) -> str:
    """Verify typing without retaining or exposing editable-field contents."""
    before_edit = _focused_editable(before)
    after_edit = _focused_editable(after)
    if before_edit is None or after_edit is None:
        return "input_unverified"
    if before_edit.role != after_edit.role:
        return "input_unverified"
    distance_sq = (
        (before_edit.point[0] - after_edit.point[0]) ** 2
        + (before_edit.point[1] - after_edit.point[1]) ** 2
    )
    if distance_sq > 120.0 ** 2:
        return "input_unverified"
    if (
        before_edit.value_digest != after_edit.value_digest
        and (before_edit.value_digest or after_edit.value_digest)
    ):
        return "input_verified"
    return "input_unverified+focus_verified"


# Keep the bound-tool schema small. _computer_impl below is the extracted real logic
# from the @tool-decorated function — no clean single return point existed, so the thin
# @tool wrapper below calls this instead of wrapping every scattered `return`).
# Execution training belongs in this first-call manual, not graph.py.  The
# always-visible @tool docstring below retains only routing/schema facts needed
# to form the first call; after that call, this playbook enters the same ReAct
# turn and teaches the model how to finish and recover without permanently
# bloating every bound-tool schema.
_SYNTAX_MANUAL = (
    "\n\n[computer usage notes — screenshot → aim → look → click → look]\n"
    "VISION-FIRST LOOP — 1) If state is unknown, see. 2) Treat the newest screenshot as the source of truth. 3) Visually choose "
    "the needed pixel. For coordinate clicks, first hover(coord=\"x,y\", observation_id=<latest>) to move the cursor there. "
    "4) Read the NEW screenshot and confirm visually that the cursor is on the intended control. 5) Only then click the same coord "
    "using that new observation_id. 6) Read the post-click screenshot and judge whether it worked. AX/OCR/eN are helpers, not "
    "coordinate-action gates. Never chain coordinates from an old screenshot.\n"
    "MOUSE — click focuses/selects/presses; double_click opens a file/item; triple_click selects a text block; right_click "
    "opens a context menu; hover reveals hidden controls/tooltips; drag uses target=<source text>, text=<drop-target text>; "
    "scroll uses direction=up/down/left/right, amount=<positive lines>, and target/element_id when a specific pane must scroll. "
    "The returned image shows the live cursor plus capture-pixel coordinates. For coordinate clicks, move first with "
    "hover(coord=\"x,y\", observation_id=<latest>), inspect the returned screenshot, then click the SAME coord with that new "
    "observation_id. The tool mechanically refuses a coordinate click unless the cursor visible in the latest screenshot is already "
    "near that point. No AX element, OCR label, semantic anchor, or Dock tooltip is required to approve an ordinary coordinate "
    "click. If text matches several places, retry with near=top/bottom/left/right/center/corners. After a menu, right-click, hover, "
    "drag, or scroll, inspect the newly revealed state before acting.\n"
    "KEYBOARD — type(text=...) inserts literal Unicode into the currently focused editable field only. Type results distinguish "
    "input_verified from input_unverified+focus_verified/input_unverified without exposing secure-field values. key(text=...) sends "
    "ONE key/combo: enter, tab, shift+tab, escape, space, arrows, cmd+a/c/v/s/f/l, shift+arrow. To replace field text: click "
    "field → verify focus → key(cmd+a) → type(new text) → verify. To submit: key(enter), NEVER type('return') and NEVER "
    "type(text='', modifiers='return'). Put modifiers inside key text (cmd+s), not the modifiers argument; modifiers is for "
    "click/drag selection. Tab moves focus, shift+tab moves back, escape safely closes a menu/dialog. Switch apps with "
    "open_app, not cmd+tab. Never type passwords, OTPs, payment data, or credentials.\n"
    "COMMON APP PATTERNS — form: click field → type → tab/click next → type → click Submit → verify result. Menu/dialog: "
    "click menu/button → read new OBS → click item/choice → verify dismissal/result. Editor: click text → cmd+a only when the "
    "whole field/document should be replaced → type → cmd+s → verify content/title. Finder/list: double_click opens; "
    "right_click then inspect exposes actions; modifier-click selects multiple items. Multi-pane apps: scroll the named pane, "
    "not wherever the pointer happened to remain.\n"
    "RECOVERY EXAMPLES — click Play → no_visible_change → see/inspect → dismiss safe popup or choose the current Play → "
    "click once → verify progress. Type lands in the wrong place → stop typing → see → click the intended editable field → "
    "cmd+a only there → type again → verify visible text. Drag changes nothing → see → select the current source and visible "
    "drop target → retry once; otherwise report the boundary instead of guessing.\n"
    "APP/WEB — open_app(target=\"TextEdit\") launches/raises an app. open_url(target=<full http(s) URL>, "
    "app=\"Google Chrome\") opens a site directly in the requested real browser; prefer this over open_app→cmd+l→type. "
    "Use it when the user names Chrome/Safari; generic web automation may use browser_use.\n"
    "YOUTUBE DJ RECIPE — 1) open_url(target=\"https://www.youtube.com/results?search_query=<song/artist/genre/playlist/mix>\", "
    "app=\"Google Chrome\"); spaces are allowed. 2) Read the new screen; click the requested result, or a clearly labeled "
    "mix/playlist when no song was named. 3) Verify player/title/progress before saying it plays. 4) With player focused: "
    "key(space)=play/pause, key(shift+n)=next, key(m)=mute, key(right/left)=seek, key(up/down)=volume; verify each. "
    "Prefer a mix/playlist for continuous DJ playback; do not loop indefinitely.\n"
    "POPUPS/ADS — inspect first; click an unambiguous Skip/ข้าม or Play once and verify. Never bypass CAPTCHA, sign-in, "
    "age/subscription/region gates, or ads deceptively; stop for manual intervention.\n"
    "EXPECT — pass expect=<plain text> (optionally expect=\"text:<plain text>\", equivalent) to check it appears "
    "inside the frontmost window after the action; background-window OCR is never accepted as verification. Use "
    "expect=\"app:<name>\"/\"window:<text>\" for the frontmost app/window, or "
    "expect=\"focus:<label>\" to check that the element whose visible text contains <label> is now focused (e.g. after "
    "clicking a field) — expect=\"focus:\" alone checks that ANYTHING is focused. Only \"app:\", \"window:\", \"text:\", "
    "and \"focus:\" are recognized prefixes — a colon anywhere else (a URL like https://youtube.com, a timestamp like "
    "3:45) is part of the plain text, not a prefix, so pass those as-is; a state description with NO colon (e.g. "
    "\"Search field focused\") is also searched as literal on-screen text and will never match — use focus:<label> "
    "instead for that. A failed unrecognized-form expect returns a 'hint:' line naming the right prefix; read and use "
    "it on the next call rather than repeating the same expect= verbatim. When expect= (or open_app/open_url's app=) "
    "hasn't been satisfied yet, the tool keeps polling a few extra seconds before giving up — slower than the default "
    "settle, never faster. The result's effect combines independent signals, e.g. changed+verified or "
    "changed+expectation_not_met: a '+expectation_not_met'/'+app_not_frontmost' suffix means only that specific check "
    "missed, not that the screen failed to change; '+expect_unknown' means the requested focus/text check lacked "
    "trustworthy Accessibility/window evidence — neither is a failed action. Look at the attached image/[OBS] to judge what "
    "actually happened rather than treating any [warning] as a failed action.\n"
    "KNOWN LIMIT — in Chrome/Chromium, focus: is reliable for the browser's own controls (address bar, buttons) but "
    "not for a field INSIDE a loaded web page (a page's search box, a form input): Chrome only builds its web-content "
    "accessibility tree once it detects a persistent assistive-technology client, which this tool is not, so "
    "page-content elements may be invisible to Accessibility even though ax=ok. For those, verify with "
    "expect=\"text:<value>\" instead of focus:."
)


def _validate_web_url(url: str) -> None:
    parsed = urlsplit(url.strip())
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("open_url requires a valid http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("open_url refuses URLs containing credentials")


def _computer_impl(action: str, target: str = "", text: str = "", coord: str = "", direction: str = "", amount: int = 3, near: str = "", element_id: str = "", observation_id: str = "", question: str = "", expect: str = "", modifiers: str = "", app: str = "") -> str:
    if not _ACTION_LOCK.acquire(blocking=False):
        return _error("another computer action is already running — retry after it finishes")
    try:
        action = action.strip().lower()
        _phase(f"🖥 ควบคุมหน้าจอ: {action}" + (f" ({target or text or element_id})"[:60] if (target or text or element_id) else ""))
        if action not in {"see", "inspect", "click", "double_click", "triple_click", "right_click", "type", "key", "scroll", "open_app", "open_url", "drag", "hover"}:
            return _error(f"unknown action: {action}")
        if action == "inspect":
            return _inspect(observation_id, element_id, question)
        if _backend.failsafe_abort():
            return _error("failsafe abort — user took the mouse")
        _effective_max = _ACTION_MAX_OVERRIDE[0] or _ACTION_MAX
        if _ACTION_ATTEMPTS[0] >= _effective_max:
            return _error("computer action limit reached — stop and report the current screen")
        if action == "see":
            _ACTION_ATTEMPTS[0] += 1
            _LAST_SIGNATURE[0] = "see"
            snapshot = _publish(_capture_snapshot())
            _LAST_SCREEN_TEXT[0] = "\n".join(str(b.get("text", "")) for b in snapshot.ocr_boxes); _LAST_FRONTMOST_APP[0] = snapshot.app
            return _result("[ok] see", snapshot)
        before = _capture_snapshot()
        if action in _CONTEXT_DEPENDENT_ACTIONS and _LAST_FRONTMOST_APP[0] and before.app != _LAST_FRONTMOST_APP[0]:
            _publish(before)
            return _error(f"frontmost app changed since last action — latest is {before.observation_id}; inspect it before acting")
        # near/modifiers included (ported from reference audit F2): without them, a
        # click that hit an ambiguity error recommending "retry with near="
        # was itself refused as a "repeated action" on retry (identical
        # action/target/text otherwise), and a plain click followed by the
        # SAME click again with modifiers= (e.g. cmd-click to extend a
        # selection) was wrongly treated as a no-op repeat of the first click.
        signature = "|".join((
            action, target.strip().casefold(), text.strip().casefold(), direction,
            str(amount), coord.strip().casefold(), element_id, observation_id,
            near.strip().casefold(), modifiers.strip().casefold(),
            app.strip().casefold(),
        ))
        if (
            signature == _LAST_SIGNATURE[0]
            and _LATEST_OBSERVATION[0]
            and before.fingerprint == _LATEST_OBSERVATION[0].fingerprint
            and before.visual_fingerprint == _LATEST_OBSERVATION[0].visual_fingerprint
        ):
            Path(before.image_path).unlink(missing_ok=True); return _error("repeated action, screen unchanged — change approach")
        chosen: ScreenElement | None = None
        coord_point: tuple[float, float] | None = None
        if coord:
            if action not in _COORD_ACTIONS:
                Path(before.image_path).unlink(missing_ok=True)
                return _error(f"coord is supported only for {', '.join(sorted(_COORD_ACTIONS))}")
            if element_id:
                Path(before.image_path).unlink(missing_ok=True)
                return _error("use either coord or element_id, not both")
            latest = _LATEST_OBSERVATION[0]
            if latest is None or not observation_id or observation_id != latest.observation_id:
                Path(before.image_path).unlink(missing_ok=True)
                return _error("coord requires the latest observation_id from see/previous action")
            if not _coord_geometry_matches(latest, before):
                _publish(before)
                return _error(
                    f"screen geometry changed since {observation_id}; latest is {before.observation_id}; "
                    "look at the new screenshot before using coordinates"
                )
            coord_point, problem = _coord_point_from_snapshot(latest, coord)
            if problem:
                Path(before.image_path).unlink(missing_ok=True)
                return problem
            assert coord_point is not None
            if action in _COORD_CLICK_ACTIONS and not _coord_cursor_is_aimed(latest, coord):
                Path(before.image_path).unlink(missing_ok=True)
                return _error(
                    f"aim before coordinate click: hover(coord=\"{coord}\", observation_id=\"{observation_id}\"), "
                    "look at the returned screenshot to confirm the cursor is on the intended control, then click the same "
                    "coord using that new observation_id"
                )
        if element_id:
            latest = _LATEST_OBSERVATION[0]
            if latest is None or observation_id != latest.observation_id:
                Path(before.image_path).unlink(missing_ok=True); return _error("element_id requires the latest observation_id; call see")
            if before.fingerprint != latest.fingerprint or before.app != latest.app:
                _publish(before); return _error(f"screen changed since {observation_id}; latest is {before.observation_id}")
            chosen = latest.get(element_id)
            if chosen is None:
                Path(before.image_path).unlink(missing_ok=True); return _error(f"element not found: {element_id}")
        if action == "type" and (
            any(marker in text.casefold() for marker in _PASSWORD_MARKERS)
            or any(marker in _LAST_CLICK_TARGET[0].casefold() for marker in _PASSWORD_MARKERS)
            or any(e.focused and e.role == "AXSecureTextField" for e in before.elements)
        ):
            return _error("refusing to type into what looks like a password field — enter credentials manually")
        if action == "key" and _normalize_combo(text) in _FORBIDDEN_KEY_COMBOS:
            return _error(f"refusing dangerous hotkey: {text}")
        if _DESTRUCTIVE_GUARD[0] and action in {"click", "double_click", "triple_click", "right_click", "drag", "key"}:
            # `key` is checked against an exact, normalized hotkey set (a bare
            # Backspace/Delete press while editing text is ordinary, not the
            # same risk as a Delete/Trash BUTTON) — never a loose substring
            # match against the marker lists, which would false-positive on
            # every routine typo correction. click/double_click/right_click/drag
            # match target/element text by substring since that IS a visible
            # label — the marker list itself is tiered by supervision context
            # (see _DESTRUCTIVE_MARKERS_SUPERVISED/_UNSUPERVISED above): an
            # awake-fired turn always sets action_max, an ordinary turn's
            # default guard never does, so that existing distinction alone
            # picks the right tier without a new per-turn flag.
            _markers = _DESTRUCTIVE_MARKERS_UNSUPERVISED if _ACTION_MAX_OVERRIDE[0] is not None else _DESTRUCTIVE_MARKERS_SUPERVISED
            destructive_hit = (
                _normalize_combo(text) in _DESTRUCTIVE_KEY_COMBOS if action == "key" else
                any(
                    marker in str(c).casefold()
                    for c in (target, chosen.text if chosen is not None else None,
                              text if action == "drag" else None)
                    if c for marker in _markers
                )
            )
            if destructive_hit:
                # Terminate, not just refuse-and-let-retry: force the ceiling so no
                # further computer action can execute this turn — inspect is the only
                # exception (it's gated before this check, always read-only on a
                # retained crop); every other action, including see, hits the
                # action-limit error above from here on. Applies uniformly whether
                # this is an unsupervised background turn (always blocked, no
                # exception) or an ordinary turn whose own request never said to
                # delete/remove anything (see graph.py's per-turn intent check) —
                # either way, one shot at it, not a free retry loop. A later turn
                # where the user explicitly confirms starts with a fresh
                # reset_computer_guards() and is unaffected.
                _ACTION_ATTEMPTS[0] = _ACTION_MAX_OVERRIDE[0] or _ACTION_MAX
                return _error(
                    "refusing an action that looks delete/remove-related — this turn's own request "
                    "never explicitly asked for deletion/removal (or this is an unsupervised background "
                    "turn, where it is never allowed) — computer access terminated for this turn; report "
                    "this to the user and ask for explicit confirmation instead of proceeding"
                )
        _ACTION_ATTEMPTS[0] += 1
        _LAST_SIGNATURE[0] = signature
        _progress(f"กำลัง {action}: {(target or text or element_id)[:40]}…" if (target or text or element_id) else f"กำลัง {action}…")
        type_delivery = ""
        if action in {"click", "double_click", "triple_click", "right_click"}:
            point, problem = (
                (coord_point, None) if coord_point is not None else
                ((chosen.point, None) if chosen else _point_from_snapshot(before, target, near))
            )
            if problem:
                return problem
            assert point is not None
            x, y = point
            try:
                _backend.click(
                    x, y,
                    button="right" if action == "right_click" else "left",
                    count=3 if action == "triple_click" else (2 if action == "double_click" else 1),
                    modifiers=modifiers,
                )
            except ValueError as exc:
                return _error(str(exc))
            label = chosen.text if chosen else target or coord
            _log_action(action, label, (x, y))
            _LAST_CLICK_TARGET[0] = label
        elif action == "hover":
            point, problem = (
                (coord_point, None) if coord_point is not None else
                ((chosen.point, None) if chosen else _point_from_snapshot(before, target, near))
            )
            if problem:
                return problem
            assert point is not None
            _backend.hover(*point)
            label = chosen.text if chosen else target or coord
            _log_action(action, label, point)
        elif action == "type":
            if not text:
                return _error("type requires text")
            type_delivery = _backend.type_text(text)
            _log_action(action)
        elif action == "key":
            _backend.key(text)
            _log_action(action, text)
        elif action == "scroll":
            point = coord_point
            if point is None and target:
                point, problem = (chosen.point, None) if chosen else _point_from_snapshot(before, target, near)
                if problem:
                    return problem
            _backend.scroll(direction, amount, point=point)
            _log_action(action, target or coord or direction, point)
        elif action == "drag":
            if not text:
                return _error("drag requires text as the drop-target's visible text")
            # near= disambiguates the SOURCE only (audit F8) — applying it to
            # the drop target too meant a near= chosen to pick out the source
            # could just as easily filter out the correct drop target, since
            # the two are rarely in the same screen region by definition.
            from_point, problem = (chosen.point, None) if chosen else _point_from_snapshot(before, target, near)
            if problem:
                return problem
            to_point, problem = _point_from_snapshot(before, text, "")
            if problem:
                return problem
            try:
                _backend.drag(from_point[0], from_point[1], to_point[0], to_point[1], modifiers=modifiers)
            except ValueError as exc:
                return _error(str(exc))
            _log_action(action, f"{target} -> {text}")
        elif action == "open_app":
            _backend.open_app(target)
            _log_action(action, target)
        else:
            _validate_web_url(target)
            _backend.open_url(target, app=app)
            _log_action(action, f"{app or 'default'}: {target}")
        # Settle before observing: live-reproduced (2026-07-17) a save-sheet's
        # slide-in animation not finished when the very next action (typing the
        # filename) fired, so it landed before the field was actually focused/
        # selected -- the default "Untitled" silently never got replaced, no
        # error anywhere. cmd+s-class actions are the common trigger but any
        # click/key can open a sheet/dialog, so this applies broadly rather
        # than special-casing specific hotkeys.
        _backend.settle()
        requested_app = target if action == "open_app" else app if action == "open_url" else ""
        after, ui_effect = _post_action_snapshot(before, expect=expect, requested_app=requested_app)
        Path(before.image_path).unlink(missing_ok=True)
        _LAST_SCREEN_TEXT[0] = "\n".join(str(b.get("text", "")) for b in after.ocr_boxes); _LAST_FRONTMOST_APP[0] = after.app
        # Independent verification signals, combined rather than overwritten —
        # the previous version set a single `effect` string for each check in
        # turn, so supplying expect= silently discarded both whether the UI
        # actually changed AND the app_not_frontmost guard's own result. That
        # produced exactly the live-usage symptom this fixes: a `[warning]`
        # response whose own attached screenshot showed the action had, in
        # fact, succeeded — the model had no way to tell "verification heuristic
        # missed" apart from "nothing happened".
        notes: list[str] = []
        if action == "type":
            notes.append(_type_input_verification(before, after))
            if type_delivery == "clipboard_changed_externally":
                notes.append("clipboard_changed_externally")
        if requested_app.strip() and not _app_matches(requested_app, after.app):
            # `open -a` can return 0 without the target ever actually becoming
            # frontmost — reporting [ok] here would let the model believe
            # subsequent type/key calls land in the target app when they're
            # still landing wherever WAS frontmost (live-reproduced 2026-07-21
            # on reference walk R7-CB, ported here — same design, same gap).
            notes.append("app_not_frontmost")
        expect_hint = ""
        if expect:
            verdict = _expected(after, expect)
            if verdict is None:
                # A genuinely unknown state (for example focus: without AX, or
                # text: without trustworthy focused-window bounds) must not
                # collapse into expectation_not_met.
                notes.append("expect_unknown")
            elif verdict:
                notes.append("verified")
            else:
                notes.append("expectation_not_met")
                if not _expect_uses_recognized_kind(expect):
                    # The failure is because expect used no recognized kind —
                    # because it wasn't "app:"/"window:"/"text:"/"focus:" —
                    # so it was searched as literal on-screen text and can
                    # never match a prose state description. Tell the model
                    # the right form instead of letting it repeat the same
                    # broken expect= verbatim.
                    expect_hint = (
                        '\nhint: expect="' + expect[:60] + '" was searched as literal on-screen '
                        'text and not found — for state checks use expect="focus:<label>" '
                        '(is it focused now), "app:<name>", or "window:<text>" instead.'
                    )
        verification_warning = bool(
            {"app_not_frontmost", "expectation_not_met", "expect_unknown", "clipboard_changed_externally"} & set(notes)
        ) or (
            action == "type"
            and "input_verified" not in notes
            and "verified" not in notes
        )
        if verification_warning:
            prefix = f"[warning] {action}"
        elif ui_effect == "changed" or "verified" in notes:
            prefix = f"[ok] {action}"
        else:
            prefix = f"[no_effect] {action}"
        effect = "+".join([ui_effect, *notes]) if notes else ui_effect
        return _result(prefix, after, effect) + expect_hint
    except Exception as exc:
        return _error(str(exc))
    finally:
        _ACTION_LOCK.release()


@tool
def computer(action: str, target: str = "", text: str = "", coord: str = "", direction: str = "", amount: int = 3, near: str = "", element_id: str = "", observation_id: str = "", question: str = "", expect: str = "", modifiers: str = "", app: str = "") -> str:
    """Control a native Mac app one guarded action at a time. ENDEAVOR_LOCAL_AGENT_TH sees the newest screen image
    after every call and also receives a compact [OBS] Accessibility/OCR element list.

    Start with action="see" when state is unknown — queues the current image and returns [OBS obs_N]
    with eN elements to act on (scroll to reveal more). Prefer element_id="eN" + observation_id="obs_N"
    over target text for drag and non-visual fallbacks. For ordinary mouse actions, use the newest screenshot directly.
    Coordinate clicks follow move → look → click → look: first hover(coord="x,y", observation_id=<latest>), inspect the
    returned screenshot and confirm the visible cursor is on the intended control, then click the same coord with that new
    observation_id. AX/OCR/semantic anchors and Dock tooltips are helpers only and do not gate coordinate approval. The tool
    mechanically refuses a coordinate click unless the latest screenshot shows the cursor already near that point. Verify
    the new image/[OBS] after every mutation. Post-action change detection combines
    semantic AX/OCR state with a compact visual fingerprint; type additionally reports input_verified or an explicit
    unverified/focus-only state without retaining secure-field contents.

    Silent-failure recovery: click → no_visible_change → detect failure from effect/image/[OBS] →
    see or inspect → retry a different current element/target. Never repeat or claim success. A
    delete/remove-looking click/drag target or permanently-remove hotkey is refused UNLESS the
    user's own current message explicitly asked for deletion/removal.

    Actions: see/inspect; click/double_click/triple_click/right_click/hover/drag; type literal text;
    key(text=one key/combo, e.g. enter or cmd+s); scroll(direction, amount, optional target/element);
    open_app(target=installed app); open_url(target=full HTTP(S) URL, app=installed browser). Use
    open_url when the user explicitly wants Chrome/Safari to open a site. Optional expect=<text>,
    expect="app:<name>"/"window:<text>", or expect="focus:<label>" (is that element focused now)
    asks the tool to keep waiting (up to a few extra seconds, never less than normal) for that
    condition and report effect=verified/expectation_not_met/expect_unknown instead of just
    changed/no_effect — the ui-changed and expectation signals are independent, so a screen that
    clearly changed is never reported as if nothing happened just because expect= missed. A bare
    prose expect with no "app:"/"window:"/"text:"/"focus:" prefix (e.g. "field focused") is searched
    as literal frontmost-window text and will never match — use focus:<label> for that, not prose. Detailed
    mouse/keyboard, multi-app workflows, recovery, media, and safety notes return once after the
    first call each turn.
    ❌ submit typed URL → type(text="return")
    ✅ submit typed URL → key(text="enter")
    ❌ site requested → open_app(app="Google Chrome")
    ✅ site requested → open_url(target="https://www.youtube.com", app="Google Chrome")
    """
    from tools._call_guard import first_call_this_turn
    first = first_call_this_turn("computer")
    result = _computer_impl(
        action=action, target=target, text=text, coord=coord, direction=direction,
        amount=amount, near=near, element_id=element_id,
        observation_id=observation_id, question=question, expect=expect, modifiers=modifiers,
        app=app,
    )
    if first and _SYNTAX_MANUAL:
        result += _SYNTAX_MANUAL
    return result
