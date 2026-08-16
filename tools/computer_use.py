# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Single-step, guarded native-computer actions, forked from
ENDEAVOR_LOCAL_AGENT_MAX's tools/computer_use.py (byte-identical logic —
only this header changed).

TH's main model is TEXT-ONLY and — unlike a fork with its own small VLM
(config.py has no VL_BASE_URL/VL_MODEL here) — the code below still tries the
optional one-line [vision] gist described below, but that lookup always fails
closed (caught, returns "") on this fork, so every result is plain
"[ok] {action}\\n{OCR text}" with no [vision] line, ever. The only feedback
channel back to the model is text: Apple Vision OCR (exact, but misses
icons/charts). Never a raw pixel channel.
"""
from __future__ import annotations

import re
import tempfile
import threading
import json
import time
from datetime import datetime, timezone
from pathlib import Path

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
# Two tiers, not one — live-reproduced 2026-07-21 (audit F1): the single broad
# list below wrongly caught "Format" (TextEdit/Pages menu-bar item — pure
# text-formatting, not disk-erase) and "Trash" (Finder sidebar item — opening
# it to LOOK is not emptying it) the moment the destructive guard became the
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
# fires far more often per task than a single read_image/web call. Raised
# from MAX_VLM's donor value of 500 to 900: in MAX_VLM the model also SEES the
# screenshot pixel, so a terse OCR sample was enough context. In MAX the model
# is text-only — this OCR text (plus the one-line [vision] gist) is its ONLY
# view of the screen, so it needs more room to actually see what's there.
_SUMMARY_MAX_CHARS = 900
_PASSWORD_MARKERS = ("password", "รหัสผ่าน")
# Versioned semantic observation retained for the duration of one real agent
# turn.  The screenshot is kept only so action="inspect" can ask read_image a
# targeted follow-up about the exact pixels the model already reasoned from.
_OBSERVATION_SEQ = [0]
_LATEST_OBSERVATION: list[ScreenSnapshot | None] = [None]
_POST_ACTION_TIMEOUT = 1.2
_POST_ACTION_POLL = 0.12
_ACTION_LOCK = threading.Lock()


def reset_computer_guards() -> None:
    _ACTION_ATTEMPTS[0] = 0
    _LAST_SIGNATURE[0] = ""
    _LAST_SCREEN_TEXT[0] = ""
    _LAST_FRONTMOST_APP[0] = ""
    _LAST_CLICK_TARGET[0] = ""
    _ACTION_MAX_OVERRIDE[0] = None
    _DESTRUCTIVE_GUARD[0] = False
    _discard_snapshot(_LATEST_OBSERVATION[0])
    _LATEST_OBSERVATION[0] = None


def set_computer_turn_scope(action_max: int | None = None, block_destructive: bool = False) -> None:
    """Tighten this turn's computer-use ceiling below the normal per-turn defaults —
    called by graph.py for turns that run with no human watching the screen (e.g. an
    AWAKE-fired background turn). Takes effect immediately and lasts until the next
    reset_computer_guards() call (every real turn), so a tightened scope never leaks
    into a later, normal (human-supervised) turn."""
    _ACTION_MAX_OVERRIDE[0] = action_max
    _DESTRUCTIVE_GUARD[0] = block_destructive


def _discard_snapshot(snapshot: ScreenSnapshot | None) -> None:
    if snapshot and snapshot.image_path:
        try:
            Path(snapshot.image_path).unlink(missing_ok=True)
        except Exception:
            pass


def _next_observation_id() -> str:
    _OBSERVATION_SEQ[0] += 1
    return f"obs_{_OBSERVATION_SEQ[0]}"


def _publish_snapshot(snapshot: ScreenSnapshot) -> ScreenSnapshot:
    previous = _LATEST_OBSERVATION[0]
    if previous is not snapshot:
        _discard_snapshot(previous)
    _LATEST_OBSERVATION[0] = snapshot
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


# Reuses read_image's region vocabulary (region=<...> for zoom) for consistency.
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


def _region_name(box: dict) -> str:
    """Name the ninth of the screen `box` sits in, same thirds convention as
    _in_region. Text-only model has no other way to tell two ambiguous matches
    apart — the ambiguity error (below) names each match's region so the model
    can pick near= without ever having seen the screen."""
    cx = float(box["x"]) + float(box["w"]) / 2
    cy = float(box["y"]) + float(box["h"]) / 2  # Vision: high y = top
    row = "top" if cy > 2 / 3 else ("bottom" if cy < 1 / 3 else "")
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "")
    if row and col:
        return f"{row}-{col}"
    return row or col or "center"


def _find_target(boxes: list[dict], target: str, near: str = "") -> tuple[dict | None, str | None]:
    needle = target.strip().casefold()
    matches = [box for box in boxes if needle and needle in str(box.get("text", "")).casefold()]
    if not matches:
        return None, _error(f"text not found: {target}")
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
        # Text-only model cannot see the screen, so each match's label alone
        # ("Save", "Save") gives it nothing to disambiguate with — name each
        # match's region too, so it can pick near= without ever seeing pixels.
        labels = ", ".join(
            f"{box.get('text', '')!r} ({_region_name(box)})" for box in matches[:3]
        )
        hint = "" if near else ' — add near="top/bottom/left/right/center/top-left/top-right/bottom-left/bottom-right"'
        return None, _error(f"'{target}' matches {len(matches)} locations: {labels}{hint}")
    return matches[0], None


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


# Singleton guard for the background :8082 boot — repeated actions while the
# server is still warming must not each spawn another _ensure_vlm_server call.
_VLM_BOOT: dict = {"thread": None}


def _kick_vlm_server() -> None:
    """Port of read_image's :8082 startup (same `_ensure_vlm_server()` — spawn
    mlx_vlm.server if nothing serves VL_MODEL, poll up to 90s), but on a daemon
    thread: read_image rightly BLOCKS on it because vision IS that call's
    product, whereas here the action already succeeded and [vision] is
    garnish — a click result must not stall for a server boot. Actions during
    the warm-up return OCR-only; [vision] appears from the first action after
    the server becomes ready."""
    boot = _VLM_BOOT["thread"]
    if boot is not None and boot.is_alive():
        return

    def _boot() -> None:
        try:
            from .read_image import _ensure_vlm_server
            _ensure_vlm_server()
        except Exception:
            pass  # best-effort — a failed boot just means [vision] stays absent

    thread = threading.Thread(target=_boot, daemon=True, name="computer-vlm-boot")
    _VLM_BOOT["thread"] = thread
    thread.start()


def _describe_screen(path: str) -> str:
    """Best-effort one-line visual gist of the captured screen from the small
    VLM (read_image.py's server on VL_BASE_URL/VL_MODEL) — the only substitute
    a text-only main model gets for actually seeing the screenshot. Always
    best-effort: any failure (server not ready, network error, empty
    response) returns "" so the action result never fails merely because the
    describer failed — OCR text alone is still a usable result on its own.

    Server not up yet → kick the ported read_image boot in the background
    (_kick_vlm_server) and return "" for THIS action instead of blocking up
    to 90s inline the way read_image legitimately does.

    The HTTP call itself is read_image's `_call_vlm_request` — NOT a duplicate
    requests.post here — so this inherits the already-debugged path wholesale:
    de-loop sampling (temperature/frequency_penalty/presence_penalty, the small
    VLM repeats itself without them), error→string mapping, timeout, and each
    fork's transport specifics (the API fork's version carries Bearer/tunnel
    auth internally), with zero per-fork adaptation needed in this file."""
    try:
        from .read_image import _call_vlm_request, _encode_vlm, _vlm_serving
        from config import VL_BASE_URL, VL_MODEL

        if _vlm_serving(VL_BASE_URL, VL_MODEL) is not True:
            _kick_vlm_server()
            return ""
        _progress("encoding image…")
        image_part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_vlm(path)}"}}
        _progress("VLM: บรรยายหน้าจอ…")
        content = _call_vlm_request(image_part, (
            "Describe this screen in at most 2 short sentences. Focus on the "
            "frontmost window/app, any dialog or alert, and the primary "
            "actionable buttons."
        ), max_tokens=120)
        if content.startswith("[error]"):
            return ""  # best-effort channel — an error gist is worse than none
        return content
    except Exception:
        return ""


def _target_point(target: str, near: str) -> tuple[tuple[float, float] | None, str | None]:
    """Locate `target` via a fresh OCR pass and return its center in display points.
    Shared by click/scroll(target=)/drag so every text-targeted action resolves
    coordinates the same way — OCR box -> capture pixels -> display points."""
    boxes, _, capture_width, capture_height, screen_width, screen_height, _desc = _observe()
    box, problem = _find_target(boxes, target, near)
    if problem:
        return None, problem
    # Vision y is bottom-left normalized; CGEvent/display coordinates are top-left.
    capture_x = (float(box["x"]) + float(box["w"]) / 2) * capture_width
    capture_y = (1 - (float(box["y"]) + float(box["h"]) / 2)) * capture_height
    point = _backend.capture_px_to_points(capture_x, capture_y, capture_width, capture_height,
                                           screen_width, screen_height)
    return point, None


def _capture_snapshot(describe: bool = False, notify: bool = True) -> ScreenSnapshot:
    """Capture one full-resolution screen state without publishing it.

    The caller owns the returned temporary screenshot until it either calls
    ``_publish_snapshot`` or ``_discard_snapshot``. AX is optional and never
    allowed to break the OCR path.

    `notify=False` skips the per-call progress() emits — used by
    _post_action_snapshot's tight poll loop, where repeating "screenshot…/
    OCR…" every ~120ms is pure noise and, on Telegram, real edit_text() calls
    that can trip flood control (_cosmetic_edit swallows the error, but the
    status message just goes stale instead of updating).
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = handle.name
    try:
        if notify:
            _progress("กำลัง screenshot…")
        _backend.capture(path)
        if notify:
            _progress("Apple Vision OCR…")
        boxes = _ocr_layout(path)
        capture_size = _backend.image_dimensions(path)
        screen_size = _backend.primary_display_size_points()
        app = _backend.frontmost_app_name()
        try:
            from ._accessibility import read_frontmost
            ax_state = read_frontmost()
        except Exception:
            ax_state = {"status": "unavailable", "elements": []}
        description = _describe_screen(path) if describe else ""
        return build_snapshot(
            _next_observation_id(),
            app=app,
            ocr_boxes=boxes,
            ax_state=ax_state,
            image_path=path,
            capture_size=capture_size,
            screen_size=screen_size,
            description=description,
        )
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise


def _observe(describe: bool = False) -> tuple[list[dict], str, int, int, float, float, str]:
    """Capture + OCR the current screen. `describe=True` also asks the small
    VLM for a one-line gist FROM THE SAME capture file before it is deleted —
    only the post-action observe and action="see" pay for this; the cheap
    repeat-guard pre-check (screen-unchanged detection) never does."""
    snapshot = _capture_snapshot(describe=describe)
    try:
        text = "\n".join(str(box.get("text", "")) for box in snapshot.ocr_boxes if box.get("text"))
        return (
            snapshot.ocr_boxes,
            text or "[no OCR text found]",
            snapshot.capture_size[0], snapshot.capture_size[1],
            snapshot.screen_size[0], snapshot.screen_size[1],
            snapshot.description,
        )
    finally:
        _discard_snapshot(snapshot)


def _format_result(prefix: str, snapshot: ScreenSnapshot, effect: str = "") -> str:
    status = f"{prefix} effect={effect}" if effect else prefix
    out = f"{status}\n{format_snapshot(snapshot, max_chars=760)}"
    if snapshot.description:
        out += f"\n[vision] {snapshot.description}"
    # Preserve the machine-readable first line. _sample_coverage is right for a
    # raw OCR document, but wrapping the whole result replaces [ok]/[no_effect]
    # with its own document header and destroys the computer tool's status
    # contract. The semantic formatter already prioritizes actionable elements;
    # a final hard tail cap only bounds an unusually long VLM description.
    if len(out) > _SUMMARY_MAX_CHARS:
        out = out[:_SUMMARY_MAX_CHARS - 42].rstrip() + "\n… observation output truncated"
    return out


def _snapshot_text(snapshot: ScreenSnapshot) -> str:
    return "\n".join(str(box.get("text", "")) for box in snapshot.ocr_boxes if box.get("text")) or "[no OCR text found]"


def _target_point_in_snapshot(snapshot: ScreenSnapshot, target: str, near: str) -> tuple[tuple[float, float] | None, str | None]:
    box, problem = _find_target(snapshot.ocr_boxes, target, near)
    if problem:
        return None, problem
    capture_width, capture_height = snapshot.capture_size
    screen_width, screen_height = snapshot.screen_size
    capture_x = (float(box["x"]) + float(box["w"]) / 2) * capture_width
    capture_y = (1 - (float(box["y"]) + float(box["h"]) / 2)) * capture_height
    return _backend.capture_px_to_points(
        capture_x, capture_y, capture_width, capture_height, screen_width, screen_height,
    ), None


def _expectation_met(snapshot: ScreenSnapshot, expect: str) -> bool:
    spec = expect.strip()
    if not spec:
        return False
    kind, sep, value = spec.partition(":")
    if not sep:
        kind, value = "text", spec
    needle = value.strip().casefold()
    if not needle:
        return False
    if kind.casefold() == "app":
        return needle in snapshot.app.casefold()
    if kind.casefold() == "window":
        return needle in snapshot.window.casefold()
    return any(needle in element.text.casefold() for element in snapshot.elements)


def _element_problem(observation_id: str, element_id: str, current: ScreenSnapshot) -> tuple[ScreenElement | None, str | None]:
    latest = _LATEST_OBSERVATION[0]
    if latest is None:
        return None, _error("no retained observation — call see first")
    if not observation_id:
        return None, _error("element_id requires observation_id from the latest [OBS ...] result")
    if observation_id != latest.observation_id:
        return None, _error(
            f"stale observation_id: expected {latest.observation_id}, got {observation_id} — call see and choose a current element"
        )
    element = latest.get(element_id)
    if element is None:
        return None, _error(f"element not found in {observation_id}: {element_id}")
    if current.app != latest.app or current.fingerprint != latest.fingerprint:
        _publish_snapshot(current)
        return None, _error(
            f"screen changed since {observation_id}; latest is {current.observation_id}\n"
            + format_snapshot(current, max_chars=650)
        )
    return element, None


def _crop_element(snapshot: ScreenSnapshot, element: ScreenElement) -> str:
    """Crop an element plus context from the retained full-resolution capture."""
    import cv2

    frame = cv2.imread(snapshot.image_path)
    if frame is None:
        raise ValueError("retained screenshot is unreadable")
    capture_width, capture_height = snapshot.capture_size
    screen_width, screen_height = snapshot.screen_size
    x, y, w, h = element.bounds
    scale_x = capture_width / screen_width
    scale_y = capture_height / screen_height
    pad_x = max(w * 0.75, screen_width * 0.04)
    pad_y = max(h * 1.25, screen_height * 0.04)
    x0 = max(0, round((x - pad_x) * scale_x))
    y0 = max(0, round((y - pad_y) * scale_y))
    x1 = min(capture_width, round((x + w + pad_x) * scale_x))
    y1 = min(capture_height, round((y + h + pad_y) * scale_y))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("element crop is empty")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        crop_path = handle.name
    if not cv2.imwrite(crop_path, frame[y0:y1, x0:x1]):
        Path(crop_path).unlink(missing_ok=True)
        raise ValueError("could not write element crop")
    return crop_path


def _inspect_observation(observation_id: str, element_id: str, question: str) -> str:
    latest = _LATEST_OBSERVATION[0]
    if latest is None:
        return _error("no retained observation — call see first")
    if observation_id != latest.observation_id:
        return _error(f"stale observation_id: expected {latest.observation_id}, got {observation_id}")
    element = latest.get(element_id) if element_id else None
    if element_id and element is None:
        return _error(f"element not found in {observation_id}: {element_id}")
    source_path = latest.image_path
    crop_path = ""
    try:
        if element is not None:
            crop_path = _crop_element(latest, element)
            source_path = crop_path
        from .read_image import read_image
        reader = getattr(read_image, "func", read_image)
        _progress("encoding image…")
        prompt = question.strip() or (
            f"อธิบาย element {element_id} นี้อย่างเจาะจง: เป็นอะไร มีข้อความหรือสถานะอะไร และกดได้หรือไม่"
            if element is not None else "อธิบายหน้าจอนี้โดยเน้น dialog และสิ่งที่กดได้"
        )
        result = reader(source_path, question=prompt)
        target = f" element={element_id}" if element_id else ""
        return f"[ok] inspect observation={observation_id}{target}\n{result}"
    except Exception as exc:
        return _error(f"inspect failed: {exc}")
    finally:
        if crop_path:
            Path(crop_path).unlink(missing_ok=True)


def _post_action_snapshot(before: ScreenSnapshot) -> tuple[ScreenSnapshot, str]:
    """Poll semantic state until it changes and stabilizes, or the timeout expires."""
    _progress("กำลังตรวจสอบผลลัพธ์บนหน้าจอ…")
    deadline = time.monotonic() + max(0.0, _POST_ACTION_TIMEOUT)
    chosen: ScreenSnapshot | None = None
    previous_fingerprint = ""
    changed = False
    stable_hits = 0
    while True:
        current = _capture_snapshot(describe=False, notify=False)
        changed = changed or current.fingerprint != before.fingerprint
        stable_hits = stable_hits + 1 if previous_fingerprint and current.fingerprint == previous_fingerprint else 1
        previous_fingerprint = current.fingerprint
        if chosen is not None:
            _discard_snapshot(chosen)
        chosen = current
        if (changed and stable_hits >= 2) or time.monotonic() >= deadline:
            break
        time.sleep(max(0.0, _POST_ACTION_POLL))
    assert chosen is not None
    try:
        chosen.description = _describe_screen(chosen.image_path)
    except Exception:
        chosen.description = ""
    return _publish_snapshot(chosen), ("changed" if changed else "no_visible_change")


# Split per ENDMEMEX KNOW-LITE-DOCSTRING-SPLIT (2026-07-26, user request): _SYNTAX_MANUAL now
# carries the detailed action-vocabulary/WORKFLOW content that used to sit in the always-visible
# docstring (see git history pre-34c8a99) — injected into the tool's own return value on the
# first call each turn (tools/_call_guard.py), so it costs context only in turns that actually
# use `computer`, never the always-sent schema.
_SYNTAX_MANUAL = (
    "\n\n[computer usage notes]\n"
    "WORKFLOW:\n"
    "1. Start with action=\"see\" when state is unknown. It takes no action and returns [OBS obs_N] "
    "with current eN elements. The view covers only visible content; scroll to reveal more.\n"
    "2. For click/double_click/triple_click/right_click/hover, scroll, or drag, prefer element_id=\"eN\" "
    "plus the latest observation_id=\"obs_N\". Fallback: target=\"visible text\"; add near=\"top|bottom|"
    "left|right|center|top-left|top-right|bottom-left|bottom-right\" only to disambiguate. coord is "
    "disabled. modifiers=\"shift\"|\"cmd\"|\"shift+cmd\"|\"option\"|\"ctrl\" (any subset, \"+\"-joined) held "
    "during click/double_click/triple_click/right_click/drag — e.g. shift-click or cmd-click to "
    "multi-select in a list/Finder, or option-drag to copy a file instead of moving it. triple_click "
    "selects a whole line/paragraph in most text views. hover moves the pointer without clicking — "
    "use it to reveal a tooltip or per-row action button, then see/inspect to read what appeared (OCR "
    "picks up newly-visible text same as anything else on screen). For an unclear/icon-only element, "
    "use action=\"inspect\" with the latest observation_id, optional element_id, and question; it reads "
    "the retained screenshot/crop.\n"
    "3. Other actions: type uses text; key uses text=\"cmd+s|enter|escape|tab|cmd+m (minimize)|cmd+w "
    "(close window)|cmd+ctrl+f (toggle fullscreen, app-dependent)|...\"; scroll uses direction="
    "\"up|down|left|right\" and amount (optional target/element_id chooses the pane; left/right for a "
    "wide spreadsheet/Finder column view/timeline); open_app uses target=\"TextEdit|Finder|...\" and "
    "also raises an already-running app — effect=app_not_frontmost means it did not, retry open_app "
    "before typing/clicking; drag uses source target/element_id and destination visible text in text. "
    "No dedicated maximize/resize action — macOS has no universal maximize hotkey; drag a window's edge "
    "via element/target only works if that edge happens to be a labeled, OCR/AX-visible element. "
    "Reading real on-screen text: OCR/[OBS] is a summary and can misread or truncate — for exact, "
    "full-length text (a document, a long error message) prefer key text=\"cmd+a\" (or select the "
    "specific text) then text=\"cmd+c\", then bash(pbpaste) to read back the real content with no OCR "
    "error and no length cap. The same pair in reverse (bash(pbcopy) then key text=\"cmd+v\") is also "
    "the fastest, IME-safe way to type long or Thai text instead of action=\"type\". A save/open dialog's "
    "path field is reached with key text=\"cmd+shift+g\" (Go to Folder) rather than clicking through "
    "Finder columns.\n"
    "4. Verify every mutation from its new [OBS]. Optional expect=\"text:X|window:X|app:X\" returns "
    "verified or expectation_not_met; otherwise effect is changed or no_visible_change. A `type` whose "
    "typed text does not show up in the now-focused field (AX-confirmed) instead of the intended one "
    "returns effect=type_unconfirmed — treat this the same as no_visible_change, not [ok].\n"
    "Auto-refused: raw coordinates, password-like typing/secure fields, cmd+q/cmd+shift+q, stale "
    "observations, concurrent actions, failsafe mouse corner, and >15 actions (background-fired turns "
    "get a lower, fork-set ceiling and always refuse a delete/remove-looking target — see "
    "effect=app_not_frontmost note above for the retry pattern that applies there too). Prefer open_app "
    "over cmd+tab.\n"
    "\"กดปุ่ม popup/dialog ให้หน่อย\": action=\"see\" first if unsure it's still on screen → pick the button by "
    "its exact visible text (target=\"Allow\"/\"OK\"/\"Don't Save\"/\"Cancel\", or element_id+observation_id from "
    "the latest [OBS]) → click. Never guess a button exists from the task description alone; if [OBS] shows "
    "no matching text, say so instead of clicking the nearest thing.\n"
    "❌ target=\"Yes\" when [OBS] only shows \"OK\"/\"Cancel\" — clicks the wrong control\n"
    "✅ read the actual [OBS] text first, click the button that is really there\n"
    "[OBS ... ax=X] header: ax=ok means Accessibility data is live; ax=permission_required means macOS "
    "Accessibility access is OFF for this process — OCR-based actions still work, but focused-field "
    "read-back, secure-field detection, and icon-only element roles are degraded. Tell the user once "
    "(System Settings → Privacy & Security → Accessibility) instead of repeating it every turn."
)


@tool
def computer(action: str, target: str = "", text: str = "", coord: str = "", direction: str = "", amount: int = 3, near: str = "", element_id: str = "", observation_id: str = "", question: str = "", expect: str = "", modifiers: str = "") -> str:
    """Control a native Mac app one guarded action at a time. The main model is text-only:
    trust the returned [OBS] Accessibility/OCR elements and optional [vision] gist; never infer pixels.

    Start with action="see" when state is unknown — returns [OBS obs_N] with current eN elements
    to act on (scroll to reveal more). Prefer element_id="eN" + observation_id="obs_N" over target
    text for click/double_click/triple_click/right_click/hover/scroll/drag. coord is disabled.
    Verify every mutation from the new [OBS] the action returns.

    Silent-failure recovery: click → no_visible_change → detect failure from effect/[OBS] → see or
    inspect → retry a different current element/target. Never repeat the unchanged action or claim
    success. A delete/remove-looking click/drag target or permanently-remove hotkey is refused
    UNLESS the user's own current message explicitly asked for deletion/removal.

    Detailed per-action parameter guidance (key combos, drag, scroll, modifiers, disambiguation,
    popup handling, [OBS] ax= status) is not yet written — see this tool's own result for now.
    """
    from tools._call_guard import first_call_this_turn
    first = first_call_this_turn("computer")
    if not _ACTION_LOCK.acquire(blocking=False):
        return _error("another computer action is already running — retry after it finishes")
    try:
        result = _computer_impl(
            action=action, target=target, text=text, coord=coord, direction=direction,
            amount=amount, near=near, element_id=element_id,
            observation_id=observation_id, question=question, expect=expect, modifiers=modifiers,
        )
    finally:
        _ACTION_LOCK.release()
    if first and _SYNTAX_MANUAL:
        result += _SYNTAX_MANUAL
    return result


def _computer_impl(*, action: str, target: str, text: str, coord: str, direction: str,
                   amount: int, near: str, element_id: str, observation_id: str,
                   question: str, expect: str, modifiers: str = "") -> str:
    before: ScreenSnapshot | None = None
    try:
        action = action.strip().lower()
        _phase(f"🖥 ควบคุมหน้าจอ: {action}" + (f" ({target or text or element_id})"[:60] if (target or text or element_id) else ""))
        allowed = {"see", "inspect", "click", "double_click", "triple_click", "right_click", "type", "key", "scroll", "open_app", "drag", "hover"}
        if action not in allowed:
            return _error(f"unknown action: {action}")
        if action == "inspect":
            return _inspect_observation(observation_id, element_id, question)
        if _backend.failsafe_abort():
            return _error("failsafe abort — user took the mouse")
        _effective_max = _ACTION_MAX_OVERRIDE[0] or _ACTION_MAX
        if _ACTION_ATTEMPTS[0] >= _effective_max:
            return _error("computer action limit reached — stop and report the current screen")
        if action == "see":
            _ACTION_ATTEMPTS[0] += 1
            _LAST_SIGNATURE[0] = "see"
            snapshot = _publish_snapshot(_capture_snapshot(describe=True))
            _LAST_SCREEN_TEXT[0] = _snapshot_text(snapshot)
            _LAST_FRONTMOST_APP[0] = snapshot.app
            return _format_result("[ok] see", snapshot)

        if coord:
            return _error("raw coord is disabled; use element_id from [OBS] or visible target text")

        # One pre-action snapshot is shared by stale-state validation, target
        # resolution, repeat detection, secure-field checks and effect comparison.
        before = _capture_snapshot(describe=False)
        if action in _CONTEXT_DEPENDENT_ACTIONS and _LAST_FRONTMOST_APP[0] and before.app != _LAST_FRONTMOST_APP[0]:
            _publish_snapshot(before)
            before = None
            latest = _LATEST_OBSERVATION[0]
            return _error(
                f"frontmost app changed since last action (expected '{_LAST_FRONTMOST_APP[0]}', "
                f"now '{latest.app if latest else ''}') — inspect the latest observation before acting\n"
                + (format_snapshot(latest, max_chars=620) if latest else "")
            )

        # near/modifiers included (audit F2): without them, a click that hit an
        # ambiguity error recommending "retry with near=" was itself refused as
        # a "repeated action" on retry (identical action/target/text otherwise),
        # and a plain click followed by the SAME click again with modifiers=
        # (e.g. cmd-click to extend a selection) was wrongly treated as a no-op
        # repeat of the first click.
        signature = "|".join((
            action, target.strip().casefold(), text.strip().casefold(), direction,
            str(amount), element_id.strip().casefold(), observation_id.strip().casefold(),
            near.strip().casefold(), modifiers.strip().casefold(),
        ))
        if signature == _LAST_SIGNATURE[0] and before.fingerprint == (
            _LATEST_OBSERVATION[0].fingerprint if _LATEST_OBSERVATION[0] else ""
        ):
            return _error("repeated action, screen unchanged — change approach")

        chosen_element: ScreenElement | None = None
        if element_id:
            chosen_element, problem = _element_problem(observation_id, element_id, before)
            if problem:
                if _LATEST_OBSERVATION[0] is before:
                    before = None
                return problem

        secure_focused = any(
            element.focused and element.role == "AXSecureTextField" for element in before.elements
        )
        if action == "type" and (
            any(marker in text.casefold() for marker in _PASSWORD_MARKERS)
            or any(marker in _LAST_CLICK_TARGET[0].casefold() for marker in _PASSWORD_MARKERS)
            or secure_focused
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
            # awake-fired turn always set action_max, an ordinary turn's
            # default guard never does, so that existing distinction alone
            # picks the right tier without a new per-turn flag.
            _markers = _DESTRUCTIVE_MARKERS_UNSUPERVISED if _ACTION_MAX_OVERRIDE[0] is not None else _DESTRUCTIVE_MARKERS_SUPERVISED
            destructive_hit = (
                _normalize_combo(text) in _DESTRUCTIVE_KEY_COMBOS if action == "key" else
                any(
                    marker in str(c).casefold()
                    for c in (target, chosen_element.text if chosen_element is not None else None,
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

        if action in {"click", "double_click", "triple_click", "right_click"}:
            if chosen_element is not None:
                point, label = chosen_element.point, chosen_element.text
            else:
                point, problem = _target_point_in_snapshot(before, target, near)
                if problem:
                    return problem
                label = target
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
            _log_action(action, label, (x, y))
            _LAST_CLICK_TARGET[0] = label
        elif action == "hover":
            if chosen_element is not None:
                point, label = chosen_element.point, chosen_element.text
            else:
                point, problem = _target_point_in_snapshot(before, target, near)
                if problem:
                    return problem
                label = target
            _backend.hover(*point)
            _log_action(action, label, point)
        elif action == "type":
            if not text:
                return _error("type requires text")
            _backend.type_text(text)
            _log_action(action)
        elif action == "key":
            _backend.key(text)
            _log_action(action, text)
        elif action == "scroll":
            point = chosen_element.point if chosen_element is not None else None
            if point is None and target:
                point, problem = _target_point_in_snapshot(before, target, near)
                if problem:
                    return problem
            _backend.scroll(direction, amount, point=point)
            _log_action(action, direction)
        elif action == "drag":
            if not text:
                return _error("drag requires text as the drop-target's visible text")
            if chosen_element is not None:
                from_point = chosen_element.point
            else:
                from_point, problem = _target_point_in_snapshot(before, target, near)
                if problem:
                    return problem
            to_point, problem = _target_point_in_snapshot(before, text, "")
            if problem:
                return problem
            try:
                _backend.drag(from_point[0], from_point[1], to_point[0], to_point[1], modifiers=modifiers)
            except ValueError as exc:
                return _error(str(exc))
            _log_action(action, f"{chosen_element.text if chosen_element else target} -> {text}")
        else:
            _backend.open_app(target)
            _log_action(action, target)

        _backend.settle()
        snapshot, effect = _post_action_snapshot(before)
        _LAST_SCREEN_TEXT[0] = _snapshot_text(snapshot)
        _LAST_FRONTMOST_APP[0] = snapshot.app
        if action == "open_app" and target.strip():
            # `open -a` can return 0 without the target ever actually becoming
            # frontmost — e.g. a dormant app reactivating slower than settle()+
            # the post-action poll window under this Mac's single-GPU memory
            # pressure, or a zero-window instance not stealing focus at all.
            # Reporting [ok]/effect=changed here (fingerprint-only check) would
            # let the model believe subsequent type/key calls land in the target
            # app when they're still landing wherever WAS frontmost — a silent-
            # failure class worse than an explicit error (live-reproduced
            # 2026-07-21 walk R7-CB: 8 actions in a row typed into the terminal
            # app hosting this session while it reported [ok] the whole time).
            # Checked BOTH directions (audit F5): target="Visual Studio Code"
            # vs. the real localized app name "Code" (or vice versa, target=
            # "Chrome" vs. "Google Chrome") are legitimate matches either way —
            # only flag when NEITHER string contains the other.
            _t, _a = target.strip().casefold(), snapshot.app.casefold()
            if _t not in _a and _a not in _t:
                effect = "app_not_frontmost"
        if action == "type" and text.strip():
            # Best-effort AX read-back: a generic screen-fingerprint change (the
            # base `effect` above) proves SOMETHING changed, not that the typed
            # text landed in the field it was meant for — Thai IME composition,
            # autocomplete, or focus drift mid-type can all silently produce a
            # changed screen with garbled/partial content. Skip entirely when no
            # AX-focused text element is available (e.g. Accessibility permission
            # off, ax=permission_required) — no regression, same as [vision]'s
            # best-effort pattern elsewhere in this file, just no NEW check.
            typed_cf = text.strip().casefold()
            focused = next((e for e in snapshot.elements if e.focused and e.text), None)
            if focused is not None and typed_cf not in focused.text.casefold():
                # A focused-element mismatch alone isn't proof of failure (audit
                # F3): ScreenElement.text prefers AX's "name" over "value"
                # (_screen_state.py), so a field with a static accessibility
                # name/placeholder ("Search", "Username") compares the typed
                # text against that LABEL, not the field's live content — a
                # guaranteed false mismatch on any named field. Cross-check the
                # OCR-visible screen text (what's actually rendered, independent
                # of which AX attribute won) before flagging; only report
                # type_unconfirmed when the typed text shows up NOWHERE on screen.
                if typed_cf not in _snapshot_text(snapshot).casefold():
                    effect = "type_unconfirmed"
        if expect:
            effect = "verified" if _expectation_met(snapshot, expect) else "expectation_not_met"
        prefix = f"[ok] {action}" if effect in {"changed", "verified"} else (
            f"[warning] {action}" if effect in {"expectation_not_met", "app_not_frontmost", "type_unconfirmed"} else f"[no_effect] {action}"
        )
        return _format_result(prefix, snapshot, effect)
    except Exception as exc:
        return _error(str(exc))
    finally:
        if before is not None and _LATEST_OBSERVATION[0] is not before:
            _discard_snapshot(before)
