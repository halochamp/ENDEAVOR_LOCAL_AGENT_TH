# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""_ocr.py — the one OCR entry point of the agent: Apple Vision (default) or OpenThai, chosen in Settings.

Every OCR caller (read_image, read_pdf, read_file, pdf_documents, pdf_to_text, computer_use, awake_engine) goes
through read_layout()/read_text() here, so the engine owner picked in Settings -> OCR (user_settings
"ocr_engine": "apple" | "openthai") applies to all of them at once, including PDF -> Markdown.  The choice is
read on every call, so a change takes effect on the next OCR without a restart.

Apple Vision: adapted from ENDEAVOR_VISSION/perception/ocr_reader.py. Accepts a file path directly, no cv2 needed —
the Swift binary reads the file itself (NSImage supports JPEG/PNG/HEIC natively), so no intermediate PNG write.
OpenThai: tools/_ocr_openthai.py (CPU, ONNX).  If it is selected but cannot process a given image (weights or
packages missing, an undecodable file), that call falls back to Apple Vision with a warning in the log.

Returns list[dict] / list[str] (one per text line), [] on any failure — never raises.
"""
from __future__ import annotations
import hashlib
import logging
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from . import _ocr_openthai as _openthai

log = logging.getLogger(__name__)

ENGINE_LABELS = {"apple": "Apple Vision", "openthai": "OpenThai"}
_fallback_warned = False

_SRC = Path(__file__).resolve().parent / "_vision_ocr.swift"


def _binary_path(source: Path = _SRC) -> Path:
    """Compiled helper location, named after a hash of its Swift source.  A fixed name let two code versions share one
    binary: when the source dropped --fast, the still-running older agent kept sending it to the new build and read
    nothing, silently.  With the hash in the name each version compiles and uses only its own binary."""
    try:
        tag = hashlib.sha256(source.read_bytes()).hexdigest()[:10]
    except OSError:
        tag = "nosource"
    return Path(tempfile.gettempdir()) / f"endeavor_v2_vision_ocr-{tag}"


_BIN = _binary_path()
_compiled: bool | None = None   # None=unknown, True=ready, False=unavailable (this call only)
_compile_lock = threading.Lock()


def _ensure_binary() -> bool:
    """Compile-on-demand, guarded so concurrent first calls don't race on the same
    swiftc output, and so a transient failure doesn't permanently disable OCR for
    the rest of the process lifetime — only success is cached (audit.md #9)."""
    global _compiled
    if _compiled:
        return _compiled
    with _compile_lock:
        if _compiled:  # another thread finished compiling while we waited
            return _compiled
        try:
            fresh = _BIN.exists() and _BIN.stat().st_mtime >= _SRC.stat().st_mtime
            if not fresh:
                subprocess.run(
                    ["swiftc", "-O", str(_SRC), "-o", str(_BIN)],
                    check=True, capture_output=True, timeout=120,
                    stdin=subprocess.DEVNULL,
                )
            _compiled = _BIN.exists()
        except Exception as e:
            log.warning(f"[ocr] swift helper unavailable ({e}) — OCR disabled for this call, will retry next call")
            _compiled = False
    return _compiled


def _selected_engine() -> str:
    """The engine chosen in Settings; "apple" when unset, unreadable or unknown (the safe default)."""
    try:
        import user_settings
        return user_settings.ocr_engine()
    except Exception:
        return "apple"


def _effective_engine() -> str:
    """The engine a call would really use right now: OpenThai only while it is selected AND ready."""
    if _selected_engine() == "openthai" and _openthai.availability()[0]:
        return "openthai"
    return "apple"


def engine_label() -> str:
    """Human name of the engine in effect ("Apple Vision" / "OpenThai"), for tool output and progress text."""
    return ENGINE_LABELS[_effective_engine()]


def ocr_ready() -> bool:
    """True when OCR can run with the engine in effect.  read_pdf's gate before it OCRs scanned pages."""
    if _effective_engine() == "openthai":
        return True
    return _ensure_binary()


def engine_status() -> dict:
    """{"selected", "effective", "engines": {name: {"available", "detail"}}} for Settings.  Cheap: loads nothing."""
    ready, reason = _openthai.availability()
    apple_ready = _BIN.exists() or shutil.which("swiftc") is not None
    return {
        "selected": _selected_engine(),
        "effective": _effective_engine(),
        "engines": {
            "apple": {"available": apple_ready, "detail": "" if apple_ready else "swiftc not found and no compiled helper"},
            "openthai": {"available": ready, "detail": reason},
        },
    }


def read_layout(image_path: str) -> list[dict]:
    """Run the selected OCR engine, returning one dict per recognized text line:
    {"text": str, "x": float, "y": float, "w": float, "h": float}.
    Coordinates are the Vision boundingBox — normalized [0,1], origin BOTTOM-LEFT
    (y increases upward); OpenThai results are converted to the same convention.
    [] on any failure, never raises. Used to reconstruct
    table layout from cell positions; read_text() is the plain-text view of this.
    There is one recognition mode on purpose: Vision's .fast level cannot read Thai.
    """
    global _fallback_warned
    if image_path and _selected_engine() == "openthai":
        boxes = _openthai.read_layout(image_path)
        if boxes is not None:
            return boxes
        if not _fallback_warned:
            _fallback_warned = True
            log.warning("[ocr] OpenThai is selected but could not process this call (%s); using Apple Vision. "
                        "Further fallbacks are not logged again until restart.", _openthai.availability()[1] or "see earlier warning")
    return _read_layout_apple(image_path)


def _read_layout_apple(image_path: str) -> list[dict]:
    """The Apple Vision helper: compile-on-demand Swift binary, tab-separated boxes on stdout."""
    if not image_path or not _ensure_binary():
        return []
    try:
        command = [str(_BIN), image_path]
        out = subprocess.run(
            command,
            capture_output=True, text=True, timeout=15,
            stdin=subprocess.DEVNULL,
        )
        raw = (out.stdout or "").strip()
        if not raw or raw.startswith("ERR"):
            return []
        boxes: list[dict] = []
        for line in raw.split("\n"):
            parts = line.split("\t", 4)         # text is last → tabs in text are safe
            if len(parts) != 5:
                continue
            try:
                x, y, w, h = (float(parts[i]) for i in range(4))
            except ValueError:
                continue
            text = parts[4].strip()
            if text:
                boxes.append({"text": text, "x": x, "y": y, "w": w, "h": h})
        return boxes
    except Exception as e:
        log.debug(f"[ocr] read error: {e}")
        return []


def read_text(image_path: str) -> list[str]:
    """Run Apple Vision OCR on image_path. [] on any failure, never raises.
    Plain-text view (reading order) of read_layout()."""
    return [b["text"] for b in read_layout(image_path)]
