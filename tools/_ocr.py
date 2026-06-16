# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""_ocr.py — Apple Vision OCR wrapper (accepts file path directly, no cv2 needed)

Adapted from ENDEAVOR_VISSION/perception/ocr_reader.py. Key difference: accepts a
file path string instead of a cv2 ndarray — the Swift binary reads the file itself
(NSImage supports JPEG/PNG/HEIC natively), so no intermediate PNG write is needed.

Returns list[str] (one per text line), [] on any failure — never raises.
"""
from __future__ import annotations
import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parent / "_vision_ocr.swift"
_BIN = Path(tempfile.gettempdir()) / "endeavor_v2_vision_ocr"
_compiled: bool | None = None   # None=unknown, True=ready, False=unavailable


def _ensure_binary() -> bool:
    global _compiled
    if _compiled is not None:
        return _compiled
    try:
        fresh = _BIN.exists() and _BIN.stat().st_mtime >= _SRC.stat().st_mtime
        if not fresh:
            subprocess.run(
                ["swiftc", "-O", str(_SRC), "-o", str(_BIN)],
                check=True, capture_output=True, timeout=120,
            )
        _compiled = _BIN.exists()
    except Exception as e:
        log.warning(f"[ocr] swift helper unavailable ({e}) — OCR disabled")
        _compiled = False
    return _compiled


def read_text(image_path: str) -> list[str]:
    """Run Apple Vision OCR on image_path. [] on any failure, never raises."""
    if not image_path or not _ensure_binary():
        return []
    try:
        out = subprocess.run(
            [str(_BIN), image_path],
            capture_output=True, text=True, timeout=15,
        )
        raw = (out.stdout or "").strip()
        if not raw or raw.startswith("ERR"):
            return []
        return [s.strip() for s in raw.split(" / ") if s.strip()]
    except Exception as e:
        log.debug(f"[ocr] read error: {e}")
        return []
