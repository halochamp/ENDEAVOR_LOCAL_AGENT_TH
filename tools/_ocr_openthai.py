# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""_ocr_openthai.py — OpenThai2.1-OCR-Text (iApp, Apache-2.0) behind the same box contract as _ocr.read_layout.

The optional second OCR engine of Settings -> OCR (user_settings "ocr_engine").  A DB text-line detector and a
CTC line recognizer on ONNX Runtime, CPU only, about 0.25 s per A4 page and ~0.5 GB resident once loaded.  The
runtime code is vendored unmodified in tools/_openthai_ocr/ (SHA-256 pinned by the unit test); the 4 MB weights
live in config.OPENTHAI_OCR_DIR (scripts/install_openthai_ocr.py).

read_layout() returns the same dicts as the Apple Vision helper: {"text","x","y","w","h"}, normalized to [0,1]
with the origin at the BOTTOM-LEFT, one per detected text line in reading order, so table reconstruction, click-
by-text and pdf_to_text work unchanged.  Unlike Apple's, it returns None when the engine cannot process the image
(model files or packages missing, an image OpenCV cannot decode such as HEIC, an inference error): _ocr.py then
falls back to Apple Vision for that call and logs a warning.  [] means the page was read and has no text.
Never raises.  Lines under the model's own confidence floor (config.json: 0.8) are dropped by the engine — that is
how it ignores stamps and handwriting; the Apple helper keeps everything it recognizes.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger(__name__)

_REQUIRED_FILES = ("det.onnx", "rec.onnx", "rec_dict.txt")
_REQUIRED_MODULES = ("cv2", "numpy", "onnxruntime", "pyclipper")

# One lock serialises load and read: the detector/recognizer sessions are shared, a read is CPU-bound anyway, and
# read_image's batch OCR may call from several threads at once.
_lock = threading.Lock()
_engine = None
_engine_dir: str | None = None
_load_error: str | None = None


def model_dir() -> str:
    try:
        import config
        return str(config.OPENTHAI_OCR_DIR)
    except Exception:
        return os.getenv("V2_OPENTHAI_OCR_DIR", os.path.expanduser("~/Models/OpenThai2.1-OCR-Text"))


def availability() -> tuple[bool, str]:
    """(ready, reason).  Cheap (file checks + find_spec): no model is loaded and no heavy package is imported."""
    directory = Path(model_dir())
    missing = [name for name in _REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        return False, (f"model files missing in {directory}: {', '.join(missing)}"
                       " — run: python scripts/install_openthai_ocr.py")
    absent = [name for name in _REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if absent:
        return False, f"Python packages missing: {', '.join(absent)} — pip install {' '.join(absent)}"
    if _load_error and _engine_dir == str(directory):
        return False, f"model failed to load: {_load_error}"
    return True, ""


def _get_engine():
    """The loaded engine, or None.  Caller holds _lock."""
    global _engine, _engine_dir, _load_error
    directory = model_dir()
    if _engine_dir == directory:
        if _engine is not None:
            return _engine
        if _load_error is not None:
            return None  # this directory already failed to load; do not retry on every call
    if not availability()[0]:
        return None
    try:
        import cv2
        from ._openthai_ocr.line_engine import LineOCR

        threads = max(1, min(4, os.cpu_count() or 1))
        previous = cv2.getNumThreads()
        try:
            _engine = LineOCR(directory, threads=threads)
        finally:
            cv2.setNumThreads(previous)  # LineOCR pins OpenCV's pool for the whole process; put it back
        _engine_dir, _load_error = directory, None
    except Exception as exc:
        _engine, _engine_dir, _load_error = None, directory, f"{type(exc).__name__}: {exc}"[:300]
        log.warning("[ocr] OpenThai engine failed to load (%s)", _load_error)
    return _engine


def _boxes_from_lines(lines: list[dict], width: int, height: int) -> list[dict]:
    """Engine lines (4 corner points in image pixels, y down) -> Vision-style boxes (normalized, origin bottom-left)."""
    boxes: list[dict] = []
    for line in lines:
        text = str(line.get("text") or "").strip()
        points = line.get("box") or []
        if not text or not points:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        left, right = max(0.0, min(xs)), min(float(width), max(xs))
        top, bottom = max(0.0, min(ys)), min(float(height), max(ys))
        if right <= left or bottom <= top:
            continue
        boxes.append({
            "text": text,
            "x": left / width,
            "y": 1.0 - bottom / height,
            "w": (right - left) / width,
            "h": (bottom - top) / height,
        })
    return boxes


def read_layout(image_path: str) -> list[dict] | None:
    """Boxes for every recognized line, [] when the page has none, None when this engine cannot process the image."""
    try:
        with _lock:
            engine = _get_engine()
            if engine is None:
                return None
            import cv2

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                log.warning("[ocr] OpenThai could not decode %s; falling back", Path(str(image_path)).name)
                return None
            height, width = image.shape[:2]
            result = engine.read(image)
        return _boxes_from_lines(result.get("lines") or [], width, height)
    except Exception as exc:
        log.warning("[ocr] OpenThai read failed (%s: %s); falling back", type(exc).__name__, str(exc)[:200])
        return None


def reset_for_tests() -> None:
    """Drop the loaded engine and any remembered load error (unit tests only)."""
    global _engine, _engine_dir, _load_error
    with _lock:
        _engine, _engine_dir, _load_error = None, None, None
