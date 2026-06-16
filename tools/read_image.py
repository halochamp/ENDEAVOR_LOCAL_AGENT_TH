# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""read_image.py — OCR-only image reading for ENDEAVOR_LOCAL_AGENT_TH

Apple Vision OCR (full-res, Thai+English) extracts text from the image.
No VLM/vision-server dependency — only the LLM needs to be running.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
import urllib.request
import uuid
from pathlib import Path

from langchain_core.tools import tool

from tools._ocr import read_text as _ocr_read
from tools._progress import progress, phase
from tools._safety import resolve_read_path

_DEEP_MAX_CHARS = 4000


def _hard_truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n - 3].rstrip() + "..."


def _download_url(url: str) -> str:
    """Download URL to a temp file, return local path. Raises on failure."""
    progress(f"downloading: {url[:70]}")
    suffix = ".jpg" if any(url.lower().endswith(e) for e in (".jpg", ".jpeg")) else ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            Path(tmp.name).write_bytes(resp.read())
    except Exception:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    return tmp.name


@tool
def read_image(source: str) -> str:
    """Extract text from an image via Apple Vision OCR (full-res, Thai+English).

    source : file path in workspace / absolute path | https:// URL | "screen" (screenshot)

    Returns "[OCR]\\n<text>" when text is found, or "[OCR] no text detected"
    when the image has no readable text (photos, diagrams, illustrations —
    this tool does not describe non-text image content).
    Returns [error] prefix on failure.
    """
    if not source or not source.strip():
        return "[error] source is required"

    phase(f"🖼 อ่านภาพ: {source[:50]}")
    downloaded_tmp: str | None = None

    try:
        src = source.strip()

        # ── resolve source ────────────────────────────────────────────────
        if src.lower() == "screen":
            tmp_screen = f"/tmp/endeavor_screen_{uuid.uuid4().hex[:8]}.png"
            progress("กำลัง screenshot…")
            result = subprocess.run(
                ["screencapture", "-x", "-m", tmp_screen],
                capture_output=True, timeout=5,
            )
            if result.returncode != 0 or not Path(tmp_screen).exists():
                return (
                    "[error] screencapture failed — กรุณาเปิดสิทธิ์ Screen Recording:\n"
                    "System Settings → Privacy & Security → Screen Recording → เปิดให้ Terminal/Claude Code"
                )
            local_path = tmp_screen
            downloaded_tmp = tmp_screen

        elif src.startswith("http://") or src.startswith("https://"):
            local_path = _download_url(src)
            downloaded_tmp = local_path

        else:
            local_path = resolve_read_path(src)
            if not Path(local_path).exists():
                return f"[error] file not found: {src}"

        # ── OCR (full-res, never raises) ──────────────────────────────────
        progress("Apple Vision OCR…")
        ocr_lines = _ocr_read(local_path)
        ocr_text  = "\n".join(ocr_lines)

        if ocr_text:
            progress(f"พบข้อความ {len(ocr_lines)} บรรทัด")
            return _hard_truncate(f"[OCR]\n{ocr_text}", _DEEP_MAX_CHARS)

        return "[OCR] no text detected"

    except PermissionError as e:
        return f"[error] read_image: {e}"
    except Exception as e:
        return f"[error] read_image: {e}"
    finally:
        if downloaded_tmp:
            try:
                os.unlink(downloaded_tmp)
            except Exception:
                pass
