# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""_doc_extract.py — PDF/Word/Excel → markdown via MarkItDown (Microsoft)

Lazy-imports MarkItDown so the dependency is only loaded when a document is
actually read. One module-level singleton is reused across calls.

Returns markdown str, or "[error] ..." on failure — never raises.
Install: pip install 'markitdown[pdf,docx,xlsx,xls]'
"""
from __future__ import annotations
import subprocess

_MD = None  # MarkItDown singleton, lazily initialised


def _get_md():
    global _MD
    if _MD is None:
        from markitdown import MarkItDown
        _MD = MarkItDown()
    return _MD


def to_markdown(path: str) -> str:
    """Convert a PDF/DOC/DOCX/XLSX/XLS file to markdown text.

    Returns the markdown string (may be empty for scanned/image-only PDFs),
    or "[error] ..." when MarkItDown is missing or conversion fails.
    """
    if path.lower().endswith(".doc"):
        return _legacy_doc_to_text(path)
    try:
        md = _get_md()
    except ImportError:
        return "[error] markitdown not installed — run: pip install 'markitdown[pdf,docx,xlsx,xls]'"
    try:
        result = md.convert(path)
        return result.text_content or ""
    except Exception as e:
        return f"[error] document parse failed: {e}"


def _legacy_doc_to_text(path: str) -> str:
    """Extract old binary Word .doc files with the built-in macOS converter."""
    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", "-encoding", "UTF-8", path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as e:
        return f"[error] legacy .doc extraction failed: {e}"
    if result.returncode != 0:
        detail = " ".join(result.stderr.split())[:240]
        return f"[error] legacy .doc extraction failed{': ' + detail if detail else ''}"
    if not result.stdout.strip():
        return "[error] legacy .doc has no extractable text"
    return result.stdout
