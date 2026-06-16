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

_MD = None  # MarkItDown singleton, lazily initialised


def _get_md():
    global _MD
    if _MD is None:
        from markitdown import MarkItDown
        _MD = MarkItDown()
    return _MD


def to_markdown(path: str) -> str:
    """Convert a PDF/DOCX/XLSX/XLS file to markdown text.

    Returns the markdown string (may be empty for scanned/image-only PDFs),
    or "[error] ..." when MarkItDown is missing or conversion fails.
    """
    try:
        md = _get_md()
    except ImportError:
        return "[error] markitdown not installed — run: pip install 'markitdown[pdf,docx,xlsx,xls]'"
    try:
        result = md.convert(path)
        return result.text_content or ""
    except Exception as e:
        return f"[error] document parse failed: {e}"
