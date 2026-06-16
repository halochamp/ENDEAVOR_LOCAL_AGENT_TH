# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import READ_FILE_MAX_CHARS as _MAX_CHARS, READ_FILE_MAX_BYTES as _MAX_FILE_BYTES
_CODE_EXT = {".py", ".js", ".ts", ".go", ".java", ".cpp", ".c", ".rs", ".rb", ".php", ".swift", ".kt"}
_DOC_EXT = {".pdf", ".docx", ".xlsx", ".xls"}


@tool
def read_file(path: str) -> str:
    """Read file contents — plain text, code, and PDF/Word/Excel documents.

    Code files over the size limit return a structure map (symbols + line numbers).
    PDF/DOCX/XLSX/XLS are converted to markdown; large documents are sampled for
    coverage (outline + paragraphs/rows spread across the whole file). Scanned or
    image-only PDFs (no extractable text) return [error] — use read_image instead.
    Files larger than READ_FILE_MAX_BYTES (default 5 MB) are rejected — use grep/bash to target sections.
    """
    if not path:
        return _missing_path_hint()
    try:
        from ._safety import resolve_read_path
        p = Path(resolve_read_path(path))
        if not p.exists():
            return f"[error] file not found: {path}"

        size = p.stat().st_size
        if size > _MAX_FILE_BYTES:
            mb = size / (1024 * 1024)
            limit_mb = _MAX_FILE_BYTES / (1024 * 1024)
            return (f"[error] file too large: {mb:.1f} MB (limit {limit_mb:.0f} MB) — "
                    f"use grep/bash to read specific sections")

        suffix = p.suffix.lower()
        if suffix in _DOC_EXT:
            return _read_document(p, path)

        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) <= _MAX_CHARS:
            return content
        if suffix in _CODE_EXT:
            return _extract_structure(content, path)
        return content[:_MAX_CHARS] + f"\n...(truncated — {len(content)} chars total, use grep/bash for specific sections)"
    except Exception as e:
        return f"[error] read_file failed: {e}"


def _missing_path_hint() -> str:
    """No path given — list candidate documents in the workspace so the agent
    can ask the user which one, instead of guessing or running find/grep."""
    try:
        from config import WORKSPACE
        candidates = []
        for dirpath, dirnames, filenames in os.walk(WORKSPACE):
            dirnames[:] = [d for d in dirnames if not d.startswith("._exec_")]
            for name in filenames:
                if Path(name).suffix.lower() in _DOC_EXT and not name.startswith("._exec_"):
                    rel = os.path.relpath(os.path.join(dirpath, name), WORKSPACE)
                    candidates.append(rel)
        candidates.sort()
    except Exception:
        candidates = []

    if not candidates:
        return ("[error] path is required — no PDF/Word/Excel files found in the workspace. "
                "Ask the user for the file name or path.")
    if len(candidates) == 1:
        return (f"[error] path is required. Found one document in the workspace: {candidates[0]}. "
                f"Ask the user to confirm before reading it.")
    listing = "\n".join(f"  - {c}" for c in candidates[:20])
    return (f"[error] path is required. Found {len(candidates)} documents in the workspace:\n{listing}\n"
            f"Ask the user which file they mean — do not guess.")


def _read_document(p: Path, path: str) -> str:
    """Convert a PDF/DOCX/XLSX/XLS to markdown, sampling for coverage if large."""
    from ._doc_extract import to_markdown
    md = to_markdown(str(p))
    if md.startswith("[error]"):
        return md
    if len(md.strip()) < 10:
        return (f"[error] no extractable text in {path} — likely a scanned/image PDF. "
                f"Use read_image for image-based documents.")
    if len(md) <= _MAX_CHARS:
        return md
    return _sample_coverage(md, path)


def _sample_coverage(text: str, path: str, max_chars: int | None = None) -> str:
    """Approach C — outline + uniform sampling across the whole document.

    Keeps whole lines (paragraphs for prose, rows for tables) spread evenly so
    the head, middle, and tail are all represented, with [... skipped N ...] markers.
    max_chars overrides the module-level _MAX_CHARS for callers with a different budget.
    """
    if max_chars is None:
        max_chars = _MAX_CHARS
    _LINE_CAP = 2_000   # hard cap on any single line so one giant unit can't blow the budget
    lines = text.split("\n")
    total_chars = len(text)
    units = [(ln if len(ln) <= _LINE_CAP else ln[:_LINE_CAP] + " …[line truncated]")
             for ln in lines if ln.strip()]
    n = len(units)

    # Pin structural lines that must survive sampling: markdown headings (#…) and
    # table header rows (the line directly above a "| --- |" separator + the
    # separator itself) — without them, sampled table rows are unlabelled.
    pinned: set[int] = set()
    for i, u in enumerate(units):
        s = u.lstrip()
        if s.startswith("#"):
            pinned.add(i)
        elif "---" in s and set(s) <= set("| -:"):
            pinned.add(i)
            if i > 0:
                pinned.add(i - 1)

    headings = [units[i].strip() for i in sorted(pinned)
                if units[i].lstrip().startswith("#")][:40]
    header = [f"[{path} — document, {total_chars} chars total, sampled for coverage]", ""]
    if headings:
        header += ["outline:"] + [f"  {h}" for h in headings] + [""]
    header_str = "\n".join(header)

    budget = max_chars - len(header_str) - 200
    if budget < 500:
        budget = 500

    body_chars = sum(len(u) for u in units)
    if body_chars <= budget:
        return header_str + "\n".join(units)

    # Keep a uniform fraction of lines spread across the whole document. Size the
    # ratio so the entire pass (kept lines + skip markers) fits inside the budget,
    # so we never stop early and the tail is always reached. _MARKER_LEN is the
    # estimated cost of one "[... skipped N lines ...]" line.
    _MARKER_LEN = 28
    ratio = min(1.0, budget / (body_chars + n * _MARKER_LEN))

    out: list[str] = []
    acc = 1.0 - ratio  # ensures the first line is always kept (head coverage)
    skipped = 0
    for i, u in enumerate(units):
        keep = i in pinned
        if not keep:
            acc += ratio
            if acc >= 1.0:
                keep = True
                acc -= 1.0
        if keep:
            if skipped:
                out.append(f"[... skipped {skipped} lines ...]")
                skipped = 0
            out.append(u)
        else:
            skipped += 1
    if skipped:
        # force-include the final line so the document tail is always represented
        if skipped > 1:
            out.append(f"[... skipped {skipped - 1} lines ...]")
        out.append(units[-1])

    result = header_str + "\n".join(out)
    if len(result) > max_chars:        # safety net — never exceed the cap
        result = result[:max_chars] + "\n…[truncated to cap]"
    return result


def _extract_structure(content: str, path: str) -> str:
    lines = content.splitlines()
    symbols: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(def |class |async def |function |const |export const |export function |export class )([\w]+)', line)
        if m:
            sig = line.rstrip()
            j = i + 1
            while j < min(i + 6, len(lines)) and not re.search(r'[):{]', sig):
                sig += ' ' + lines[j].strip()
                j += 1
            sig = re.split(r'\s*[:{]\s*$', sig.strip())[0][:90]
            symbols.append(f"  {sig:<92} :{i + 1}")
        i += 1
    imports = [l for l in lines[:30] if l.startswith(("import ", "from ", "require", "use ", "#include"))]
    out = [f"[{path} — {len(lines)} lines, structure map only]", ""]
    if imports:
        out += ["imports:"] + [f"  {l[:80]}" for l in imports[:10]] + [""]
    out += ["symbols:"] + (symbols if symbols else ["  (none found)"])
    out += ["", "Use grep or bash to read specific sections by line number."]
    return "\n".join(out)
