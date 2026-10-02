# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Literal
from langchain_core.tools import tool
from ._safety import plan_write


def _normalize_lines(text: str) -> str:
    """Strip trailing whitespace per line — fixes whitespace mismatch on old_string."""
    return "\n".join(line.rstrip() for line in text.splitlines())


def _as_bool(v) -> bool:
    """Batch hunks arrive as plain dicts (not pydantic-validated field-by-field), so a
    local model can hand back a JSON string like "false" for a bool field — plain
    bool(v) would treat that non-empty string as truthy. Coerce defensively."""
    if isinstance(v, str):
        return v.strip().lower() not in ("", "false", "0", "no")
    return bool(v)


def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _line_count(s: str) -> int:
    """Logical line count for a report message — a trailing "\\n" terminates the
    last line, it does not start an extra empty one, so plain count('\\n')+1 is
    off by one for any string that ends with a newline."""
    if not s:
        return 0
    n = s.count("\n")
    return n if s.endswith("\n") else n + 1


def _join_with_boundary(before: list[str], middle: str, after: list[str]) -> str:
    """Join keepends-lines + inserted text + more keepends-lines. Ensures a newline
    separates `before`'s last line from `middle`/`after` when it's missing one (the
    file's previous final line had no trailing newline) — otherwise two lines would
    be silently fused together. Also newline-terminates `middle` itself whenever it's
    non-empty, unconditionally (including at true EOF, `after` empty) — a
    replace/insert landing at the end of the file must not silently strip the file's
    trailing newline convention."""
    if before and not before[-1].endswith("\n") and (middle or after):
        before = before[:-1] + [before[-1] + "\n"]
    if middle and not middle.endswith("\n"):
        middle = middle + "\n"
    return "".join(before) + middle + "".join(after)


def _apply_line_hunk(content: str, line_start: int, line_end: int, new_string: str) -> tuple[str | None, str | None, str]:
    """LINE mode: line_end>=line_start replaces that inclusive 1-indexed range
    (empty new_string deletes it); line_end omitted (0) inserts new_string as new
    line(s) immediately after line_start."""
    lines = content.splitlines(True)
    n_lines = len(lines)
    if line_start < 1:
        return None, f"line_start must be >= 1 (got {line_start})", ""
    if line_start > n_lines:
        return None, f"line_start {line_start} is beyond end of file ({n_lines} lines)", ""
    if line_end and line_end < line_start:
        return None, f"line_end ({line_end}) must be >= line_start ({line_start})", ""

    if line_end and line_end > 0:
        end_idx = min(line_end, n_lines)
        before = lines[:line_start - 1]
        after = lines[end_idx:]
        new_content = _join_with_boundary(before, new_string, after)
        if not new_string:
            desc = f"deleted lines {line_start}-{end_idx}"
        else:
            desc = f"replaced lines {line_start}-{end_idx} with {_line_count(new_string)} line(s)"
        return new_content, None, desc

    if not new_string:
        return None, (
            "insert mode (line_end omitted) requires non-empty new_string — "
            "use line_end=line_start with empty new_string to delete a line instead"
        ), ""
    before = lines[:line_start]
    after = lines[line_start:]
    new_content = _join_with_boundary(before, new_string, after)
    desc = f"inserted {_line_count(new_string)} line(s) after line {line_start}"
    return new_content, None, desc


def _apply_hunk(content: str, hunk: dict) -> tuple[str | None, str | None, str]:
    """Apply one hunk to `content` in memory. Returns (new_content, error, description);
    on error new_content is None. The single code path every mode (STRING/LINE/BATCH)
    routes through — one shared place for match/replace logic, not three."""
    line_start = _as_int(hunk.get("line_start") or 0)
    if line_start != 0:
        # Route any explicitly-set line_start (including invalid negatives) into LINE
        # mode so _apply_line_hunk's bounds-check reports it, instead of falling
        # through to STRING mode's "old_string is required" — misleading when the
        # caller did set line_start, just to an invalid value.
        return _apply_line_hunk(content, line_start, _as_int(hunk.get("line_end") or 0), hunk.get("new_string") or "")

    old_string = hunk.get("old_string") or ""
    new_string = hunk.get("new_string") or ""
    replace_all = _as_bool(hunk.get("replace_all", False))
    near_line = _as_int(hunk.get("near_line") or 0)

    if not old_string:
        return None, "old_string is required (or set line_start for LINE mode)", ""

    count = content.count(old_string)
    if count == 0:
        norm_content = _normalize_lines(content)
        norm_old = _normalize_lines(old_string)
        if norm_old in norm_content:
            # Map match back to original lines — avoid writing normalized whole file
            norm_idx = norm_content.index(norm_old)
            start_line = norm_content[:norm_idx].count('\n')
            n_lines = norm_old.count('\n') + 1
            orig_lines = content.splitlines(True)
            orig_chunk = "".join(orig_lines[start_line:start_line + n_lines])
            if orig_chunk in content:
                old_string = orig_chunk
                count = content.count(orig_chunk)
    if count == 0:
        return None, (
            f"old_string not found.\nFile content (first 600 chars):\n{content[:600]}\n"
            "Copy old_string exactly from the content above."
        ), ""
    if count > 1 and not replace_all and near_line > 0:
        offsets = []
        start = 0
        for _ in range(count):
            idx = content.index(old_string, start)
            offsets.append(idx)
            start = idx + len(old_string)
        # read_file/grep number lines via splitlines() (which recognizes form-feed,
        # vertical-tab, NEL, LS/PS as breaks, not just \n); match that here so a
        # near_line the agent read from those tools lands on the same occurrence.
        # The "\x00" sentinel is required so a match sitting immediately after a
        # break is counted on the NEXT line — a bare splitlines() would drop the
        # trailing empty element and under-count by one. (edit reads via read_text()
        # which universal-newline-normalizes \r/\r\n→\n, so content never holds a
        # lone \r; the sentinel only reconciles \n + the exotic breaks.)
        lines = [len((content[:idx] + "\x00").splitlines()) for idx in offsets]
        best_i = min(range(len(lines)), key=lambda i: (abs(lines[i] - near_line), i))
        best_offset = offsets[best_i]
        actual_line = lines[best_i]
        new_content = content[:best_offset] + new_string + content[best_offset + len(old_string):]
        desc = f"replaced 1 occurrence near line {actual_line} (of {count} matches, requested near_line={near_line})"
        return new_content, None, desc
    if count > 1 and not replace_all:
        return None, (
            f"old_string found {count} times — use replace_all=true, "
            "make it more unique, or pass near_line=<line number> to target the closest match"
        ), ""
    new_content = (
        content.replace(old_string, new_string)
        if replace_all else content.replace(old_string, new_string, 1)
    )
    n = count if replace_all else 1
    desc = f"replaced {n} occurrence(s)"
    return new_content, None, desc


def _py_syntax_check(p: Path) -> str:
    """Inline syntax check after a successful .py edit — catches a syntax error in the
    same turn instead of costing a separate bash round-trip. Uses the builtin compile()
    (in-memory only, no bytecode file written). Never masks a successful edit: a
    checker failure of any other kind returns "" (silent), the write itself already
    succeeded."""
    try:
        source = p.read_text(encoding="utf-8")
        compile(source, str(p), "exec")
        return "\n✓ syntax OK"
    except SyntaxError as e:
        where = f" (line {e.lineno})" if e.lineno else ""
        return f"\n⚠ SYNTAX ERROR{where}: {e.msg}"
    except Exception:
        return ""


def _atomic_replace(path: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.chmod(temporary, path.stat().st_mode & 0o777)
        except OSError:
            pass
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_create(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # link() is atomic and fails if a file or symlink raced into place; unlike
        # replace(), it can never silently overwrite an existing create target.
        os.link(temporary, path, follow_symlinks=False)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@tool
def edit(path: str, mode: Literal["", "create", "replace", "line", "batch"] = "",
         content: str = "", old_string: str = "", new_string: str = "",
         replace_all: bool = False, near_line: int = 0, line_start: int = 0,
         line_end: int = 0, edits: list | str | None = None) -> str:
    """Create or atomically modify a file in the internal or explicitly enabled workspace.

    MODES:
    CREATE: mode="create" with content; creates a new target and refuses to overwrite.
    REPLACE: mode="replace" with content; explicitly replaces an existing whole file.
    STRING (default): old_string→new_string; old_string must be a unique substring of
    the file (or replace_all=true). near_line disambiguates when old_string matches >1 place.
    LINE: mode="line" and line_start (1-indexed). line_end>=line_start replaces that inclusive range
    (empty new_string deletes it); line_end omitted inserts new_string as new line(s)
    after line_start (new_string required for insert).
    BATCH: mode="batch" with edits=[{...}, ...] — each item uses the same fields as above (old_string
    OR line_start, not both). Applied in order to ONE file, atomically: any hunk failure
    discards the whole batch, nothing is written. A later hunk's line numbers are resolved
    against the file as already changed by earlier hunks in the SAME batch — order
    line-based hunks bottom-to-top (highest line_start first) if a batch mixes them.
    Read an existing target with read_file first. Relative paths use Active Workspace if set;
    otherwise they use the internal workspace. Active Workspace and Approved Edit Folders
    are shared with guarded Bash; protected paths and symlink traversal are always blocked.
    .py files get an inline syntax check after a successful write (✓/⚠ appended to
    the result) — a syntax error is reported but NOT reverted; no separate bash
    round-trip needed just to catch it."""
    if not path:
        return "[error] path is required"

    mode = mode or ("batch" if edits is not None else "line" if line_start else "string")
    if mode not in {"create", "replace", "line", "batch", "string"}:
        return f"[error] unsupported edit mode: {mode}"

    target, err, _note = plan_write(path, allow_approved_edit=True)
    if err:
        return err
    p = Path(target)

    try:
        if mode == "create":
            if p.exists() or p.is_symlink():
                return f"[error] create target already exists: {p} — use mode=replace or a precise edit"
            p.parent.mkdir(parents=True, exist_ok=True)
            target, err, _note = plan_write(path, allow_approved_edit=True)
            if err:
                return err
            p = Path(target)
            if p.exists() or p.is_symlink():
                return f"[error] create target already exists: {p} — use mode=replace or a precise edit"
            _atomic_create(p, content)
            tail = _py_syntax_check(p) if p.suffix == ".py" else ""
            return f"created {p}{tail}"

        if mode == "replace":
            if not p.is_file() or p.is_symlink():
                return f"[error] replace mode requires an existing regular file: {p}"
            _atomic_replace(p, content)
            tail = _py_syntax_check(p) if p.suffix == ".py" else ""
            return f"replaced {p} ({len(content)} chars){tail}"

        if mode == "batch":
            if edits is None:
                return "[error] mode=batch requires edits"
            if isinstance(edits, str):
                try:
                    edits = json.loads(edits)
                except Exception:
                    return "[error] edits must be a list of hunk objects — could not parse it as JSON"
            if not isinstance(edits, list) or not edits:
                return "[error] edits must be a non-empty list of hunk objects"
            for i, h in enumerate(edits):
                if not isinstance(h, dict):
                    return f"[error] edits[{i}] must be an object, got {type(h).__name__}"
            hunks = edits
        elif mode == "line":
            if not line_start:
                return "[error] mode=line requires line_start"
            hunks = [{"new_string": new_string, "line_start": line_start, "line_end": line_end}]
        else:
            if not old_string:
                return "[error] old_string is required for string mode"
            hunks = [{
                "old_string": old_string, "new_string": new_string,
                "replace_all": replace_all, "near_line": near_line,
            }]

        if not p.is_file() or p.is_symlink():
            return f"[error] edit target must be an existing regular file: {p}"
        content_before = p.read_text(encoding="utf-8")
        updated = content_before
        descriptions = []
        for i, hunk in enumerate(hunks):
            new_content, herr, desc = _apply_hunk(updated, hunk)
            if herr:
                prefix = f"edit batch failed at hunk {i}: " if mode == "batch" else ""
                return f"[error] {prefix}{herr}"
            updated = new_content
            descriptions.append(desc)

        target, err, _note = plan_write(path, allow_approved_edit=True)
        if err:
            return err
        p = Path(target)
        if not p.is_file() or p.is_symlink():
            return f"[error] edit target changed while preparing mutation: {p}"
        _atomic_replace(p, updated)
        summary = "; ".join(descriptions)
        batch_note = f" ({len(hunks)} hunks)" if mode == "batch" and len(hunks) > 1 else ""
        tail = _py_syntax_check(p) if p.suffix == ".py" else ""
        return f"edited {p} — {summary}{batch_note}{tail}"
    except Exception as e:
        return f"[error] edit failed: {e}"
