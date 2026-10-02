# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from langchain_core.tools import tool
from ._progress import progress
from ._safety import plan_write

_WRITE_CHUNK_BYTES = 256 * 1024


@tool
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Create a new workspace file, or replace the whole file when overwrite=true.

    Use this for full-file writes:
    - creating a new file from scratch
    - saving generated output to a named file
    - intentionally replacing the entire contents of an existing file

    Do NOT use this for small/localized changes inside an existing file — use `edit`
    for that. This tool never appends and never does partial in-place edits: when
    overwrite=true it replaces the whole file atomically.

    path      : workspace-relative ("notes/out.md", "script.py") or an absolute
                path outside the workspace ("~/Desktop/out.md")
    content   : the exact full file contents to write
    overwrite : default false; if the file already exists, set true only when you
                intend to replace the entire file

    Relative paths use Active Workspace when set, otherwise the internal workspace.
    The internal workspace, Active Workspace, and Approved Edit Folders are the
    only writable roots; protected paths and symlink traversal are blocked.
    Existing files must be read with read_file earlier in the current turn before
    overwrite=true is accepted by the normal agent tool guard.

    Returns an [error] if the target exists and overwrite is false. For
    code/scripts, follow with `bash` to verify when execution matters.
    """
    if not path:
        return "[error] path is required"
    target, err, note = plan_write(path, allow_approved_edit=True)
    if err:
        return err
    tmp = None
    try:
        p = Path(target)
        progress(f"กำลังเขียนไฟล์ {path}")
        if p.exists() and not overwrite:
            return f"[error] file exists: {p} ({p.stat().st_size} bytes) — use edit for changes, or retry with overwrite=true to replace the whole file"
        prev_size = p.stat().st_size if p.exists() else None
        p.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        total = len(data)
        written = 0
        fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
        with os.fdopen(fd, "wb") as f:
            if total == 0:
                progress(f"กำลังเขียนไฟล์ {path} (0 bytes)")
            for idx in range(0, total, _WRITE_CHUNK_BYTES):
                chunk = data[idx:idx + _WRITE_CHUNK_BYTES]
                f.write(chunk)
                written += len(chunk)
                progress(f"กำลังเขียนไฟล์ {path} ({written:,}/{total:,} bytes)")
            f.flush()
            os.fsync(f.fileno())
        progress(f"กำลัง finalize ไฟล์ {path}")
        target, err, _ = plan_write(path, allow_approved_edit=True)
        if err:
            os.unlink(tmp)
            return err
        p = Path(target)
        if overwrite:
            os.replace(tmp, p)
        else:
            try:
                os.link(tmp, p, follow_symlinks=False)
            except FileExistsError:
                return f"[error] file exists: {p} — use edit for changes, or set overwrite=true for a full replacement"
            finally:
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass
        hint = f"\n→ verify with bash: python3 {p}" if str(p).endswith(".py") else ""
        cow = f"\nNOTE: {note}" if note else ""
        n_lines = content.count("\n") + 1 if content else 0
        if prev_size is not None:
            return f"overwrote {p}: {len(content)} chars ({n_lines} lines, previous file was {prev_size} bytes){cow}{hint}"
        return f"written {len(content)} chars ({n_lines} lines) to {p}{cow}{hint}"
    except Exception as e:
        if tmp is not None:
            try:
                if isinstance(tmp, str):
                    os.unlink(tmp)
            except Exception:
                pass
        return f"[error] write_file failed: {e}"
