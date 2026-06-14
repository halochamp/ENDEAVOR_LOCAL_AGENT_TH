from __future__ import annotations
import os
from pathlib import Path
from langchain_core.tools import tool
from ._safety import check_path, resolve_path


def _normalize_lines(text: str) -> str:
    """Strip trailing whitespace per line — fixes whitespace mismatch on old_string."""
    return "\n".join(line.rstrip() for line in text.splitlines())


@tool
def edit(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Modify an EXISTING file by replacing old_string with new_string.
    old_string must be a unique substring of the file (or set replace_all=true). Use write_file for new files / large rewrites."""
    if not path:
        return "[error] path is required"
    if not old_string:
        return "[error] old_string is required"
    err = check_path(path)
    if err:
        return err
    try:
        p = Path(resolve_path(path))
        if not p.exists():
            return f"[error] file not found: {path}"
        content = p.read_text(encoding="utf-8")
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
            return (
                f"[error] old_string not found in {path}.\n"
                f"File content (first 600 chars):\n{content[:600]}\n"
                "Copy old_string exactly from the content above."
            )
        if count > 1 and not replace_all:
            return f"[error] old_string found {count} times — use replace_all=true or make it more unique"
        new_content = (
            content.replace(old_string, new_string)
            if replace_all else content.replace(old_string, new_string, 1)
        )
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(new_content, encoding="utf-8")
        os.replace(tmp, p)
        n = count if replace_all else 1
        hint = f"\n→ verify with bash: python3 {path}" if path.endswith(".py") else ""
        return f"edited {path} — replaced {n} occurrence(s){hint}"
    except Exception as e:
        return f"[error] edit failed: {e}"
