from __future__ import annotations
import re
from pathlib import Path
from langchain_core.tools import tool

_MAX_RESULTS = 50
_MAX_LINE_LEN = 2000  # skip longer lines — avoids ReDoS on minified/base64 content


@tool
def grep(pattern: str, path: str = ".", glob: str = "*") -> str:
    """Search for a regex pattern across files in a directory. Returns matching lines as file:line: content."""
    if not pattern:
        return "[error] pattern is required"
    try:
        from ._safety import resolve_read_path
        regex = re.compile(pattern)
        root = Path(resolve_read_path(path))
        files = list(root.rglob(glob)) if root.is_dir() else [root]
        files = [f for f in files if f.is_file() and not any(p.startswith(".") for p in f.relative_to(root).parts)]
        results: list[str] = []
        for f in sorted(files):
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if len(line) > _MAX_LINE_LEN:
                        continue
                    if regex.search(line):
                        results.append(f"{f}:{i}: {line.rstrip()}")
                        if len(results) >= _MAX_RESULTS:
                            break
            except Exception:
                pass
            if len(results) >= _MAX_RESULTS:
                break
        if not results:
            return f"(no matches for '{pattern}')"
        out = "\n".join(results)
        if len(results) >= _MAX_RESULTS:
            out += f"\n...(showing first {_MAX_RESULTS} matches)"
        return out
    except re.error as e:
        return f"[error] invalid regex: {e}"
    except Exception as e:
        return f"[error] grep failed: {e}"
