# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

from __future__ import annotations
import os
from pathlib import Path
from langchain_core.tools import tool
from ._safety import check_path, resolve_path


@tool
def write_file(path: str, content: str) -> str:
    """Create a NEW file with the given content (workspace only). For modifying existing files, use edit instead."""
    if not path:
        return "[error] path is required"
    err = check_path(path)
    if err:
        return err
    try:
        p = Path(resolve_path(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, p)
        hint = f"\n→ verify with bash: python3 {p}" if str(p).endswith(".py") else ""
        return f"written {len(content)} chars to {p}{hint}"
    except Exception as e:
        return f"[error] write_file failed: {e}"
