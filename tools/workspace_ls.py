"""workspace_ls.py — list all files in workspace as a tree"""
from __future__ import annotations
import os
from pathlib import Path
from langchain_core.tools import tool


@tool
def workspace_ls() -> str:
    """List all files currently in the workspace (recursive tree).
    Call this when the user mentions a filename or dataset by name only, to discover its full relative path before reading or analyzing it."""
    try:
        from config import WORKSPACE
        root = Path(WORKSPACE)
        lines: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("._exec_"))
            rel_dir = Path(dirpath).relative_to(root)
            prefix = f"{rel_dir}/" if str(rel_dir) != "." else ""
            for name in sorted(filenames):
                if not name.startswith("._exec_"):
                    lines.append(f"{prefix}{name}")
        if not lines:
            return "(workspace is empty)"
        return "\n".join(lines)
    except Exception as e:
        return f"[error] workspace_ls failed: {e}"
