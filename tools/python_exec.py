# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""python_exec.py — run Python code in the same env as endeavor_agent.py

ทำไมไม่ใช้ bash + python3 -c:
- shell escaping ของ pandas/matplotlib code ฝันร้าย (nested quotes, df.query, regex)
- python3 ระบบไม่มี pandas/matplotlib — ต้องใช้ sys.executable (env mlx)
- pandas group/agg อาจใช้เวลา → timeout สูงกว่า bash

scope: data analysis + matplotlib plots. agent เขียน code เอง (27B ทำได้)
"""
from __future__ import annotations
import os
import sys
import subprocess
import tempfile
import uuid
from pathlib import Path
from langchain_core.tools import tool
from tools._progress import progress as _progress
from tools.bash import _build_sandbox_profile

_TIMEOUT_DEFAULT = 120  # data analysis ใช้เวลานานกว่า bash

# skills/ คือจุดเดียวที่ python_exec เขียน legitimate นอก workspace (/build skill เขียน skills/<name>.md)
_SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills")


@tool
def python_exec(code: str, timeout: int = _TIMEOUT_DEFAULT) -> str:
    """Run Python code (cwd = workspace) using the SAME interpreter that runs the agent — pandas, numpy, matplotlib are available.
    Use this for: data analysis (read CSV → filter/group/agg/stats), creating plots with matplotlib (use plt.savefig to save to a file).
    NOT for: shell commands (use bash), file I/O without Python (use read_file/write_file).
    Plot tip: matplotlib.use('Agg') before pyplot import, then savefig('plot.png') — workspace is the cwd."""
    from config import WORKSPACE
    if not code or not code.strip():
        return "[error] code is required"
    script = None
    profile_path = None
    try:
        Path(WORKSPACE).mkdir(parents=True, exist_ok=True)
        script = Path(WORKSPACE) / f"._exec_{uuid.uuid4().hex[:8]}.py"
        # OS-level sandbox เหมือน bash — เขียนได้เฉพาะ workspace + /tmp + skills/ (audit P1)
        profile = _build_sandbox_profile(WORKSPACE, extra_write_paths=(_SKILLS_DIR,))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", delete=False) as pf:
            pf.write(profile)
            profile_path = pf.name
        _progress("running code…")
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["sandbox-exec", "-f", profile_path, sys.executable, str(script)],
                capture_output=True, text=True,
                timeout=timeout, cwd=WORKSPACE,
            )
        finally:
            try:
                script.unlink()
            except Exception:
                pass
            try:
                os.unlink(profile_path)
            except Exception:
                pass
        out = proc.stdout or ""
        if proc.stderr:
            out += f"\n[stderr]\n{proc.stderr}"
        if proc.returncode != 0 and not out.strip():
            return f"[error] exited {proc.returncode}, no output"
        out = out.strip() or "(no output)"
        if len(out) > 10_000:
            out = out[:10_000] + f"\n...[truncated: output exceeded 10,000 chars]"
        return out
    except subprocess.TimeoutExpired:
        return f"[error] timed out after {timeout}s"
    except FileNotFoundError:
        return "[error] sandbox-exec not found — macOS only"
    except Exception as e:
        return f"[error] python_exec failed: {e}"
