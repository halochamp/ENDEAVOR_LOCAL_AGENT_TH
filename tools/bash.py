from __future__ import annotations
import os
import subprocess
import tempfile
from langchain_core.tools import tool


def _build_sandbox_profile(workspace: str, extra_write_paths: tuple[str, ...] = ()) -> str:
    """สร้าง macOS sandbox-exec profile — allow default, deny writes นอก workspace

    extra_write_paths: subpath เพิ่มที่อนุญาตให้เขียน (เช่น skills/ สำหรับ python_exec)
    — append หลัง deny block → last-match wins → override deny Desktop
    """
    home = os.path.expanduser("~")
    extra = "".join(f' (subpath "{os.path.realpath(p)}")' for p in extra_write_paths)
    return f"""(version 1)
(allow default)

; deny writes ไปยัง paths อันตราย
(deny file-write*
  (subpath "/etc") (subpath "/private/etc")
  (subpath "/usr") (subpath "/bin") (subpath "/sbin")
  (subpath "/System") (subpath "/Library") (subpath "/Applications")
  (subpath "{home}/Desktop")
  (subpath "{home}/Documents") (subpath "{home}/Downloads")
  (subpath "{home}/Movies") (subpath "{home}/Music") (subpath "{home}/Pictures")
  (subpath "{home}/.ssh") (subpath "{home}/.aws")
  (subpath "{home}/.config") (subpath "{home}/.gnupg")
  (subpath "{home}/Library")
)

; deny read credentials
(deny file-read*
  (subpath "{home}/.ssh")
  (subpath "{home}/.aws")
  (subpath "{home}/.gnupg")
)

; workspace + /tmp + extra — allow ทีหลัง (last-match wins) override deny Desktop
(allow file-write* (subpath "{workspace}") (subpath "/private/tmp"){extra})
(allow file-read*  (subpath "{workspace}"))
"""


@tool
def bash(command: str, timeout: int = 30) -> str:
    """Run a bash command on the local machine (cwd = workspace) — system operations, run scripts, check processes/disk/memory.
    NOT for arithmetic or math — answer math questions directly without this tool.

    FILE SEARCH (this machine runs macOS) — cwd = workspace/, so relative paths (find ., ls) only search there.
    To find files elsewhere on the machine, use absolute paths or ~:
      - mdfind -name "keyword"                              → macOS Spotlight, whole-disk, fastest — try first
      - find ~ -iname "*keyword*" 2>/dev/null | head -20    → fallback when mdfind misses unindexed files
      - grep -rl "keyword" ~/Desktop/<project> 2>/dev/null  → search by file CONTENT (don't know the filename)
      - common dirs: ~/Desktop, ~/Documents, ~/Downloads

    KNOWN PATHS — this agent's own files (use when user asks about yourself / your architecture):
      Find the project root first (location may change): mdfind -name "ENDEAVOR_AGENTIC" -onlyin ~ | head -1
      Then inside <project_root>/ENDEAVOR_AGENT_V2/:
        - logs/memory.md  → persistent memory
        - developer/      → PROJECT_MEMORY.md (tools/routing/models/bugs/plans), agent_content.md, agent_decision_v2.md, agent_knowhow.md

    FILE WRITE — sandboxed: only workspace/ (cwd) and /tmp allow writes.
    Writing elsewhere (~/Desktop, ~/Documents, ~/Pictures, etc.) errors/fails silently —
    save to workspace/ then tell user the path, or use write_file/edit for workspace files.
    """
    from config import WORKSPACE

    if not command:
        return "[error] command is required"

    # Block pure-echo progress markers — model uses bash('echo "..."') as step announcements
    # during plan execution. Only block when echo has no redirect / pipe / variable (those are
    # legitimate: echo "x" > file.txt, echo $PATH, echo "x" | grep ...).
    _cmd = command.strip()
    if _cmd.startswith("echo ") and not any(c in _cmd for c in (">", "|", "$", "&", "`")):
        return ""

    profile_path = None
    try:
        profile = _build_sandbox_profile(WORKSPACE)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", delete=False) as f:
            f.write(profile)
            profile_path = f.name
        result = subprocess.run(
            ["sandbox-exec", "-f", profile_path, "bash", "-c", command],
            capture_output=True, text=True,
            timeout=timeout, cwd=WORKSPACE,
        )
        output = result.stdout or ""
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        output = output.strip() or "(no output)"
        if len(output) > 10_000:
            output = output[:10_000] + f"\n...[truncated: output exceeded 10,000 chars]"
        return output
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {timeout}s"
    except FileNotFoundError:
        return "[error] sandbox-exec not found — macOS only"
    except Exception as e:
        return f"[error] bash failed: {e}"
    finally:
        if profile_path:
            try:
                os.unlink(profile_path)
            except Exception:
                pass
