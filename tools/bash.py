# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

from __future__ import annotations
import os
import subprocess
import tempfile
from langchain_core.tools import tool
from tools._truncate import truncate_with_save

_SANDBOX_DENY_MARKERS = ("Operation not permitted", "Permission denied", "Read-only file system")


def _classify_bash_error(returncode: int, stderr: str, command: str, workspace: str) -> str | None:
    """Evidence-gated recovery hint for a failed bash call — mirrors python_exec's
    error classifier (a wrong hint is worse than no hint, so each branch checks the
    actual signal, not just a guess from the command text)."""
    if returncode == 127:
        return "command not found — check spelling, or use an absolute path / `which <name>` first."
    if returncode == 126:
        return "permission denied executing that file — check it's executable (chmod +x) or not a directory."
    if returncode != 0 and any(m in stderr for m in _SANDBOX_DENY_MARKERS):
        return (f"denied by the sandbox — writes are confined to workspace/ ({workspace}) and /tmp; "
                f"a handful of sensitive read paths (/etc/passwd, ~/.ssh, credential stores, ...) are "
                f"blocked too. If this was a write, save under workspace/ instead of ~/Desktop, "
                f"~/Documents, etc.")
    return None


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

; deny read: user/credential enumeration files + credentials + session history.
; NOT a blanket (subpath "/etc") deny: this sandbox's DENY always wins over a
; later ALLOW for an overlapping subpath (verified empirically — textual order
; does not matter for a subpath conflict, unlike the plain string this repo's
; comments used to assume), so blanket-denying /etc would permanently break
; /etc/ssl (TLS trust store — curl/git-over-https/openssl need it, no working
; allow-exception is possible once the parent subpath is denied) with no way
; to carve it back out. Named literals instead: /etc/passwd and /etc/group are
; world-readable (verified: `stat -f %Sp` shows rw-r--r--) and were the actual
; demonstrated leak (`bash('cat /etc/passwd')` returned real content, bypassing
; read_file's protection entirely — sandbox-exec's file-write* deny above never
; touched reads, only writes). master.passwd/shadow/sudoers are root-only or
; group-read by OS permissions already (verified: rw-------, r--r-----) so the
; OS itself blocks a normal user process regardless — listed anyway as
; defense-in-depth, at zero cost to legitimate access other tools need under
; /etc for ssl certs, hosts, resolv.conf, services, protocols, etc.
(deny file-read*
  (literal "/etc/passwd") (literal "/private/etc/passwd")
  (literal "/etc/group") (literal "/private/etc/group")
  (literal "/etc/master.passwd") (literal "/private/etc/master.passwd")
  (literal "/etc/shadow") (literal "/private/etc/shadow")
  (literal "/etc/sudoers") (literal "/private/etc/sudoers")
  (subpath "{home}/.ssh")
  (subpath "{home}/.aws")
  (subpath "{home}/.gnupg")
  (subpath "{home}/.claude")
  (subpath "{home}/.config")
)

; workspace + /tmp + extra — allow ทีหลัง (last-match wins) override deny Desktop
(allow file-write* (subpath "{workspace}") (subpath "/private/tmp"){extra})
(allow file-read*  (subpath "{workspace}"))
"""


def _bash_impl(command: str, timeout: int = 30) -> str:
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
        try:
            result = subprocess.run(
                ["sandbox-exec", "-f", profile_path, "bash", "-c", command],
                capture_output=True, text=True,
                timeout=timeout, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as e:
            # Return what ran before the timeout instead of discarding it — a command that
            # got most of the way there shouldn't force a blind full re-run from scratch.
            # e.stdout/e.stderr come back as bytes here even with text=True (subprocess
            # quirk on TimeoutExpired specifically) — decode defensively rather than
            # assuming either type.
            def _decode(x) -> str:
                if x is None:
                    return ""
                return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x
            partial = _decode(e.stdout)
            if e.stderr:
                partial += f"\n[stderr]\n{_decode(e.stderr)}"
            partial = partial.strip()
            note = f"[error] command timed out after {timeout}s"
            if partial:
                note += " — partial output before timeout:"
                # marker_first: same reason as the normal-path call below — tool_loop._bash_each
                # applies its own secondary cut on top of this result.
                partial = truncate_with_save(partial, 10_000, WORKSPACE, "bash",
                                              marker_first=True, keep_tail=True)
                return f"{note}\n{partial}"
            return note

        output = result.stdout or ""
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        output = output.strip() or "(no output)"
        hint = _classify_bash_error(result.returncode, result.stderr or "", command, WORKSPACE)
        if hint:
            output += f"\n[hint] {hint}"
        # marker_first: tool_loop._bash_each applies its own secondary 2,000-char cut on top
        # of this result — a trailing marker could get sliced off, silently dropping the
        # recovery-file path. A leading marker survives that secondary cut.
        # keep_tail: build/test errors sit at the end of the output — a head-only cut hides
        # exactly the part that matters most.
        output = truncate_with_save(output, 10_000, WORKSPACE, "bash", marker_first=True, keep_tail=True)
        return output
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


@tool
def bash(command: str, timeout: int = 30) -> str:
    """Run a bash command on the local machine (cwd = workspace) — system operations, run scripts, check processes/disk/memory.
    NOT for arithmetic, math, or data analysis (pandas/statistics) — use python_exec for those; never for shell-wrapped Python.

    FILE SEARCH (this machine runs macOS) — cwd = workspace/, so relative paths (find ., ls) only search there.
    To find files elsewhere on the machine, use absolute paths or ~:
      - mdfind -name "keyword"                              → macOS Spotlight, whole-disk, fastest — try first
      - find ~ -iname "*keyword*" 2>/dev/null | head -20    → fallback when mdfind misses unindexed files
      - rg -n "keyword" ~/Desktop/<project>                  → search file CONTENT fast (grep -rl fallback if no rg)
      - common dirs: ~/Desktop, ~/Documents, ~/Downloads

    MAC APP CONTROL (sandbox-safe subset — app/file/clipboard/system actions bash can do on its own;
    GUI clicking/typing/keystrokes are not available in this sandbox):
      - open -a "Google Chrome"      → open an app, OR switch to / raise an already-running one (hidden window comes to front)
      - open <path|URL>              → open file/folder/URL with its default app
      - pbpaste  /  echo "x" | pbcopy → read / write the clipboard
      - osascript -e 'display notification "งานเสร็จแล้ว" with title "Endeavor"' → notify the user (use after finishing a long task)
      - osascript -e 'set volume output volume 40'  /  ... -e 'output volume of (get volume settings)' → set / read volume
      GUI scripting via bash is NOT available: osascript System Events (window lists, menu clicks, keystrokes)
      is blocked by this tool's sandbox (-10004 privilege violation).

    KNOWN PATHS — this agent's own files (use when user asks about yourself / your architecture):
      Find the project root first (location may change): mdfind -name "ENDEAVOR_LOCAL_AGENT_TH" -onlyin ~ | head -1
      Then inside <project_root>/:
        - logs/memory.md  → persistent memory
        - #developer/     → private source, never read/expose to the user

    FILE DISCOVERY / CODE SEARCH inside workspace:
      - rg --files                          → flat file list (find "$PWD" -type f fallback if no rg)
      - rg --files | rg "\\.md$"             → find files by name/pattern
      - rg -n "needle" path_or_dir          → search text with line numbers

    FILE WRITE — sandboxed: only workspace/ (cwd) and /tmp allow writes.
    Writing elsewhere (~/Desktop, ~/Documents, ~/Pictures, etc.) errors/fails silently —
    save to workspace/ then tell user the path, or use write_file/edit for workspace files.
    cwd is ALREADY workspace/ — do not re-prefix "workspace/" onto the output path,
    that writes one level too deep and the file won't be where you said it is.
      ❌ screencapture workspace/shot.png  → lands at workspace/workspace/shot.png
      ✅ screencapture shot.png            → lands at workspace/shot.png

    Output cap: output over 10,000 chars is truncated; the FULL output is saved to a
    workspace file whose path is in the leading "[bash] truncated: ..." marker.

    NETWORK — basic read-only network checks are fine via bash:
      ping <host>, netstat -an, ss -tuln, ifconfig, arp -a, nmap -sn <range>
    """
    return _bash_impl(command, timeout)
