# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

from __future__ import annotations
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from langchain_core.tools import tool
from tools._truncate import truncate_with_save

_SANDBOX_DENY_MARKERS = ("Operation not permitted", "Permission denied", "Read-only file system")
_DELTA_MAX_FILES = 8_000
_DELTA_MAX_DIRS = 20_000
_DELTA_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_DELTA_MAX_FILE_BYTES = 4 * 1024 * 1024
_DELTA_MAX_CHANGES = 100
_SYMLINK_SCAN_MAX_DIRS = 20_000
_SYMLINK_SCAN_MAX_ENTRIES = 100_000
_SYMLINK_SCAN_MAX_LINKS = 2_000
_DELTA_IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".cache", "cache", "Caches", ".next", ".venv", "venv",
}


def _classify_bash_error(returncode: int, stderr: str, command: str, workspace: str) -> str | None:
    """Evidence-gated recovery hint for a failed bash call — mirrors python_exec's
    error classifier (a wrong hint is worse than no hint, so each branch checks the
    actual signal, not just a guess from the command text)."""
    if returncode == 127:
        return "command not found — check spelling, or use an absolute path / `which <name>` first."
    if returncode == 126:
        return "permission denied executing that file — check it's executable (chmod +x) or not a directory."
    if returncode != 0 and any(m in stderr for m in _SANDBOX_DENY_MARKERS):
        return ("denied by the sandbox — writes are confined to the internal workspace, current "
                "Active Workspace, Approved Edit Folders, and /tmp; "
                f"a handful of sensitive read paths (/etc/passwd, ~/.ssh, credential stores, ...) are "
                "blocked too. If this was a write, use an authorized workspace folder.")
    return None


def _build_sandbox_profile(
    workspace: str,
    extra_write_paths: tuple[str, ...] = (),
    *,
    strict_write_roots: tuple[str, ...] | None = None,
) -> str:
    """สร้าง macOS sandbox-exec profile — allow default, deny writes นอก workspace

    extra_write_paths: subpath เพิ่มที่อนุญาตให้เขียน (เช่น skills/ สำหรับ python_exec)
    — append หลัง deny block → last-match wins → override deny Desktop
    """
    home = os.path.expanduser("~")
    extra = "".join(f' (subpath "{os.path.realpath(p)}")' for p in extra_write_paths)
    if strict_write_roots is not None:
        roots = list(dict.fromkeys(os.path.realpath(p) for p in strict_write_roots))
        allow_roots = "".join(f" (subpath {json.dumps(p)})" for p in roots)
        from tools._safety import _PROTECTED_PATHS
        protected = "".join(f" (subpath {json.dumps(os.path.realpath(p))})" for p in _PROTECTED_PATHS)
        return f"""(version 1)
(allow default)
(deny file-write* (subpath "/"))
(allow file-write* (subpath "/private/tmp") (subpath "/dev/null"){allow_roots})
(deny file-write*{protected})
(deny file-read*
  (literal "/etc/passwd") (literal "/private/etc/passwd")
  (literal "/etc/group") (literal "/private/etc/group")
  (literal "/etc/master.passwd") (literal "/private/etc/master.passwd")
  (literal "/etc/shadow") (literal "/private/etc/shadow")
  (literal "/etc/sudoers") (literal "/private/etc/sudoers")
  (subpath "{home}/.ssh") (subpath "{home}/.aws")
  (subpath "{home}/.gnupg") (subpath "{home}/.claude")
  (subpath "{home}/.config")
)
"""
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


def _hash_regular_file(path: str) -> tuple[str, int]:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("not a regular file")
        if before.st_nlink > 1:
            raise OSError("hard-linked target is not eligible for mutation evidence")
        if before.st_size > _DELTA_MAX_FILE_BYTES:
            raise OSError("file exceeds bounded snapshot size")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _DELTA_MAX_FILE_BYTES:
                raise OSError("file exceeds bounded snapshot size")
            digest.update(chunk)
        after = os.fstat(fd)
        current = os.lstat(path)
        signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        if signature(before) != signature(after) or signature(after) != signature(current):
            raise OSError("file changed while snapshot was being read")
        return digest.hexdigest(), total
    finally:
        os.close(fd)


def _snapshot_target(path: str) -> tuple[dict, str | None]:
    from tools._safety import _protected_hit, validate_write_target
    canonical, error = validate_write_target(path)
    if error:
        return {}, error
    if os.path.islink(canonical):
        return {}, "symlink targets are not eligible for Bash mutation"
    real = os.path.realpath(canonical)
    if _protected_hit(real):
        return {}, "protected target"
    try:
        os.lstat(canonical)
    except FileNotFoundError:
        return {os.path.abspath(canonical): None}, None
    try:
        digest, size = _hash_regular_file(canonical)
    except OSError as exc:
        return {}, f"snapshot failed: {exc}"
    return {os.path.abspath(canonical): (digest, size)}, None


def _snapshot_tree(root: str) -> tuple[dict, str | None]:
    from tools._safety import _authorized_write_roots, _is_within, _protected_hit
    root = os.path.realpath(root)
    if not any(_is_within(root, allowed) for allowed in _authorized_write_roots()):
        return {}, "tree scope is outside an authorized write root"
    result: dict[str, tuple[str, int] | None] = {}
    total_bytes = 0
    stack = [root]
    visited_dirs = 0
    while stack:
        dirpath = stack.pop()
        visited_dirs += 1
        if visited_dirs > _DELTA_MAX_DIRS:
            return {}, "tree snapshot exceeded bounded directory limit"
        try:
            entries = os.scandir(dirpath)
        except OSError as exc:
            return {}, f"incomplete tree snapshot at {dirpath}: {exc}"
        with entries:
            for entry in entries:
                name = entry.name
                if name in _DELTA_IGNORED_DIRS:
                    continue
                path = entry.path
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                    if stat.S_ISLNK(mode) or _protected_hit(os.path.realpath(path)):
                        continue
                    if stat.S_ISDIR(mode):
                        stack.append(path)
                        if visited_dirs + len(stack) > _DELTA_MAX_DIRS:
                            return {}, "tree snapshot exceeded bounded directory limit"
                        continue
                    if not stat.S_ISREG(mode):
                        continue
                except OSError as exc:
                    return {}, f"incomplete tree snapshot at {path}: {exc}"
                if name.endswith((".pyc", ".pyo", ".tmp")) or name.startswith(("._", ".#")):
                    continue
                try:
                    digest, size = _hash_regular_file(path)
                except OSError as exc:
                    return {}, f"incomplete tree snapshot at {path}: {exc}"
                result[os.path.abspath(path)] = (digest, size)
                total_bytes += size
                if len(result) > _DELTA_MAX_FILES or total_bytes > _DELTA_MAX_TOTAL_BYTES:
                    return {}, "tree snapshot exceeded bounded file/byte limits"
    return result, None


def _scan_authorized_symlinks(roots: tuple[str, ...]) -> tuple[list[str], str | None]:
    """Boundedly reject existing symlinks anywhere Bash could write this turn.

    Seatbelt evaluates file writes against the resolved destination, so a symlink
    from one authorized root into another can otherwise pass the root allow rules.
    Mutation calls fail closed if any existing file/directory symlink is present
    in their writable roots, or if this scan cannot complete within its limits.
    """
    from tools._safety import _protected_hit

    links: list[str] = []
    visited_roots: set[str] = set()
    visited_dirs = 0
    visited_entries = 0
    for raw_root in roots:
        root = os.path.realpath(raw_root)
        if not os.path.isdir(root):
            return [], f"authorized write root is unavailable: {root}"
        # Scan nested roots independently: an ancestor may intentionally skip an
        # ignored cache/vendor directory that is itself the current Active Workspace.
        if root in visited_roots:
            continue
        visited_roots.add(root)
        stack = [root]
        while stack:
            directory = stack.pop()
            visited_dirs += 1
            if visited_dirs > _SYMLINK_SCAN_MAX_DIRS:
                return [], "authorized roots exceeded the bounded symlink directory scan"
            try:
                entries = os.scandir(directory)
            except OSError as exc:
                return [], f"cannot complete authorized-root symlink scan: {exc}"
            with entries:
                for entry in entries:
                    visited_entries += 1
                    if visited_entries > _SYMLINK_SCAN_MAX_ENTRIES:
                        return [], "authorized roots exceeded the bounded symlink entry scan"
                    path = entry.path
                    try:
                        mode = entry.stat(follow_symlinks=False).st_mode
                        if stat.S_ISLNK(mode):
                            links.append(os.path.abspath(path))
                            if len(links) > _SYMLINK_SCAN_MAX_LINKS:
                                return [], "authorized roots exceeded the bounded symlink count"
                            continue
                        if entry.name in _DELTA_IGNORED_DIRS:
                            continue
                        if not stat.S_ISDIR(mode) or _protected_hit(os.path.realpath(path)):
                            continue
                        stack.append(path)
                    except OSError as exc:
                        return [], f"cannot safely inspect authorized-root entry: {exc}"
    return links, None


def _filesystem_delta(before: dict, after: dict, scope_kind: str, scope_path: str) -> dict | None:
    changed = []
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path), after.get(path)
        if old == new:
            continue
        if old is None and new is None:
            continue
        changed.append({
            "path": os.path.realpath(path),
            "change_type": "created" if old is None else "deleted" if new is None else "modified",
            "before_sha256": old[0] if old else None,
            "after_sha256": new[0] if new else None,
        })
    if not changed or len(changed) > _DELTA_MAX_CHANGES:
        return None
    exact = scope_kind == "exact"
    return {
        "provenance_type": "bash_filesystem_delta_v1",
        "scope": scope_kind,
        "changed_paths": changed,
        "file_count": len(changed),
        "observation_scope": {
            "coverage": "requested_targets_only" if exact else "bounded_authorized_tree",
            "path": None if exact else scope_path,
            "requested_targets": sorted(os.path.realpath(p) for p in before) if exact else [],
            "change_list_complete_within_scope": True,
            "exclusions": ["protected paths", "symlinks", "cache directories", "temporary files", "bytecode"],
        },
        "snapshot_limits": {
            "max_files": _DELTA_MAX_FILES,
            "max_directories": _DELTA_MAX_DIRS,
            "max_total_bytes": _DELTA_MAX_TOTAL_BYTES,
            "max_file_bytes": _DELTA_MAX_FILE_BYTES,
            "max_changed_paths": _DELTA_MAX_CHANGES,
        },
    }


def _capture_call_delta(scope_kind: str, targets: tuple[str, ...], tree_root: str, before: dict,
                       command_exit_code: int | None) -> tuple[dict | None, str | None]:
    if scope_kind == "exact":
        after: dict = {}
        for target in targets:
            snap, error = _snapshot_target(target)
            if error:
                return None, error
            after.update(snap)
    else:
        after, error = _snapshot_tree(tree_root)
        if error:
            return None, error
    scope_path = tree_root if scope_kind == "tree" else ",".join(targets)
    delta = _filesystem_delta(before, after, scope_kind, scope_path)
    if delta is None and before != after:
        return None, "changed path count exceeded the bounded evidence limit"
    if delta:
        delta["command_exit_code"] = command_exit_code
    return delta, None


def _bash_execute(command: str, timeout: int = 30) -> tuple[str, dict | None]:
    from config import WORKSPACE
    from tools import edit_access
    from tools._safety import _authorized_write_roots, validate_write_target

    if not command:
        return "[error] command is required", None

    # Block pure-echo progress markers — model uses bash('echo "..."') as step announcements
    # during plan execution. Only block when echo has no redirect / pipe / variable (those are
    # legitimate: echo "x" > file.txt, echo $PATH, echo "x" | grep ...).
    _cmd = command.strip()
    if _cmd.startswith("echo ") and not any(c in _cmd for c in (">", "|", "$", "&", "`")):
        return "", None

    scope = edit_access.get_bash_mutation_scope()
    scope_kind = ""
    targets: tuple[str, ...] = ()
    tree_root = ""
    write_roots: tuple[str, ...] | None = None
    before: dict = {}
    cwd = WORKSPACE
    if scope:
        scope_kind = str(scope.get("kind") or "")
        targets = tuple(str(item) for item in scope.get("targets") or ())
        tree_root = str(scope.get("tree_root") or "")
        write_roots = tuple(str(item) for item in scope.get("write_roots") or ())
        if not write_roots or any(not os.path.isdir(root) for root in write_roots):
            return "[error] mutation scope lost its authorized roots; command was not executed", None
        links, scan_error = _scan_authorized_symlinks(write_roots)
        if scan_error:
            return f"[error] {scan_error}; command was not executed", None
        if links:
            return (
                "[error] Bash mutation refused because an authorized write root contains an existing symlink; "
                f"use an exact non-symlink target or remove the link first: {links[0]}; command was not executed",
                None,
            )
        if scope_kind == "exact":
            for target in targets:
                checked, error = validate_write_target(target)
                if error:
                    return f"[error] {error}; command was not executed", None
                snap, error = _snapshot_target(checked)
                if error:
                    return f"[error] {error}; command was not executed", None
                before.update(snap)
            cwd = os.path.realpath(edit_access.get_focus_folder() or WORKSPACE)
        elif scope_kind == "tree":
            checked, error = validate_write_target(tree_root)
            if error:
                return f"[error] {error}; command was not executed", None
            cwd = os.path.realpath(checked)
            before, error = _snapshot_tree(cwd)
            if error:
                return f"[error] {error}; command was not executed", None
        else:
            return "[error] invalid mutation scope; command was not executed", None

    profile_path = None
    try:
        profile = (
            _build_sandbox_profile(WORKSPACE, strict_write_roots=write_roots)
            if scope else _build_sandbox_profile(WORKSPACE)
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", delete=False) as f:
            f.write(profile)
            profile_path = f.name
        try:
            env = os.environ.copy()
            if scope:
                env.update({"TMPDIR": "/tmp", "TMP": "/tmp", "TEMP": "/tmp"})
            result = subprocess.run(
                ["sandbox-exec", "-f", profile_path, "bash", "-c", command],
                capture_output=True, text=True,
                timeout=timeout, cwd=cwd, stdin=subprocess.DEVNULL, env=env,
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
                if scope:
                    partial = partial[:9_000] + ("\n...[truncated; scoped mutation runs do not create output files]" if len(partial) > 9_000 else "")
                else:
                    partial = truncate_with_save(partial, 10_000, WORKSPACE, "bash",
                                                  marker_first=True, keep_tail=True)
                timeout_artifact = None
                if scope:
                    timeout_artifact, delta_error = _capture_call_delta(scope_kind, targets, tree_root, before, None)
                    if delta_error:
                        note += f"\n[filesystem delta unavailable: {delta_error}]"
                    elif timeout_artifact:
                        paths = ", ".join(item["path"] for item in timeout_artifact["changed_paths"][:8])
                        note += f"\n[host observed filesystem delta: {timeout_artifact['file_count']} file(s): {paths}]"
                    elif not delta_error:
                        note += "\n[no filesystem delta observed within tracked scope]"
                return f"{note}\n{partial}", timeout_artifact
            return note, None

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
        if scope:
            if len(output) > 10_000:
                output = output[:9_000] + "\n...[truncated; scoped mutation runs do not create output files]"
        else:
            output = truncate_with_save(output, 10_000, WORKSPACE, "bash", marker_first=True, keep_tail=True)
        if scope:
            delta, error = _capture_call_delta(scope_kind, targets, tree_root, before, result.returncode)
            if error:
                return output + f"\n[filesystem delta unavailable: {error}]", None
            if delta:
                paths = ", ".join(item["path"] for item in delta["changed_paths"][:8])
                output += f"\n[host observed filesystem delta: {delta['file_count']} file(s): {paths}]"
                return output, delta
            output += "\n[no filesystem delta observed within tracked scope]"
        return output, None
    except FileNotFoundError:
        return "[error] sandbox-exec not found — macOS only", None
    except Exception as e:
        return f"[error] bash failed: {e}", None
    finally:
        if profile_path:
            try:
                os.unlink(profile_path)
            except Exception:
                pass


def _bash_impl(command: str, timeout: int = 30) -> str:
    """Plain-string compatibility entry point for internal read/execution helpers."""
    return _bash_execute(command, timeout)[0]


@tool(response_format="content_and_artifact")
def bash(command: str, timeout: int = 30) -> tuple[str, dict | None]:
    """Run a bash command on the local machine (normal cwd = workspace) — system operations, run scripts, check processes/disk/memory.
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

    FILE WRITE — read/test/process commands work as usual. For a clear requested file change,
    guarded Bash can write only inside the internal workspace, current Active Workspace, or
    persistent Approved Edit Folders; protected paths and symlink traversal remain blocked.
    A mutation-scoped Bash call fails closed if an existing symlink is found anywhere in its
    writable roots, because shell commands can address files indirectly.
    A successful command is not mutation proof: the host reports a delta only after observing
    changed on-disk content. Mutation cwd is Active Workspace when set, otherwise workspace/.
    cwd is ALREADY the active workspace for an authorized mutation — do not re-prefix "workspace/" onto the output path,
    that writes one level too deep and the file won't be where you said it is.
      ❌ screencapture workspace/shot.png  → lands at workspace/workspace/shot.png
      ✅ screencapture shot.png            → lands at workspace/shot.png

    Output cap: output over 10,000 chars is truncated; the FULL output is saved to a
    workspace file whose path is in the leading "[bash] truncated: ..." marker.

    NETWORK — basic read-only network checks are fine via bash:
      ping <host>, netstat -an, ss -tuln, ifconfig, arp -a, nmap -sn <range>
    """
    return _bash_execute(command, timeout)
