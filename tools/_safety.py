# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Shared filesystem path safety for edit and guarded Bash.

Writes are limited to the internal workspace, the temporary Active Workspace,
and persistent Approved Edit Folders from tools.edit_access. Protected paths and
lexical symlink traversal are rejected before canonical-path checks. Read tools
retain their broader policy while blocking protected system and credential paths.

The default plan_write mode retains the legacy outside-workspace copy behavior
for older callers; agent mutation tools opt into the shared approved-root policy.
"""
from __future__ import annotations
import os

_PROTECTED_PATHS = [
    "/etc/", "/usr/", "/bin/", "/sbin/", "/lib/",
    "/System/", "/Library/", "/Applications/",
    os.path.expanduser("~/.ssh/"),
    os.path.expanduser("~/.aws/"),
    os.path.expanduser("~/.gnupg/"),
    os.path.expanduser("~/.claude/"),
    # blanket ~/Library/ used to block EVERYTHING under it, including
    # ~/Library/CloudStorage/ — where macOS (Monterey+) actually mounts
    # Google Drive/OneDrive/iCloud Drive, i.e. the user's own real files, not
    # app internals. Replaced with a targeted list of the credential-bearing
    # subpaths under ~/Library/ instead of the whole tree, so cloud-drive
    # files (and other ordinary per-app data like Mail/Safari/Preferences/
    # Caches) are readable while secrets stay blocked:
    os.path.expanduser("~/Library/Keychains/"),          # Keychain databases
    os.path.expanduser("~/Library/Application Support/"),  # most apps' saved
        # credentials/tokens live here (browser "Login Data", password
        # managers, cloud-CLI credential caches, crypto wallets, etc.) —
        # the single highest-value entry beyond Keychains itself
    os.path.expanduser("~/Library/Containers/"),          # sandboxed per-app
    os.path.expanduser("~/Library/Group Containers/"),    # data, incl. many
        # third-party password managers/VPN clients macOS forces in here
        # instead of Application Support
    os.path.expanduser("~/Library/Cookies/"),
    os.path.expanduser("~/Library/HTTPStorages/"),         # web session tokens
    os.path.expanduser("~/Library/Messages/"),             # iMessage chat.db —
        # can contain SMS/iMessage-delivered 2FA/OTP codes
    # WebSocket auth token — parent of workspace, reachable via "../.agent_token"
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".agent_token")),
    # User edit-access state must never be directly edited through the agent.
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "approved_edit_folders.json")),
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "approved_edit_folders.json.lock")),
]


def _strip_ws_prefix(path: str, workspace: str) -> str:
    ws_name = os.path.basename(workspace.rstrip("/\\"))
    norm = path.replace("\\", "/")
    if norm.startswith(ws_name + "/"):
        return norm[len(ws_name) + 1:]
    return path


def _protected_hit(abs_path: str) -> str | None:
    """abs_path ต้องผ่าน realpath มาแล้ว — คืนชื่อ protected path ที่โดน, None ถ้าไม่โดน"""
    for protected in _PROTECTED_PATHS:
        real_protected = os.path.realpath(protected)
        if abs_path == real_protected or abs_path.startswith(real_protected + os.sep):
            return protected
    return None


_EDITED_MARK = ".edited"


def edited_copy_path(path: str) -> str:
    """คู่ working copy ของไฟล์: report.md → report.edited.md (ไม่มีนามสกุล → Makefile.edited)"""
    root, ext = os.path.splitext(path)
    return root + _EDITED_MARK + ext


def is_edited_copy(path: str) -> bool:
    root, _ = os.path.splitext(os.path.basename(path))
    return root.endswith(_EDITED_MARK)


def _in_workspace(abs_path: str) -> bool:
    from config import WORKSPACE
    ws_abs = os.path.realpath(WORKSPACE)
    return abs_path == ws_abs or abs_path.startswith(ws_abs + os.sep)


def _is_within(path: str, root: str) -> bool:
    try:
        resolved = os.path.realpath(os.path.abspath(path))
        canonical_root = os.path.realpath(root)
        return os.path.commonpath([resolved, canonical_root]) == canonical_root
    except ValueError:
        return False


def _authorized_write_roots() -> list[str]:
    from .edit_access import authorized_write_roots
    roots = []
    for root in authorized_write_roots():
        canonical = os.path.realpath(root)
        if _protected_hit(canonical):
            continue
        if canonical not in roots:
            roots.append(canonical)
    return roots


def _lexical_symlink_error(path: str, roots: list[str]) -> str | None:
    """Reject symlink components before realpath can erase their lexical form."""
    lexical = os.path.abspath(path)
    # macOS commonly exposes canonical /private paths through /var or /tmp
    # symlink aliases. Find the lexical prefix that resolves exactly to the
    # approved canonical root, then inspect every component beneath that root.
    components_all = [part for part in lexical.split(os.sep) if part]
    matches: list[tuple[str, str, list[str]]] = []
    for candidate_root in roots:
        current_prefix = os.path.abspath(os.sep)
        for index, component in enumerate(components_all):
            current_prefix = os.path.join(current_prefix, component)
            if os.path.realpath(current_prefix) == candidate_root:
                matches.append((candidate_root, current_prefix, components_all[index + 1:]))
                break
    if not matches:
        return f"[BLOCKED] path is outside the internal workspace and outside Approved Edit Folders/current Active Workspace: {lexical}"
    root, alias_root, remaining = max(matches, key=lambda item: len(item[0]))
    if not alias_root:
        return f"[BLOCKED] path does not resolve through an authorized folder: {lexical}"
    current = alias_root
    components = remaining
    for component in components:
        if component in ("", "."):
            continue
        if component == "..":
            return "[BLOCKED] path traversal is not allowed"
        current = os.path.join(current, component)
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            # Nothing below a missing component can already be a symlink.
            break
        except OSError as exc:
            return f"[BLOCKED] cannot safely inspect path component: {exc}"
        if os.path.islink(current):
            return f"[BLOCKED] symlink path components are not allowed: {current}"
    real = os.path.realpath(lexical)
    if not any(_is_within(real, allowed) for allowed in roots):
        return f"[BLOCKED] resolved path escapes authorized folders: {lexical}"
    return None


def validate_write_target(path: str) -> tuple[str, str | None]:
    """Validate a path against shared edit/Bash roots without following symlinks."""
    resolved = os.path.abspath(resolve_path(path))
    roots = _authorized_write_roots()
    error = _lexical_symlink_error(resolved, roots)
    if error:
        return resolved, error
    real = os.path.realpath(resolved)
    hit = _protected_hit(real) or _protected_hit(resolved)
    if hit:
        return resolved, f"[BLOCKED] protected path: {hit}"
    return resolved, None


def validate_approved_edit_folder(path: str) -> str:
    """Return a canonical user-selected folder or raise a safe validation error."""
    raw = str(path or "").strip()
    if not raw or not os.path.isabs(raw):
        raise ValueError("folder path must be absolute")
    if any(ord(char) < 32 for char in raw):
        raise ValueError("folder path contains control characters")
    resolved = os.path.realpath(raw)
    if _protected_hit(resolved):
        raise ValueError("folder is protected")
    if not os.path.isdir(resolved):
        raise ValueError("folder must be an existing directory")
    return resolved


def _in_approved_edit_folder(path: str) -> bool:
    from .edit_access import path_is_approved_for_edit

    return path_is_approved_for_edit(path)


def check_path(path: str) -> str | None:
    """Return a policy error for an unauthorized or protected in-place write.

    Resolve relative paths under Active Workspace (or the internal workspace),
    reject lexical symlink components, then verify the canonical target.
    """
    resolved = os.path.abspath(resolve_path(path))
    roots = _authorized_write_roots()
    symlink_error = _lexical_symlink_error(resolved, roots)
    if symlink_error:
        return symlink_error
    abs_path = os.path.realpath(resolved)
    hit = _protected_hit(abs_path)
    if hit:
        return f"[BLOCKED] protected path: {hit}"
    if os.getenv("V2_ALLOW_OUTSIDE") or _in_workspace(abs_path):
        return None
    if not os.path.exists(abs_path) or is_edited_copy(abs_path):
        return None
    return (
        "[BLOCKED] in-place write to an existing file outside workspace. "
        f"Outside the workspace only NEW files may be created; changes to '{abs_path}' "
        f"must go to a sibling working copy: {edited_copy_path(abs_path)}"
    )


def plan_write(
    path: str, *, allow_approved_edit: bool = False
) -> tuple[str, str | None, str | None]:
    """Plan one write and return ``(effective_path, error, note)``.

    Agent edit/write tools pass ``allow_approved_edit=True`` to use the shared
    internal/Active/Approved roots. Other legacy callers retain the create-only
    outside-workspace working-copy behavior.
    """
    resolved = resolve_path(path)
    if allow_approved_edit:
        resolved, error = validate_write_target(path)
        if error:
            return resolved, error, None
        from config import WORKSPACE
        note = None if _in_workspace(os.path.realpath(resolved)) else "approved edit folder"
        return resolved, None, note
    abs_path = os.path.realpath(resolved)
    hit = _protected_hit(abs_path)
    if hit:
        return resolved, f"[BLOCKED] protected path: {hit}", None
    if _in_workspace(abs_path):
        return resolved, None, None
    if os.getenv("V2_ALLOW_OUTSIDE"):
        # Preserve the documented operator-level legacy bypass. Normal UI/CLI
        # operation does not set it and therefore remains approval-gated below.
        return resolved, None, None
    if allow_approved_edit and _in_approved_edit_folder(abs_path):
        return resolved, None, "approved edit folder"
    if not os.getenv("V2_ALLOW_OUTSIDE"):
        if allow_approved_edit:
            return resolved, "[BLOCKED] edit target is outside workspace and outside Approved Edit Folders", None
        if not os.path.exists(abs_path) or is_edited_copy(abs_path):
            return resolved, None, None
        copy = edited_copy_path(resolved)
        hit = _protected_hit(os.path.realpath(copy))
        if hit:
            return copy, f"[BLOCKED] protected path: {hit}", None
        note = (
            "original file outside workspace is never modified in place — "
            f"changes were written to the working copy: {copy}"
        )
        return copy, None, note
    return resolved, None, None


def resolve_path(path: str) -> str:
    """Resolve WRITE path: absolute → as-is; relative → Active Workspace, else WORKSPACE."""
    from config import WORKSPACE
    p = os.path.expanduser(path)
    if os.path.isabs(p):
        return p
    from .edit_access import get_focus_folder
    root = get_focus_folder() or WORKSPACE
    return os.path.join(root, _strip_ws_prefix(p, root))


def resolve_read_path(path: str) -> str:
    """Resolve READ path: reads unrestricted except system paths; relative → Active Workspace or WORKSPACE
    realpath + _protected_hit on BOTH branches — relative `../` traversal (e.g. ../../etc/passwd)
    must hit the same protected-path guard as an absolute /etc/passwd."""
    from config import WORKSPACE
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        from .edit_access import get_focus_folder
        root = get_focus_folder() or WORKSPACE
        p = os.path.join(root, _strip_ws_prefix(p, root))
    hit = _protected_hit(os.path.realpath(p))
    if hit:
        raise PermissionError(f"[BLOCKED] protected path: {hit}")
    return p


def find_readable(path: str) -> str | None:
    """Like resolve_read_path, but also self-heals the one known-shape mistake
    that keeps recurring live: `bash`'s cwd is already WORKSPACE, but a command
    sometimes re-prefixes the workspace dirname onto its own relative output
    path anyway (`cp src workspace/x.png`) — landing the file one level too
    deep (WORKSPACE/workspace/x.png) instead of where a plain lookup expects
    it (WORKSPACE/x.png). If the plain resolution isn't a real file, also try
    one workspace-dirname level deeper before giving up. Returns the real path
    if EITHER resolves to an existing file, else None (never raises for a
    plain not-found — a PermissionError from a protected path still
    propagates, since that's a policy violation, not a location guess)."""
    from config import WORKSPACE
    primary = resolve_read_path(path)
    if os.path.isfile(primary):
        return primary
    if not os.path.isabs(os.path.expanduser(path)):
        ws_name = os.path.basename(WORKSPACE.rstrip("/\\"))
        fallback = os.path.join(WORKSPACE, ws_name, _strip_ws_prefix(path, WORKSPACE))
        if os.path.isfile(fallback):
            # unlike the primary branch above (resolve_read_path, which
            # realpath+_protected_hit-checks everything), this fallback guess
            # was only isfile-checked — a symlinked "workspace" dir INSIDE
            # workspace pointing elsewhere would let it silently confirm an
            # outside file as "found" here. This self-heal only ever guesses
            # a WORKSPACE-relative doubled-prefix location (the
            # `cp src workspace/x.png` cwd-mistake shape), so re-validate the
            # realpath the same way resolve_read_path does before trusting it.
            real_fallback = os.path.realpath(fallback)
            hit = _protected_hit(real_fallback)
            if hit:
                raise PermissionError(f"[BLOCKED] protected path: {hit}")
            ws_abs = os.path.realpath(WORKSPACE)
            if real_fallback == ws_abs or real_fallback.startswith(ws_abs + os.sep):
                return fallback
    return None
