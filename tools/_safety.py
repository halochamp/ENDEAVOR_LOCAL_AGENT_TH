# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""_safety.py — path guards (V2: simplified, WORKSPACE-scoped)

write: เฉพาะใน WORKSPACE (override ด้วย env V2_ALLOW_OUTSIDE)
read:  ทุกที่ ยกเว้น system paths
"""
from __future__ import annotations
import os

_PROTECTED_PATHS = [
    "/etc/", "/usr/", "/bin/", "/sbin/", "/lib/",
    "/System/", "/Library/", "/Applications/",
    os.path.expanduser("~/.ssh/"),
    os.path.expanduser("~/.aws/"),
    os.path.expanduser("~/.gnupg/"),
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


def check_path(path: str) -> str | None:
    """คืน error string ถ้า path อันตราย/เขียนนอก workspace, None ถ้าเขียนได้
    validate RESOLVED path (relative → WORKSPACE) ให้สอดคล้องกับ resolve_path()
    realpath ทั้งสองฝั่ง — กัน symlink ใน workspace ชี้ออกนอก และ /etc→/private/etc บน macOS"""
    from config import WORKSPACE
    abs_path = os.path.realpath(resolve_path(path))
    hit = _protected_hit(abs_path)
    if hit:
        return f"[BLOCKED] protected path: {hit}"
    if not os.getenv("V2_ALLOW_OUTSIDE"):
        ws_abs = os.path.realpath(WORKSPACE)
        if not (abs_path == ws_abs or abs_path.startswith(ws_abs + os.sep)):
            return f"[BLOCKED] write outside workspace. Only '{ws_abs}' is writable."
    return None


def resolve_path(path: str) -> str:
    """Resolve WRITE path: absolute → as-is (check_path guards), relative → WORKSPACE/path"""
    from config import WORKSPACE
    p = os.path.expanduser(path)
    if os.path.isabs(p):
        return p
    return os.path.join(WORKSPACE, _strip_ws_prefix(p, WORKSPACE))


def resolve_read_path(path: str) -> str:
    """Resolve READ path: reads unrestricted except system paths; relative → WORKSPACE/path
    realpath + _protected_hit on BOTH branches — relative `../` traversal (e.g. ../../etc/passwd)
    must hit the same protected-path guard as an absolute /etc/passwd."""
    from config import WORKSPACE
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(WORKSPACE, _strip_ws_prefix(p, WORKSPACE))
    hit = _protected_hit(os.path.realpath(p))
    if hit:
        raise PermissionError(f"[BLOCKED] protected path: {hit}")
    return p
