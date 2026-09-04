# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Best-effort macOS Accessibility snapshot; OCR remains the fallback."""
from __future__ import annotations
import json, logging, subprocess, tempfile, threading
from pathlib import Path
log = logging.getLogger(__name__)
_SRC = Path(__file__).with_name("_accessibility.swift")
_BIN = Path(tempfile.gettempdir()) / "endeavor_th_accessibility"
_compiled: bool | None = None
_lock = threading.Lock()
def _ensure() -> bool:
    global _compiled
    if _compiled and _BIN.exists(): return True
    with _lock:
        if _compiled and _BIN.exists(): return True
        try:
            if not (_BIN.exists() and _BIN.stat().st_mtime >= _SRC.stat().st_mtime):
                subprocess.run(["swiftc", "-O", str(_SRC), "-o", str(_BIN), "-module-cache-path", str(Path(tempfile.gettempdir()) / "endeavor-swift-module-cache"), "-framework", "AppKit", "-framework", "ApplicationServices"], check=True, capture_output=True, timeout=120, stdin=subprocess.DEVNULL)
            _compiled = _BIN.exists()
        except Exception as exc:
            log.debug("AX helper unavailable: %s", exc); _compiled = False
    return bool(_compiled)
def read_frontmost(max_nodes: int = 220) -> dict:
    if not _ensure(): return {"status":"helper_unavailable", "elements":[]}
    try:
        r=subprocess.run([str(_BIN),str(max(1,min(int(max_nodes),500)))],capture_output=True,text=True,timeout=8,stdin=subprocess.DEVNULL)
        data=json.loads((r.stdout or "").strip())
        if not isinstance(data,dict): raise ValueError("non-object AX response")
        data.setdefault("status", "ok" if r.returncode == 0 else "helper_error"); data.setdefault("elements",[]); return data
    except subprocess.TimeoutExpired: return {"status":"timeout", "elements":[]}
    except Exception as exc: log.debug("AX read failed: %s",exc); return {"status":"helper_error", "elements":[]}

def insert_focused_text(text: str) -> dict:
    """Insert literal text at the focused non-secure AX text selection/caret.

    Input is sent over stdin so it never appears in process arguments or logs.
    This helper does not touch the clipboard.
    """
    if not _ensure():
        return {"status": "helper_unavailable"}
    try:
        r = subprocess.run(
            [str(_BIN), "--insert-focused"],
            input=text,
            capture_output=True,
            text=True,
            timeout=8,
        )
        data = json.loads((r.stdout or "").strip())
        if not isinstance(data, dict):
            raise ValueError("non-object AX insert response")
        data.setdefault("status", "ok" if r.returncode == 0 else "helper_error")
        return data
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as exc:
        log.debug("AX insert failed: %s", exc)
        return {"status": "helper_error"}
