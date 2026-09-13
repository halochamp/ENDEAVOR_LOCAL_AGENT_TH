# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""browser_use_tool.py — human-like web browsing via browser-use + Playwright

ใช้สำหรับ: เจาะลึกเว็บ, login, กรอก form, navigate หลายชั้น, JS-heavy ที่ Jina ทำไม่ได้
agent ระบุ URL + task เป็นภาษาธรรมชาติ — browser-use จัดการ click/scroll/type เอง

Two lifecycles, picked per call by `keep_open`:
  keep_open=False (default) — one-shot read. Browser closes when the task ends.
      Cache-aware: summarizes the result (query-aware if user_query provided) and
      caches it under url. Returns the compact `[web:<url>] <summary>` tag; use
      recall_web(url) for the full body.
  keep_open=True — the window stays open and running after the tool returns
      (music/video playing, a logged-in session, a half-filled form). Later calls
      reconnect to that same window over CDP instead of launching a second Chromium;
      only action="close" ends it. Action calls bypass the web cache in BOTH
      directions — a cached page summary must not stand in for "go press play", and
      an action log ("played song X") must not be written back as page content for url.

Cross-process state lives in workspace/browser_session.json (flock-guarded, same
pattern as bash_bg) — the CLI and any future long-running process are separate
processes and both must see the same open window.
"""
from __future__ import annotations
import asyncio
import fcntl
import json
import logging
import os
import random
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import web_cache
from tools.web_cache import web_count_check as _wc_check, web_count_check_and_inc as _wc_check_and_inc
from tools._summarize import summarize
from tools._progress import progress as _progress, phase as _phase

from config import BROWSER_USE_MAX_CHARS

log = logging.getLogger(__name__)

_TIMEOUT = int(os.getenv("V2_BROWSER_TIMEOUT", "300"))  # seconds — multi-step interactive tasks (click/type/retry) need more than a single page read
_ACTIONS = {"", "browse", "close", "status"}
_CDP_PROBE_TIMEOUT = 3.0
_TERM_GRACE_SECONDS = 5
_FRESH_PROFILE_PREFIX = "browser-use-user-data-dir-"

# Basic-stealth UA: browser-use's default headless launch args already disable the
# 'AutomationControlled' Blink feature (browser/profile.py), but the default UA on some
# Chromium builds still reveals a headless fingerprint — pin a normal desktop Chrome UA.
_STEALTH_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Same keyword set browser-use's own failure-reason check uses (agent/service.py) —
# we don't bypass these, just detect them so the tool can hand off to the user
# instead of burning the timeout retrying a wall it can't get through.
_CAPTCHA_KEYWORDS = ("captcha", "cloudflare", "recaptcha", "challenge", "bot detection", "access denied")


# ── Kept-open browser registry ────────────────────────────────────────────────

def _state_path() -> Path:
    from config import WORKSPACE
    return Path(WORKSPACE) / "browser_session.json"


class _State:
    """Cross-process browser ownership registry with a fail-closed quarantine."""

    def __init__(self) -> None:
        self.path = _state_path()
        self._lock_path = self.path.with_suffix(".lock")
        self._operation_lock_path = self.path.with_suffix(".operation.lock")
        self._quarantine_path = self.path.with_suffix(".quarantine")

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return {"pid": 0, "needs_close": True, "_state_blocked": True}
        try:
            rec = json.loads(raw)
        except Exception:
            return {"pid": 0, "needs_close": True, "_state_blocked": True}
        if not isinstance(rec, dict):
            return {"pid": 0, "needs_close": True, "_state_blocked": True}
        pid = rec.get("pid")
        valid_owned = isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
        valid_blocked = pid == 0 and rec.get("needs_close") is True and rec.get("_state_blocked") is True
        return rec if valid_owned or valid_blocked else {
            "pid": 0, "needs_close": True, "_state_blocked": True,
        }

    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._lock_path, "w")
        fcntl.flock(handle, fcntl.LOCK_EX)
        return handle

    def load(self) -> dict | None:
        quarantine = self._read_json(self._quarantine_path)
        if quarantine is not None:
            return quarantine
        return self._read_json(self.path)

    @staticmethod
    def _atomic_write(path: Path, rec: dict) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(rec, stream, ensure_ascii=False, indent=1)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
            os.chmod(path, 0o600)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def save(self, rec: dict) -> None:
        handle = self._locked()
        try:
            self._atomic_write(self.path, rec)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def save_reusable(self, rec: dict) -> bool:
        handle = self._locked()
        try:
            self._atomic_write(self.path, rec)
            try:
                self._quarantine_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return False
            return not self._quarantine_path.exists()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def save_quarantine(self, rec: dict) -> None:
        handle = self._locked()
        try:
            self._atomic_write(self._quarantine_path, rec)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def try_operation(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._operation_lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
        handle = os.fdopen(fd, "w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        return handle

    @staticmethod
    def release_operation(handle) -> None:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()

    def clear(self) -> bool:
        handle = self._locked()
        ok = True
        try:
            for path in (self.path, self._quarantine_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    ok = False
            for path in (self.path, self._quarantine_path):
                try:
                    if path.exists() or path.is_symlink():
                        ok = False
                except OSError:
                    ok = False
            return ok
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _ps_lstart(pid: int) -> str | None:
    """Process start-time signature (`ps -o lstart=`) — same guard bash_bg uses: pid
    liveness alone can't tell 'the Chromium we launched' from 'the OS recycled that pid'
    once the launching process is gone. None if the pid has no process."""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _cdp_port(cdp_url: str) -> int | None:
    try:
        return urlparse(cdp_url).port
    except Exception:
        return None


def _cdp_responds(cdp_url: str) -> bool:
    """Probe the CDP HTTP endpoint. A pid that is alive but no longer serving CDP (user
    closed the last window, Chromium wedged) must not be handed back as reusable —
    otherwise one stale record fails every later browser_use call. Retried once: a false
    negative here costs the user their open window (see _live), so a single transient
    blip on a loopback request must not be enough to condemn a healthy browser."""
    port = _cdp_port(cdp_url)
    if not port:
        return False
    import urllib.request
    for attempt in range(2):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version",
                                        timeout=_CDP_PROBE_TIMEOUT) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        if attempt == 0:
            time.sleep(0.5)
    return False


# browser-use creates throwaway profiles under these two prefixes.
_TEMP_PROFILE_MARKERS = ("browser-use-user-data-dir-", "browseruse-tmp-")


def _cleanup_temp_profile(user_data_dir: str | None) -> bool:
    """Delete only a direct child of the OS temp dir owned by browser-use."""
    raw = str(user_data_dir or "").strip()
    if not raw:
        return True
    try:
        candidate = Path(raw).expanduser()
        if candidate.is_symlink():
            return False
        path = candidate.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if not path.is_relative_to(temp_root):
            return True
        if not any(candidate.name.startswith(marker) for marker in _TEMP_PROFILE_MARKERS):
            return True
        if path.parent != temp_root:
            return False
        safe_candidate = temp_root / path.name
        if safe_candidate.is_symlink() or safe_candidate.resolve() != path:
            return False
        if safe_candidate.exists() and not safe_candidate.is_dir():
            return False
    except (OSError, RuntimeError, ValueError):
        return False
    for _ in range(3):
        shutil.rmtree(safe_candidate, ignore_errors=True)
        time.sleep(0.1)
        if not safe_candidate.exists():
            return True
    return not safe_candidate.exists()


def _persist_quarantine(state: _State, rec: dict) -> bool:
    updated = dict(rec)
    updated["needs_close"] = True
    try:
        state.save(updated)
    except BaseException:
        log.warning("could not persist browser_use quarantine in main state", exc_info=True)
    try:
        state.save_quarantine(updated)
        return True
    except BaseException:
        log.warning("could not persist browser_use quarantine sidecar", exc_info=True)
        return False


def _reserve_lifecycle(state: _State, live: dict | None,
                       url: str, task: str, user_data_dir: str = "") -> bool:
    """Persist fail-closed ownership uncertainty before touching Chromium."""
    if live is not None:
        return _persist_quarantine(state, live)
    tombstone = {
        "pid": 0,
        "needs_close": True,
        "_state_blocked": True,
        "launch_in_progress": True,
        "url": url,
        "task": task,
        "user_data_dir": user_data_dir,
        "opened_at": _now_iso(),
    }
    try:
        state.save_quarantine(tombstone)
        return True
    except BaseException:
        log.warning("could not persist browser_use launch reservation", exc_info=True)
        return False


def _rollback_lifecycle_reservation(state: _State, live: dict | None) -> bool:
    try:
        return state.save_reusable(live) if live is not None else state.clear()
    except BaseException:
        log.warning("could not roll back browser_use lifecycle reservation", exc_info=True)
        return False


def _profile_browser_pids(user_data_dir: str | None = None) -> list[int] | None:
    """Find browser-use Chromium main processes by exact throwaway-profile argv."""
    raw = str(user_data_dir or "").strip()
    exact_profile_arg = f"--user-data-dir={raw}" if raw else ""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        candidates: list[int] = []
        for line in out.stdout.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) != 2:
                continue
            try:
                pid = int(fields[0])
                argv = shlex.split(fields[1])
            except (ValueError, TypeError):
                continue
            if pid == os.getpid():
                continue
            if not any(arg.startswith("--remote-debugging-port=") and arg.split("=", 1)[1]
                       for arg in argv):
                continue
            profile_args = [arg for arg in argv if arg.startswith("--user-data-dir=")]
            if exact_profile_arg:
                if exact_profile_arg not in profile_args:
                    continue
            else:
                if not any(
                    any(Path(arg.split("=", 1)[1]).name.startswith(marker)
                        for marker in _TEMP_PROFILE_MARKERS)
                    for arg in profile_args if arg.split("=", 1)[1]
                ):
                    continue
            candidates.append(pid)
        return sorted(set(candidates))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _recover_blocked_lifecycle_state(state: _State, rec: dict) -> dict | None:
    """Recover only when exact profile/process evidence makes ownership deterministic."""
    if not rec.get("_state_blocked"):
        return rec
    pid = rec.get("pid")
    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
        return rec
    profile = str(rec.get("user_data_dir") or "").strip()
    if not profile and not rec.get("launch_in_progress"):
        return rec
    candidates = _profile_browser_pids(profile or None)
    if candidates is None:
        return rec
    if profile:
        if len(candidates) > 1:
            return rec
        if len(candidates) == 1:
            recovered_pid = candidates[0]
            start_sig = _ps_lstart(recovered_pid)
            if not start_sig:
                return rec
            recovered = {
                key: value for key, value in rec.items()
                if key not in {"_state_blocked", "launch_in_progress"}
            }
            recovered.update({
                "pid": recovered_pid,
                "start_sig": start_sig,
                "needs_close": True,
                "user_data_dir": profile,
            })
            return recovered if _persist_quarantine(state, recovered) else rec
        if not _cleanup_temp_profile(profile):
            return rec
    elif candidates:
        return rec
    return None if state.clear() else rec


def _live(state: _State | None = None) -> dict | None:
    """Return reusable ownership; blocked/uncertain state stays fail-closed."""
    state = state or _State()
    rec = state.load()
    if rec is None:
        return None
    if rec.get("_state_blocked"):
        rec = _recover_blocked_lifecycle_state(state, rec)
        if rec is None or rec.get("_state_blocked"):
            return rec
    pid = rec.get("pid")
    start_sig = rec.get("start_sig")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return rec
    if not _is_alive(pid) or _ps_lstart(pid) != start_sig:
        _cleanup_temp_profile(rec.get("user_data_dir"))
        state.clear()
        return None
    if rec.get("needs_close"):
        return rec
    if not _cdp_responds(rec.get("cdp_url", "")):
        if _terminate(pid, start_sig):
            _cleanup_temp_profile(rec.get("user_data_dir"))
            state.clear()
            return None
        rec["needs_close"] = True
        _persist_quarantine(state, rec)
    return rec


def _terminate(pid: int, expected_start_sig: str | None = None) -> bool:
    """Stop only the process whose PID/start signature still matches ownership."""
    if expected_start_sig and _ps_lstart(pid) != expected_start_sig:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return not _is_alive(pid)
    deadline = time.time() + _TERM_GRACE_SECONDS
    while time.time() < deadline:
        if not _is_alive(pid):
            return True
        if expected_start_sig and _ps_lstart(pid) != expected_start_sig:
            return False
        time.sleep(0.1)
    if expected_start_sig and _ps_lstart(pid) != expected_start_sig:
        return False
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    time.sleep(0.3)
    return not _is_alive(pid)


def _result_text(result) -> str:
    """The agent's actual answer text, not `str(AgentHistoryList)`.

    str() on the history is a Python repr — `AgentHistoryList(all_results=[ActionResult(
    is_done=False, ...)], ...)`. Under SUMMARY_SKIP_LLM_BELOW (1500) summarize() passes
    short content straight through, so a short run put that repr verbatim into MainState
    and into the URL's cached body. extracted_content() is the readable text of every
    step; final_result() is the agent's closing answer."""
    try:
        parts = [c.strip() for c in (result.extracted_content() or []) if c and c.strip()]
    except Exception:
        parts = []
    try:
        final = (result.final_result() or "").strip()
    except Exception:
        final = ""
    if final:
        # final_result() is also the last extracted_content — keep it once, at the end.
        parts = [p for p in parts if p != final] + [final]
    return "\n\n".join(parts).strip() or str(result)


def _browser_pid(session) -> int | None:
    """PID of the exact browser-use Chromium process, never a port-only guess."""
    wd = getattr(session, "_local_browser_watchdog", None)
    if wd is not None:
        for handle in (getattr(wd, "_subprocess", None), getattr(wd, "browser_pid", None)):
            pid = getattr(handle, "pid", handle)
            if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
                return pid
    profile = str(getattr(getattr(session, "browser_profile", None), "user_data_dir", "") or "")
    candidates = _profile_browser_pids(profile) if profile else None
    return candidates[0] if candidates is not None and len(candidates) == 1 else None


def _compose_task(url: str, task: str, reusing: bool) -> str:
    """Prompt handed to browser-use's own agent. When reusing a window the user asked to
    keep open, a new URL goes in a NEW TAB — go_to_url takes new_tab (tools/views.py), and
    navigating the current tab instead would stop whatever it is running (the playing
    video is the whole point of keep_open)."""
    if not url:
        return task
    if reusing:
        return (f"Open {url} in a NEW TAB — do not navigate away from or disturb the tab "
                f"that is already open. {task}")
    return f"Go to {url}. {task}"


def _describe(rec: dict) -> str:
    return (f"browser เปิดค้างอยู่ — {rec.get('url') or '(current page)'} "
            f"(pid {rec.get('pid')}, เปิดตั้งแต่ {rec.get('opened_at')})")


def _browser_use_impl(url: str = "", task: str = "", user_query: str = "",
                      keep_open: bool = False, action: str = "") -> str:
    """Browse a website like a human — can click, scroll, fill forms, navigate multiple pages.
    Use ONLY when: user explicitly asks to "open" or "browse" a specific site, need to login, need to interact with JS-heavy pages, or browse_url returns insufficient content.
    Returns a compact Thai summary tagged with the URL — full result is cached. Call recall_web(url) for full body.
    Args:
        url: starting URL — omit only to continue on the browser already left open
        task: what to do or find on the site (in natural language)
        user_query: the user's current question — used to produce a query-focused summary
        keep_open: True → window stays open and running after this call (music/video playing, session logged in); later calls drive that same window and only action="close" ends it. Default False → closes when the task ends.
        action: "" = do the task (default) · "close" = shut the open browser, run no task · "status" = report whether one is open
    ✅ browser_use(url="https://youtube.com", task="ค้นเพลง X แล้วกดเล่น", keep_open=True) → เพลงเล่นค้างไว้
    ✅ browser_use(task="กดเพลงถัดไป", keep_open=True) → สั่งงานหน้าต่างเดิมที่เปิดค้างอยู่
    ✅ browser_use(action="close") → ปิดหน้าต่างนั้น
    ❌ browser_use(url="https://x.com/article", task="อ่านเนื้อหา", keep_open=True) — งานอ่านจบในตัว ไม่ต้องค้าง"""
    action = (action or "").strip().lower()
    if action not in _ACTIONS:
        return f"[error] action must be one of: {', '.join(sorted(a for a in _ACTIONS if a))}"

    state = _State()
    live = _live(state)

    if action == "status":
        return _describe(live) if live else "ไม่มี browser เปิดค้างอยู่"

    if live and live.get("_state_blocked"):
        return ("[error] lifecycle registry ของ browser ยังยืนยัน ownership ไม่ได้ — "
                "ห้ามเริ่ม/reconnect และห้ามใช้ pkill/kill Chrome เพราะอาจปิด browser ของผู้ใช้ผิดตัว; "
                "ตรวจสถานะด้วย browser_use(action=\"status\") และปิดหน้าต่าง browser-use ที่ค้างเองถ้ายังมีอยู่")

    if action == "close":
        # No page is fetched — deliberately outside the web budget counter.
        if not live:
            return "ไม่มี browser เปิดค้างอยู่ (ไม่ต้องปิด)"
        pid = live.get("pid")
        start_sig = live.get("start_sig")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return "[error] ปิด browser ไม่ได้อย่างปลอดภัย — registry ไม่มี process identity ที่ยืนยันได้"
        if not _is_alive(pid) or _ps_lstart(pid) != start_sig:
            _cleanup_temp_profile(live.get("user_data_dir"))
            return "ปิด browser เรียบร้อย" if state.clear() else "[error] browser ปิดแล้วแต่ล้าง registry ไม่สำเร็จ"
        ok = _terminate(pid, start_sig)
        if not ok:
            _persist_quarantine(state, live)
            return (f"[error] ปิด browser ไม่สำเร็จ — pid {pid} ยังทำงานอยู่หรือ identity เปลี่ยน "
                    "สถานะถูกเก็บไว้ กรุณาปิดหน้าต่างเอง")
        profile_ok = _cleanup_temp_profile(live.get("user_data_dir"))
        cleared = state.clear() if profile_ok else False
        return ("ปิด browser เรียบร้อย" if cleared else
                "[error] browser หยุดแล้ว แต่ลบ profile/registry ชั่วคราวไม่สำเร็จ")

    if live and live.get("needs_close"):
        return ("[error] browser session อยู่ในสถานะไม่แน่ชัดจากการหยุดงานครั้งก่อน — "
                "กรุณาสั่ง browser_use(action=\"close\") ก่อนเริ่มงานใหม่")

    url = (url or "").strip()
    if not url and not live:
        return "[error] url is required (no browser is currently open to continue on)"

    # Cache is bypassed entirely for keep_open calls: a cached summary answering
    # "what's on this page" must never stand in for "go to this page and press play",
    # and the resulting action log must never be written back as this url's content.
    if not keep_open:
        # Checked before the budget gate — a cache hit stays free even when the turn's
        # web budget is exhausted.
        cached = web_cache.get(url) if url else None
        if cached is not None:
            _progress(f"cache HIT (skip browser launch): {url[:60]}")
            from config import SUMMARY_MAX_CHARS
            effective_uq_read = (user_query or "").strip() or task
            cached_sum = (web_cache.get_summary(url, effective_uq_read)
                          or web_cache.get_summary(url, "")
                          or cached[:SUMMARY_MAX_CHARS])
            return f"[web:{url}] {cached_sum}"

    err = _wc_check()
    if err:
        return err
    _phase(f"🌐 {'ใช้ browser เดิม' if live else 'เปิด browser'}: {(url or 'หน้าปัจจุบัน')[:42]}")
    try:
        from browser_use import Agent as BUAgent, BrowserProfile
        from browser_use.browser.session import BrowserSession
        from browser_use.llm import ChatOpenAI  # browser-use 0.12+ dropped langchain — needs its own BaseChatModel (has .provider)
        from config import get_mlx_base_url, API_KEY, get_model
    except ImportError as e:
        return f"[error] browser-use not installed: {e}"

    fresh_profile = (
        "" if live else
        str(Path(tempfile.gettempdir()) / f"{_FRESH_PROFILE_PREFIX}{uuid.uuid4().hex}")
    )
    if not _reserve_lifecycle(state, live, url, task, user_data_dir=fresh_profile):
        return ("[error] browser_use ไม่สามารถบันทึก lifecycle reservation ได้อย่างปลอดภัย — "
                "ไม่ได้เปิดหรือ reconnect browser ใหม่ กรุณาตรวจสอบสิทธิ์/พื้นที่ของ registry แล้วลองใหม่")

    err = _wc_check_and_inc()   # นับเฉพาะ browser launch จริง — cache hit ไม่นับ (ตาม batch_browse)
    if err:
        if not _rollback_lifecycle_reservation(state, live):
            err += " (lifecycle reservation ถูกเก็บไว้เพื่อป้องกันการเปิด browser ซ้ำ)"
        return err
    _progress(f"{'reconnecting to open' if live else 'launching'} Chromium for: {(url or 'current page')[:60]}")
    t0 = time.time()
    captured: dict = ({"user_data_dir": fresh_profile} if fresh_profile else {})

    def _on_step(browser_state_summary, model_output, step_number) -> None:
        # browser_use calls this right after each step's LLM output, before the
        # action executes — surface it via _progress() so the spinner sub-status
        # shows live step-by-step activity during the (up to 5min) wait.
        try:
            goal = (model_output.next_goal or "").strip()
            actions = model_output.action or []
            action_names = ", ".join(
                next(iter(a.model_dump(exclude_unset=True)), "?") for a in actions
            )
            msg = f"[step {step_number}] {goal}"
            if action_names:
                msg += f" → {action_names}"
            _progress(msg[:150])
        except Exception:
            pass

    async def _run() -> str:
        try:
            browser_model = get_model()
            llm = ChatOpenAI(
                base_url=get_mlx_base_url(),
                api_key=API_KEY,
                model=browser_model,
                temperature=0.1,
                reasoning_effort="none",
                reasoning_models=[browser_model],
                # browser-use 0.12.9 no longer accepts LangChain's extra_body;
                # use its native reasoning control so browser action selection stays no-think.
                # Keep browser-use on its existing prompt-schema path: its parser does no repair,
                # so this integration must not rely on response_format grammar enforcement from
                # the local model server. The schema is supplied in the system prompt instead.
                dont_force_structured_output=True,
                add_schema_to_system_prompt=True,
            )
            if live:
                # Reconnect over CDP to the window already open. keep_alive=True is
                # mandatory here, not cosmetic: Agent.close() reads it to decide whether
                # to kill the browser, and a window the user asked to keep must survive
                # even a keep_open=False read that happens to reuse it.
                session = BrowserSession(cdp_url=live["cdp_url"], keep_alive=True)
            else:
                session = BrowserSession(browser_profile=BrowserProfile(
                    user_agent=_STEALTH_USER_AGENT,
                    user_data_dir=fresh_profile,
                    # randomized per-call, human-like pacing between actions (default is a
                    # robotic fixed 0.1s) — reduces timing-based bot-detection heuristics
                    wait_between_actions=random.uniform(0.4, 1.1),
                    keep_alive=keep_open,
                ))
            agent = BUAgent(
                task=_compose_task(url, task, reusing=bool(live)),
                llm=llm,
                use_vision=False,  # Browser-use remains text-only by this tool's product contract
                use_judge=False,  # post-task self-critique LLM call — not needed, and its extra latency was blowing the timeout after the real result was already computed
                register_new_step_callback=_on_step,
                browser_session=session,
            )
            try:
                result = await asyncio.wait_for(agent.run(), timeout=_TIMEOUT)
            finally:
                # Captured even on timeout/failure: a keep_alive browser that launched
                # before the task blew up is still running, and an unrecorded pid is an
                # orphan nothing can close. Captured on the one-shot path too, for the
                # throwaway-profile cleanup below.
                if not live:
                    captured["cdp_url"] = getattr(session, "cdp_url", None)
                    captured["pid"] = _browser_pid(session)
                    captured["user_data_dir"] = str(
                        getattr(session.browser_profile, "user_data_dir", "") or "")
            if result.is_successful() is not True:
                blocked_text = " ".join(filter(None, [result.final_result(), *result.errors()])).lower()
                if any(kw in blocked_text for kw in _CAPTCHA_KEYWORDS):
                    return (
                        f"[error] เว็บ {url} บล็อกด้วย CAPTCHA/bot-detection — "
                        "ไม่สามารถ bypass ได้ กรุณาเปิดเบราว์เซอร์แก้ captcha ด้วยตัวเอง แล้วลองถามใหม่อีกครั้ง"
                    )
            text = _result_text(result)
            if len(text) > BROWSER_USE_MAX_CHARS:
                text = text[:BROWSER_USE_MAX_CHARS] + f"\n...[truncated at {BROWSER_USE_MAX_CHARS} chars]"
            return text or "(no result)"
        except asyncio.TimeoutError:
            return f"[error] browser_use timed out after {_TIMEOUT}s"
        except Exception as e:
            return f"[error] browser_use failed: {e}"

    try:
        raw = asyncio.run(_run())
        _progress(f"browser done in {time.time()-t0:.1f}s ({len(raw)} chars)")
    except Exception as e:
        _progress(f"browser failed: {e}")
        raw = f"[error] browser_use runner failed: {e}"

    recorded = True
    run_failed = raw.startswith("[error]")
    if keep_open and not live and not run_failed:
        pid = captured.get("pid")
        cdp_url = captured.get("cdp_url")
        start_sig = _ps_lstart(pid) if isinstance(pid, int) and pid > 0 else None
        if pid and cdp_url and start_sig:
            recorded = state.save_reusable({
                "cdp_url": cdp_url,
                "pid": pid,
                "start_sig": start_sig,
                "user_data_dir": captured.get("user_data_dir") or fresh_profile,
                "url": url,
                "task": task,
                "opened_at": _now_iso(),
            })
        else:
            recorded = False
            current = state.load()
            if current is not None:
                _recover_blocked_lifecycle_state(state, current)
            log.warning("browser_use(keep_open=True): browser identity was not reusable; lifecycle remains guarded")
    elif keep_open and live and not run_failed:
        # Publish reusable state only after the task completed successfully.
        updated = dict(live)
        updated.pop("needs_close", None)
        updated["url"] = url or live.get("url", "")
        updated["task"] = task
        recorded = state.save_reusable(updated)
    elif live and not keep_open and not run_failed:
        # One-shot use of an existing kept-open browser must honor keep_open=False.
        pid = live.get("pid")
        start_sig = live.get("start_sig")
        stopped = isinstance(pid, int) and pid > 0 and _terminate(pid, start_sig)
        cleaned = _cleanup_temp_profile(live.get("user_data_dir")) if stopped else False
        if not (stopped and cleaned and state.clear()):
            recorded = False
            _persist_quarantine(state, live)
    else:
        # Fresh one-shot success/error, or any failed run: resolve the exact-profile
        # reservation deterministically. No Chromium -> cleanup+clear; one exact
        # Chromium -> close-only quarantine; ambiguity remains fail-closed.
        current = state.load()
        if current is not None:
            remaining = _recover_blocked_lifecycle_state(state, current)
            if remaining is not None:
                recorded = False

    if raw.startswith("[error]"):
        return raw  # don't cache errors

    if keep_open:
        # No web_cache write — see the module docstring. The status line tells the model
        # the window is still live and reusable without re-reading the state file. When
        # the pid could not be recorded, say so instead: promising action="close" that
        # would answer "nothing is open" is the one way this design orphans a browser.
        note = summarize(raw, url=url or "browser", user_query=(user_query or "").strip() or task)
        tail = ("(browser ยังเปิดค้างอยู่ สั่งงานต่อได้ด้วย browser_use อีกครั้ง / ปิดด้วย action=\"close\")"
                if recorded else
                "(⚠️ browser เปิดค้างอยู่ แต่ agent บันทึก process ไม่ได้ — สั่งต่อหรือปิดจาก agent ไม่ได้ "
                "กรุณาปิดหน้าต่างเอง)")
        return f"[browser:open] {url or 'หน้าปัจจุบัน'} — {note}\n{tail}"

    # Use task as supplemental context if user_query is empty.
    effective_uq = (user_query or "").strip() or task
    if not url:
        # Continued on the already-open window with no url of its own — there is no cache
        # key to file this under, and "" must never become one.
        return f"[browser] {summarize(raw, url='browser', user_query=effective_uq)}"
    raw_with_task = f"# Task: {task}\n# URL: {url}\n\n{raw}"
    summary = summarize(raw_with_task, url=url, user_query=effective_uq)
    web_cache.put(url, raw_with_task)
    web_cache.put_summary(url, effective_uq, summary, raw=raw_with_task)
    return f"[web:{url}] {summary}"


def _browser_use_serialized(url: str = "", task: str = "", user_query: str = "",
                            keep_open: bool = False, action: str = "") -> str:
    if (action or "").strip().lower() == "status":
        return _browser_use_impl(url, task, user_query, keep_open, action)
    state = _State()
    operation = state.try_operation()
    if operation is None:
        return ("[error] browser_use is busy with another browser operation — "
                "wait for that operation to finish, then retry")
    try:
        return _browser_use_impl(url, task, user_query, keep_open, action)
    finally:
        state.release_operation(operation)


@tool
def browser_use(url: str = "", task: str = "", user_query: str = "",
                keep_open: bool = False, action: str = "") -> str:
    """Browse an interactive website: click, scroll, fill forms, navigate, or play media.

    Prefer browse_url for static reading. Use keep_open=True only when the browser must
    remain live for media or a follow-up; continue that session with another browser_use
    call and close it with action="close". action="status" is read-only.
    Never use pkill/kill Chrome to recover lifecycle state because that can close a
    browser the user owns.
    """
    return _browser_use_serialized(
        url=url, task=task, user_query=user_query, keep_open=keep_open, action=action,
    )
