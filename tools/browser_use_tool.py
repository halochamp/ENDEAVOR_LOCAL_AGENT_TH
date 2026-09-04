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
import signal
import subprocess
import sys
import time
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
    """Flock-guarded single-record JSON state — same pattern as bash_bg._Registry."""

    def __init__(self) -> None:
        self.path = _state_path()
        self._lock_path = self.path.with_suffix(".lock")

    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._lock_path, "w")
        fcntl.flock(handle, fcntl.LOCK_EX)
        return handle

    def load(self) -> dict | None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            rec = json.loads(raw)
        except Exception:
            return None
        return rec if isinstance(rec, dict) and rec.get("pid") else None

    def save(self, rec: dict) -> None:
        handle = self._locked()
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def clear(self) -> None:
        handle = self._locked()
        try:
            try:
                self.path.unlink()
            except OSError:
                pass
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


def _live(state: _State | None = None) -> dict | None:
    """The kept-open browser if it is genuinely reusable, else None (clearing the
    record on the way out so the next call launches fresh instead of erroring)."""
    state = state or _State()
    rec = state.load()
    if rec is None:
        return None
    pid = rec.get("pid")
    if not _is_alive(pid) or _ps_lstart(pid) != rec.get("start_sig"):
        state.clear()
        return None
    if not _cdp_responds(rec.get("cdp_url", "")):
        # Process still around but not serving CDP — nothing can drive it any more.
        _terminate(pid)
        state.clear()
        return None
    return rec


def _terminate(pid: int) -> bool:
    """SIGTERM, then SIGKILL after the grace period. We cannot delegate this to
    browser-use: its LocalBrowserWatchdog only kills the process it launched itself
    (`is_local and self._subprocess`, local_browser_watchdog.py), and a session
    reconnected by cdp_url is is_local=False — session.kill() there disconnects but
    leaves Chromium running forever (verified, not assumed)."""
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return not _is_alive(pid)
    deadline = time.time() + _TERM_GRACE_SECONDS
    while time.time() < deadline:
        if not _is_alive(pid):
            return True
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    time.sleep(0.3)
    return not _is_alive(pid)


# browser-use creates a throwaway profile dir under two different prefixes:
# `browser-use-user-data-dir-` (profile.py, the common case — observed in the live
# end-to-end) and `browseruse-tmp-` (local_browser_watchdog.py, the locked-default-dir
# fallback). Matching only the second one silently leaks a profile dir per kept-open
# session, since we bypass browser-use's own kill path.
_TEMP_PROFILE_MARKERS = ("browser-use-user-data-dir-", "browseruse-tmp-")


def _cleanup_temp_profile(user_data_dir: str | None) -> None:
    """Remove a throwaway profile dir left behind by a kept-open browser. Only ever
    touches a dir carrying one of browser-use's own temp markers."""
    path = str(user_data_dir or "")
    if path and any(marker in path for marker in _TEMP_PROFILE_MARKERS):
        shutil.rmtree(path, ignore_errors=True)


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
    """PID of the Chromium browser-use launched. Primary source is the local watchdog
    handle; the port scan is the fallback for when the watchdog has been detached."""
    wd = getattr(session, "_local_browser_watchdog", None)
    pid = getattr(wd, "browser_pid", None) if wd is not None else None
    if pid:
        return pid
    port = _cdp_port(getattr(session, "cdp_url", "") or "")
    if not port:
        return None
    try:
        # Pattern deliberately has no leading "--": pgrep would parse that as an option.
        out = subprocess.run(["pgrep", "-f", f"remote-debugging-port={port}"],
                             capture_output=True, text=True, timeout=5)
        pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None


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


@tool
def browser_use(url: str = "", task: str = "", user_query: str = "",
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

    if action == "close":
        # No page is fetched — deliberately outside the web budget counter.
        if not live:
            return "ไม่มี browser เปิดค้างอยู่ (ไม่ต้องปิด)"
        ok = _terminate(live["pid"])
        _cleanup_temp_profile(live.get("user_data_dir"))
        state.clear()
        return ("ปิด browser เรียบร้อย" if ok else
                f"[error] ปิด browser ไม่สำเร็จ — pid {live['pid']} ยังทำงานอยู่ กรุณาปิดหน้าต่างเอง")

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
        from config import MLX_BASE_URL, MODEL, API_KEY
    except ImportError as e:
        return f"[error] browser-use not installed: {e}"

    err = _wc_check_and_inc()   # นับเฉพาะ browser launch จริง — cache hit ไม่นับ (ตาม batch_browse)
    if err:
        return err
    _progress(f"{'reconnecting to open' if live else 'launching'} Chromium for: {(url or 'current page')[:60]}")
    t0 = time.time()
    captured: dict = {}

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
            llm = ChatOpenAI(
                base_url=MLX_BASE_URL,
                api_key=API_KEY,
                model=MODEL,
                temperature=0.1,
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
    if keep_open and not live:
        if captured.get("pid") and captured.get("cdp_url"):
            state.save({
                "cdp_url": captured["cdp_url"],
                "pid": captured["pid"],
                "start_sig": _ps_lstart(captured["pid"]),
                "user_data_dir": captured.get("user_data_dir") or "",
                "url": url,
                "task": task,
                "opened_at": _now_iso(),
            })
        else:
            recorded = False
            log.warning("browser_use(keep_open=True): could not capture browser pid/cdp_url — "
                        "the window may stay open with no way to close it from the agent")
    elif keep_open and live:
        # Keep the record pointing at where the window actually is, so action="status"
        # doesn't keep reporting the url it was first opened at.
        live["url"] = url or live.get("url", "")
        live["task"] = task
        state.save(live)
    elif not live:
        # One-shot run: browser-use has already killed the browser by now (Agent.close()
        # is awaited inside run()), but its own cleanup only matches `browseruse-tmp-`,
        # never the `browser-use-user-data-dir-` prefix profile.py actually uses — so
        # every one-shot call used to leave a profile dir behind.
        _cleanup_temp_profile(captured.get("user_data_dir"))

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
