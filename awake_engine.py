# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Awake engine — host-level scheduler that fires agent turns from standing triggers.

Lives at the fork root (host layer, same tier as agent_server.py/runtime_common.py):
it schedules WHOLE graph invocations from timers/file/screen events and never
orchestrates inside a turn — the GRAPH-FIRST gate's "orchestration" applies to
in-turn logic, which stays in graph/. The registration interface the model uses
is `tools/awake.py`; this module only reads the registry those registrations
write, checks trigger conditions on a slow tick, and hands a synthetic user
message to whichever host (Telegram bot / agent_server) attached a callback.

Single-engine guarantee: both hosts can run at once, but only one engine may
serve one registry — ownership is a lockfile with pid+heartbeat; a second engine
starts dormant and takes over only when the heartbeat goes stale.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("awake_engine")

TICK_SECONDS = 30
CHECK_SPACING_SECONDS = 60          # min spacing between file/screen condition checks
MAX_CONSECUTIVE_ERRORS = 3
LOCK_STALE_SECONDS = 90             # heartbeat older than this → other engine may take over
# Daily times= trigger: skip-missed policy, not catch-up. A target time only
# fires if the engine observes it within this many minutes AFTER the target
# (wall-clock, local time) — e.g. Mac asleep 04:00-23:30 skips a 05:00 target
# entirely rather than firing it late at 23:30 when the engine wakes. Chosen
# well above TICK_SECONDS (30s) so a slow/busy tick never misses the window,
# well below same-day double-fire range.
_TIME_GRACE_MINUTES = 15
_VOLATILE_LINE = re.compile(r"\b\d{1,2}:\d{2}\b")  # clock text changes every minute — not a real change

# Fork-adaptive screen-fire hint: fires the "can act" branch whenever this fork
# ships tools/computer_use.py. TH has no computer_use.py, so a fired screen
# watcher always tells the model it can only notify, never click — the safety
# boundary a "can act" branch would need (destructive-action block, per-turn
# action cap) simply doesn't apply here because there is nothing to act with.
if (Path(__file__).resolve().parent / "tools" / "computer_use.py").exists():
    _SCREEN_ACT_HINT = ("(fork นี้มี computer tool — action ง่ายๆ ทีละขั้นทำได้ เช่น คลิกปุ่มที่มีข้อความชัดเจน "
                        "แล้วรายงานทุกครั้ง; งานหลายขั้น/หลาย dialog ให้แจ้ง user แทน)")
else:
    _SCREEN_ACT_HINT = ("(fork นี้คลิก/พิมพ์บนจอไม่ได้ — ถ้าการเปลี่ยนแปลงต้องการการกด ให้แจ้ง user "
                        "ผ่านคำตอบ + display notification แทน ห้ามแกล้งทำ)")


def registry_path() -> Path:
    from config import WORKSPACE
    return Path(WORKSPACE) / "watchers.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


class Registry:
    """Shared JSON registry; every writer holds the sidecar flock for the full read-modify-write."""

    def __init__(self, path: Path | None = None):
        self.path = path or registry_path()
        self._lock_path = self.path.with_suffix(".lock")

    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._lock_path, "w")
        fcntl.flock(handle, fcntl.LOCK_EX)
        return handle

    def load(self) -> list[dict]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return []  # normal case — no registry file yet (no watchers registered)
        try:
            return json.loads(raw).get("watchers", [])
        except Exception as exc:
            # A corrupt watchers.json (e.g. an iCloud conflict-copy mangling this
            # Desktop-synced file mid-write) must not silently look identical to
            # "no watchers" — the next mutate() would then persist an EMPTY
            # registry, permanently erasing every standing trigger with no trace
            # anywhere. Preserve the bad file for recovery and log the loss
            # instead of swallowing it.
            try:
                corrupt_path = self.path.with_suffix(self.path.suffix + ".corrupt")
                self.path.replace(corrupt_path)
                log.warning("awake: %s was corrupt (%s) — preserved as %s, starting with an empty registry",
                            self.path, exc, corrupt_path)
            except OSError:
                log.warning("awake: %s was corrupt (%s) and could not be preserved", self.path, exc)
            return []

    def mutate(self, fn: Callable[[list[dict]], Any]):
        """Run fn on the current watcher list under the lock; persist; return fn's result."""
        handle = self._locked()
        try:
            watchers = self.load()
            result = fn(watchers)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"watchers": watchers}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            os.replace(tmp, self.path)
            return result
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


# ── Notice channel (one-directional FYI: telegram → ui_max) ──────────────────
# Telegram always wins ownership (see _owns' priority preemption above), so a
# watcher registered from ui_max still fires and delivers through Telegram.
# This tiny flock-protected sidecar lets ui_max's dormant engine tell the user
# "this fired, check Telegram" instead of staying silent with no explanation.
# Single consumer (ui_max) — post appends, drain reads-then-clears, no
# persistent last-seen cursor needed.
_NOTICE_MAX_ENTRIES = 20


def notice_path() -> Path:
    from config import WORKSPACE
    return Path(WORKSPACE) / "awake_notice.json"


def post_notice(watch_id: str, snippet: str, via: str) -> None:
    """Append one FYI entry. Best-effort: a failed notice must never break
    the real delivery it's reporting on."""
    path = notice_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path.with_suffix(".lock"), "w")
    except OSError:
        return
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except Exception:
            entries = []
        entries.append({"id": watch_id, "snippet": snippet, "via": via, "ts": _now_iso()})
        entries = entries[-_NOTICE_MAX_ENTRIES:]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def drain_notices() -> list[dict]:
    """Read and clear all pending FYI entries — safe without a last-seen
    cursor because there is exactly one consumer (ui_max's poll loop)."""
    path = notice_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path.with_suffix(".lock"), "w")
    except OSError:
        return []
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except Exception:
            entries = []
        if entries:
            try:
                path.unlink()
            except OSError:
                pass
        return entries
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def restore_notices(entries: list[dict]) -> None:
    """Restore an undelivered drained batch ahead of notices posted later."""
    pending = [entry for entry in entries if isinstance(entry, dict)]
    if not pending:
        return
    path = notice_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path.with_suffix(".lock"), "w")
    except OSError:
        return
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            if not isinstance(current, list):
                current = []
        except Exception:
            current = []
        combined = (pending + current)[:_NOTICE_MAX_ENTRIES]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


# ── Trigger condition seams (monkeypatchable in T0 tests) ─────────────────────

def _stat_mtime(path: str) -> float | None:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def _screen_ocr_text() -> str:
    """Full-screen OCR text via the same primitives read_image uses. Raises on failure."""
    from tools._ocr import read_layout
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = handle.name
    try:
        result = subprocess.run(["screencapture", "-x", "-m", path],
                                capture_output=True, timeout=10)
        if result.returncode:
            raise RuntimeError("screencapture failed — Screen Recording permission?")
        boxes = read_layout(path)
        return "\n".join(str(b.get("text", "")) for b in boxes if b.get("text"))
    finally:
        Path(path).unlink(missing_ok=True)


def _screen_hash(text: str) -> str:
    stable = "\n".join(line for line in text.splitlines()
                       if line.strip() and not _VOLATILE_LINE.search(line))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _owner_pid_alive(owner: str) -> bool:
    """Conservatively check the local PID encoded in an ownership token."""
    try:
        pid = int(owner.rsplit(":", 1)[1])
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, IndexError):
        return True


class AwakeEngine:
    """Ticks in a daemon thread; on trigger, calls fire(watch_id, message) on the host.

    The host's fire callback must route the message through its NORMAL turn queue
    (same single-worker path as user messages) so an awake turn never runs an LLM
    in parallel with a user turn, and its answer is delivered like any reply.
    """

    def __init__(self, fire: Callable[[str, str], None], registry: Registry | None = None,
                 owner: str = "engine"):
        self._fire = fire
        self.registry = registry or Registry()
        # Telegram is the always-on surface (phone, long-running process) —
        # it preempts any other host's ownership so awake deliveries land on
        # a channel the user can actually reach, instead of "whoever ticked
        # first wins" leaving results stuck in a closed ui_max window.
        self._priority = owner == "telegram"
        self.owner = f"{owner}:{os.getpid()}"
        self._lockfile = self.registry.path.with_suffix(".engine")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_check: dict[str, float] = {}   # watcher id → monotonic-ish last condition check

    # ── ownership ──
    def _owns(self) -> bool:
        """Claim or keep engine ownership; heartbeat while owning.

        The read-decide-write is done under an flock on the lockfile itself —
        without it, this was a plain read-then-write: after the Mac wakes from
        sleep with both a Telegram-bot process and an agent_server process
        running, both could read the same stale heartbeat within the same tick
        window, both conclude "the other one is dead", and both write themselves
        in as owner — ticking (and firing due watchers) concurrently. Two
        engines calling this at the same instant now simply serialize on the
        flock; only one sees the fresh heartbeat it just wrote and the other
        correctly backs off.

        Priority preemption: a `telegram`-owned engine claims regardless of
        the current owner's freshness, as long as that owner isn't ALSO
        telegram (a different pid of the same bot restarting shouldn't fight
        itself) — this is the one asymmetry, everything else stays the
        original first-alive-wins/stale-takeover behavior."""
        now = time.time()
        self._lockfile.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = open(self._lockfile, "a+")
        except OSError:
            return False
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if self._stop.is_set():
                return False
            handle.seek(0)
            raw = handle.read()
            try:
                data = json.loads(raw) if raw else None
            except Exception:
                data = None
            if data and data.get("owner") != self.owner:
                other_owner = str(data.get("owner", ""))
                other_is_telegram = other_owner.startswith("telegram:")
                dead_telegram_peer = other_is_telegram and not _owner_pid_alive(other_owner)
                fresh = now - data.get("beat", 0) < LOCK_STALE_SECONDS
                if fresh and not (
                    self._priority and (not other_is_telegram or dead_telegram_peer)
                ):
                    return False  # someone else is alive (and outranks or ties me)
            try:
                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps({"owner": self.owner, "beat": now}))
                handle.flush()
                return True
            except OSError:
                return False
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    # ── lifecycle ──
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._owns()
        self._thread = threading.Thread(target=self._run, daemon=True, name="awake-engine")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._release_ownership()

    def _release_ownership(self) -> None:
        """Clear the heartbeat only when this exact engine still owns it."""
        try:
            handle = open(self._lockfile, "a+")
        except OSError:
            return
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            try:
                data = json.loads(handle.read() or "null")
            except Exception:
                data = None
            if data and data.get("owner") == self.owner:
                handle.seek(0)
                handle.truncate()
                handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._owns():
                    self.tick()
            except Exception:
                pass  # a broken tick must never kill the host
            if self._stop.wait(TICK_SECONDS):
                break

    # ── one tick (public for tests) ──
    def tick(self, now: float | None = None) -> list[str]:
        """Check every active watcher once; fire due ones. Returns fired ids."""
        now = time.time() if now is None else now
        fired: list[str] = []
        for watcher in self.registry.load():
            if watcher.get("paused"):
                continue
            wid = watcher["id"]
            try:
                message = self._check(watcher, now)
            except Exception as exc:
                self._record_error(wid, str(exc))
                continue
            if message is not None:
                one_shot = watcher.get("kind") == "once"
                if one_shot and not self._consume_once(wid):
                    continue
                fired.append(wid)
                if not one_shot:
                    self._record_fire(wid, watcher)
                try:
                    self._fire(wid, message)
                except Exception as exc:
                    if one_shot:
                        log.warning("awake: consumed one-shot %s but fire callback failed: %s", wid, exc)
                    else:
                        self._record_error(wid, f"fire callback failed: {exc}")
        return fired

    def _check(self, watcher: dict, now: float) -> str | None:
        kind, wid, task = watcher["kind"], watcher["id"], watcher["task"]
        if kind == "once":
            run_at = _parse_iso(watcher.get("state", {}).get("run_at"))
            if run_at <= 0:
                raise ValueError("once watcher has invalid state.run_at")
            if now >= run_at:
                label = datetime.fromtimestamp(run_at).astimezone().strftime("%Y-%m-%d %H:%M")
                return f"[AWAKE:{wid}] (ครั้งเดียวตามนัด {label}) {task}"
            return None
        if kind == "every":
            times = watcher.get("times") or []
            if times:
                return self._check_times(watcher, now)
            due = _parse_iso(watcher.get("last_fire") or watcher.get("created")) \
                + watcher.get("interval_minutes", 60) * 60
            if now >= due:
                return f"[AWAKE:{wid}] {task}"
            return None
        # file/screen checks are rate-limited independently of the 30s tick
        if now - self._last_check.get(wid, 0.0) < CHECK_SPACING_SECONDS:
            return None
        self._last_check[wid] = now
        if kind == "file":
            mtime = _stat_mtime(watcher["target"])
            known = watcher.get("state", {}).get("mtime")
            if known is None:
                self._record_state(wid, {"mtime": mtime})
                return None  # first observation = baseline, not a change
            if mtime != known:
                self._record_state(wid, {"mtime": mtime})
                note = "ไฟล์ถูกลบ" if mtime is None else "ไฟล์มีการเปลี่ยนแปลง"
                return (f"[AWAKE:{wid}] {note}: {watcher['target']} → {task}\n"
                        f"(อ่านเนื้อหาล่าสุดด้วย read_file ก่อนทำงาน)")
            return None
        if kind == "screen":
            text = _screen_ocr_text()
            digest = _screen_hash(text)
            known = watcher.get("state", {}).get("ocr_hash")
            self._record_state(wid, {"ocr_hash": digest})
            if known is None or digest == known:
                return None
            snippet = text.strip()[:300]
            return (f"[AWAKE:{wid}] หน้าจอมีการเปลี่ยนแปลง → {task}\n"
                    f"(OCR ล่าสุดบางส่วน: {snippet})\n{_SCREEN_ACT_HINT}")
        raise ValueError(f"unknown watcher kind: {kind}")

    def _check_times(self, watcher: dict, now: float) -> str | None:
        """Daily clock-time trigger (times=). Skip-missed, not catch-up: a target
        only fires within _TIME_GRACE_MINUTES after it — if the engine was down
        across the whole window (host asleep/closed), that occurrence is skipped
        entirely rather than firing stale hours later. Per-slot "already fired
        today" tracked in state.fired_dates ({"23:00": "2026-07-19", ...}) so a
        multi-time watcher fires each of its times once per day, independently."""
        wid, task = watcher["id"], watcher["task"]
        times = watcher.get("times") or []
        local = datetime.fromtimestamp(now)
        today = local.date().isoformat()
        current_minute = local.hour * 60 + local.minute
        fired_dates: dict = watcher.get("state", {}).get("fired_dates", {})
        for t in times:
            if fired_dates.get(t) == today:
                continue
            hour_s, minute_s = t.split(":")
            target_minute = int(hour_s) * 60 + int(minute_s)
            delta = current_minute - target_minute
            if 0 <= delta <= _TIME_GRACE_MINUTES:
                self._record_state(wid, {"fired_dates": {**fired_dates, t: today}})
                return f"[AWAKE:{wid}] (เวลา {t}) {task}"
        return None

    # ── state updates (each under the registry lock) ──
    def _consume_once(self, wid: str) -> bool:
        """Atomically remove one due one-shot before dispatch; only one engine can win."""
        consumed = [False]

        def update(watchers):
            for index, watcher in enumerate(watchers):
                if watcher.get("id") == wid and watcher.get("kind") == "once" and not watcher.get("paused"):
                    del watchers[index]
                    consumed[0] = True
                    break

        self.registry.mutate(update)
        return consumed[0]

    def _record_state(self, wid: str, state: dict) -> None:
        def update(watchers):
            for w in watchers:
                if w["id"] == wid:
                    w.setdefault("state", {}).update(state)
        self.registry.mutate(update)

    def _record_fire(self, wid: str, watcher: dict) -> None:
        def update(watchers):
            for w in watchers:
                if w["id"] == wid:
                    w["last_fire"] = _now_iso()
                    w["consecutive_errors"] = 0
        self.registry.mutate(update)

    def _record_error(self, wid: str, error: str) -> None:
        paused_task = [None]

        def update(watchers):
            for w in watchers:
                if w["id"] == wid:
                    w["consecutive_errors"] = w.get("consecutive_errors", 0) + 1
                    w["last_error"] = error[:200]
                    if w["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
                        w["paused"] = True
                        paused_task[0] = w["task"]
        self.registry.mutate(update)
        if paused_task[0] is not None:
            try:
                self._fire(wid, f"[AWAKE:{wid}] watcher นี้ error ติดกัน {MAX_CONSECUTIVE_ERRORS} ครั้ง "
                                f"ถูกพักอัตโนมัติแล้ว (งาน: {paused_task[0]}) — แจ้ง user และถามว่าจะแก้/ลบไหม "
                                f"(error ล่าสุด: {error[:120]})")
            except Exception:
                pass
