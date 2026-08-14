# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""awake.py — registration interface for standing triggers (Agent Awake).

This tool ONLY writes/reads the shared registry via `awake_engine.Registry` — it
never schedules anything itself. The host-level `AwakeEngine` (awake_engine.py)
owns the tick loop and fires synthetic turns. Kept as a separate tool from
`tool_loop` (which runs N items NOW, same turn) — awake registers a trigger and
returns immediately; the real work happens in a FUTURE turn the engine starts.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.tools import tool

from awake_engine import Registry

# ── test/host seam ──────────────────────────────────────────────────────────
# None = use awake_engine.registry_path() (production workspace/watchers.json).
# Tests set this to a tempdir path so registration/list/stop never touch the
# real workspace file.
_REGISTRY_PATH: Path | None = None

_MAX_WATCHERS = 5
_MIN_EVERY_MINUTES = 5          # hard floor — engine also never checks faster than this implies
_COST_CONFIRM_MINUTES = 15      # docstring-only threshold; model must ask the user below this
_KIND_TH = {
    "file": "เฝ้าไฟล์",
    "every": "ทำซ้ำตามเวลา",
    "once": "ทำครั้งเดียว",
    "screen": "เฝ้าหน้าจอ",
}

# ── daily clock-time trigger (times=) ─────────────────────────────────────────
_MAX_TIMES = 3
_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_RUN_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}$")


def _parse_times(raw: str) -> tuple[list[str], str | None]:
    """Parse a comma-separated HH:MM list into normalized zero-padded times.
    Returns (times, error) — times is [] (not an error) when raw is blank, so
    callers can tell "no times= given" apart from "times= given but invalid"."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return [], None
    if len(parts) > _MAX_TIMES:
        return [], f"times รองรับสูงสุด {_MAX_TIMES} เวลาต่อการตั้งค่าเดียว (ได้ {len(parts)} เวลา)"
    normalized = []
    for p in parts:
        m = _TIME_RE.match(p)
        if not m:
            return [], f"รูปแบบเวลาไม่ถูกต้อง: {p!r} (ต้องเป็น HH:MM 24 ชม. เช่น 23:00)"
        normalized.append(f"{int(m.group(1)):02d}:{m.group(2)}")
    if len(set(normalized)) != len(normalized):
        return [], "times มีเวลาซ้ำกัน — แต่ละเวลาต้องไม่ซ้ำ"
    return sorted(normalized), None


def _parse_run_at(raw: str) -> tuple[str | None, str | None]:
    """Normalize one local wall-clock datetime to UTC ISO for persistence."""
    value = raw.strip()
    if not _RUN_AT_RE.match(value):
        return None, "run_at ต้องเป็น YYYY-MM-DD HH:MM เช่น 2026-08-07 23:00"
    try:
        local_naive = datetime.strptime(value.replace("T", " "), "%Y-%m-%d %H:%M")
    except ValueError:
        return None, f"run_at ไม่ใช่วันเวลาที่ถูกต้อง: {value!r}"
    local_tz = datetime.now().astimezone().tzinfo or timezone.utc
    run_at = local_naive.replace(tzinfo=local_tz).astimezone(timezone.utc)
    if run_at <= datetime.now(timezone.utc):
        return None, "run_at ต้องเป็นเวลาในอนาคต"
    return run_at.isoformat(), None


def _format_run_at(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return value or "(เวลาไม่ถูกต้อง)"


def _resolve_once_run_at(run_at: str, delay_minutes: int) -> tuple[str | None, str | None]:
    """Resolve exactly one absolute or relative one-shot schedule."""
    if run_at.strip() and delay_minutes:
        return None, "once ให้ระบุ run_at หรือ delay_minutes อย่างใดอย่างหนึ่งเท่านั้น"
    if run_at.strip():
        return _parse_run_at(run_at)
    if delay_minutes < 1:
        return None, "once ต้องระบุ run_at=YYYY-MM-DD HH:MM หรือ delay_minutes >= 1"
    return (datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)).isoformat(), None


def _registry() -> Registry:
    return Registry(path=_REGISTRY_PATH)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error(message: str) -> str:
    return f"[error] {message}"


def _new_id(existing_watchers: list[dict]) -> str:
    existing = {w.get("id") for w in existing_watchers}
    for _ in range(1000):
        wid = "w" + uuid.uuid4().hex[:4]
        if wid not in existing:
            return wid
    raise RuntimeError("could not generate a unique watcher id")


def _format_registration(record: dict) -> str:
    lines = [f"[ok] ลงทะเบียน watcher {record['id']} ({_KIND_TH.get(record['kind'], record['kind'])})"]
    if record["kind"] == "file":
        lines.append(f"target: {record['target']}")
    if record["kind"] == "every":
        if record.get("times"):
            lines.append(f"เวลา: {', '.join(record['times'])} (ทุกวัน)")
        else:
            lines.append(f"interval: ทุก {record['interval_minutes']} นาที")
    if record["kind"] == "once":
        lines.append(f"เวลา: {_format_run_at(record.get('state', {}).get('run_at', ''))} (ครั้งเดียว)")
    lines.append(f"task: {record['task']}")
    if record["kind"] == "once":
        lines.append("cost: fire เพียง 1 LLM turn แล้วลบ watcher อัตโนมัติ")
    else:
        lines.append(
            "cost: การ fire แต่ละครั้งคือ LLM turn เต็ม 1 ครั้ง ไม่ฟรี — "
            f"ถ้าตั้งถี่กว่า {_COST_CONFIRM_MINUTES} นาทีต้องยืนยันกับ user ก่อน (เตือน user ด้วยถ้ายังไม่ได้ถาม)"
        )
    lines.append("engine จะเริ่มทำงานเมื่อ host (Telegram bot / agent UI server) รันอยู่")
    return "\n".join(lines)


def _register(
    kind: str,
    target: str,
    task: str,
    interval_minutes: int,
    times: list[str] | None = None,
    state: dict | None = None,
) -> str:
    reg = _registry()
    outcome: dict = {}

    def _apply(watchers: list[dict]) -> None:
        if len(watchers) >= _MAX_WATCHERS:
            outcome["error"] = (
                f"เกินจำนวน watcher สูงสุด ({_MAX_WATCHERS} ตัว — รวม watcher ที่ paused ด้วย) "
                f"— ใช้ awake(action=\"list\") ดูก่อน แล้ว awake(action=\"stop\", awake_id=...) ตัวที่ไม่ใช้แล้ว"
            )
            return
        if kind == "screen" and any(w.get("kind") == "screen" and not w.get("paused") for w in watchers):
            outcome["error"] = (
                "มี screen watcher ทำงานอยู่แล้ว (จำกัด 1 ตัว — screencapture+OCR ทุกครั้ง check ไม่ฟรี) "
                "— stop ตัวเดิมก่อนถ้าต้องการเปลี่ยน task"
            )
            return
        wid = _new_id(watchers)
        record = {
            "id": wid,
            "kind": kind,
            "target": target,
            "task": task,
            "interval_minutes": interval_minutes,
            "times": times or [],
            "created": _now_iso(),
            "last_fire": None,
            "state": state or {},
            "consecutive_errors": 0,
            "paused": False,
        }
        watchers.append(record)
        outcome["record"] = record

    reg.mutate(_apply)
    if "error" in outcome:
        return _error(outcome["error"])
    return _format_registration(outcome["record"])


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text[:limit] + "..." if len(text) > limit else text


def _do_list() -> str:
    # Truncate the TASK/TARGET fields per-entry (not the whole listing) — a
    # single whole-listing cap could cut off later entries entirely, hiding
    # the very id the "list before stop" flow depends on (max 5 watchers).
    # Per-field caps keep every entry's id/kind/status visible.
    watchers = _registry().load()
    if not watchers:
        return "[ok] ไม่มี watcher ที่ลงทะเบียนอยู่ตอนนี้"
    lines = ["[ok] watcher ที่ลงทะเบียน:"]
    for w in watchers:
        status = "paused" if w.get("paused") else "active"
        if w.get("kind") == "every":
            detail = f"เวลา {', '.join(w['times'])} (ทุกวัน)" if w.get("times") else f"ทุก {w.get('interval_minutes', 60)} นาที"
        elif w.get("kind") == "once":
            detail = f"ครั้งเดียว {_format_run_at(w.get('state', {}).get('run_at', ''))}"
        elif w.get("kind") == "file":
            detail = f"ไฟล์ {_truncate(w.get('target', ''), 60)}"
        else:
            detail = "หน้าจอ"
        task = _truncate(w.get("task", ""), 60)
        lines.append(f"- {w.get('id')} [{w.get('kind')}/{status}] {detail} — {task}")
    return "\n".join(lines)


def _do_stop(awake_id: str) -> str:
    awake_id = awake_id.strip()
    if not awake_id:
        return _error('stop ต้องระบุ awake_id (หรือ "all") — ดู id ปัจจุบันด้วย awake(action="list")')
    reg = _registry()
    outcome: dict = {}

    def _apply(watchers: list[dict]) -> None:
        if awake_id.lower() == "all":
            outcome["stopped"] = [w.get("id") for w in watchers]
            watchers.clear()
            return
        for i, w in enumerate(watchers):
            if w.get("id") == awake_id:
                outcome["stopped"] = [awake_id]
                del watchers[i]
                return
        outcome["known_ids"] = [w.get("id") for w in watchers]

    reg.mutate(_apply)
    if "stopped" not in outcome:
        known = outcome.get("known_ids", [])
        listing = ", ".join(known) if known else "(ไม่มี watcher ที่ลงทะเบียนอยู่เลย)"
        return _error(f"ไม่พบ watcher id \"{awake_id}\" — id ปัจจุบัน: {listing}")
    if awake_id.lower() == "all":
        return f"[ok] หยุด watcher ทั้งหมดแล้ว ({len(outcome['stopped'])} ตัว)"
    return f"[ok] หยุด watcher {awake_id} แล้ว"


@tool
def awake(
    action: str,
    target: str = "",
    task: str = "",
    interval_minutes: int = 60,
    awake_id: str = "",
    times: str = "",
    run_at: str = "",
    delay_minutes: int = 0,
) -> str:
    """Register/list/stop a STANDING trigger that fires a NEW agent turn later — this is not tool_loop
    (which runs N items NOW, same turn). Calling awake() just registers the trigger and ends THIS turn
    immediately; the real work happens automatically in a future turn when the trigger condition is met.
    การลงทะเบียนจบ turn ทันที — งานจริงจะเกิดใน turn ใหม่อัตโนมัติเมื่อ trigger เข้าเงื่อนไข.

    action="watch_file" — target=<path> (required), task=<สิ่งที่ต้องทำเมื่อไฟล์เปลี่ยน> (required).
      Fires when the file's mtime changes (created/edited/deleted). target does NOT need to exist yet —
      watching for a file that will be CREATED later is legitimate; the first observation is just the
      baseline (no fire), a later create/edit/delete triggers.
    action="every" — recurring, two mutually exclusive modes:
      • interval_minutes=<int, min 5>, task — fires every N minutes, first fire after one full interval.
      • times="HH:MM"|"HH:MM,HH:MM"|"HH:MM,HH:MM,HH:MM" (24h, max 3), task — fires daily at each clock
        time instead of on an interval; interval_minutes is ignored when times is given. A time already
        passed today when you register does NOT fire immediately — it waits for its next occurrence
        (tomorrow). If the engine/host was offline across a target time (Mac asleep, app closed), that
        occurrence is SKIPPED, not fired late — the next tick simply waits for the next day's time.
        ❌ ตั้ง times="05:00" ตอน 14:00 → คาดว่า fire ทันที (ผิด — รอถึงพรุ่งนี้ 05:00)
        ✅ ตั้ง times="23:00,05:00" → บอก user ว่าจะ fire ทุกวันเวลานั้น ครั้งแรกคือรอบถัดไปที่ถึงเวลา
    action="once" — run task exactly once, then atomically unregister before dispatch so it cannot fire twice.
      Give exactly ONE schedule form:
      • run_at="YYYY-MM-DD HH:MM" — local Mac time, future only (also accepts T instead of the space).
      • delay_minutes=<int, min 1> — relative delay from registration time.
      If the host is asleep/offline at the target, it catches up once on the next tick after restart; unlike
      recurring times=, a missed one-shot is not skipped. One-shot cost is exactly one full LLM turn total.
        ❌ awake(action="once", task="เตือนประชุม")  — no schedule
        ✅ awake(action="once", run_at="2026-08-07 15:30", task="เตือนประชุม")
        ✅ awake(action="once", delay_minutes=20, task="ตรวจสถานะงานแล้วรายงาน")
    action="watch_screen" — task=<สิ่งที่ต้องทำเมื่อจอเปลี่ยน> (required). OCR-diff on the whole screen;
      only ONE active screen watcher allowed at a time (screencapture+OCR every check is not free). This
      fork has no computer tool — the fired turn can only NOTIFY (answer text + macOS notification), it
      cannot click/type on its own. Write task as the GOAL (e.g. "แจ้งเตือนเมื่อ...") not a click-by-click
      script; the fired turn's own trigger message states this scope explicitly.
    action="list" — no args. Shows active/paused watcher ids, kind, target/interval/times, task. Check this
      BEFORE stop so you have a real id, and before registering a new one (max 5 total, paused count too).
    action="stop" — awake_id=<id from list> or awake_id="all". Unregisters.

    COST — every fire is a full LLM turn, not free. interval_minutes below 5 is refused outright;
    below 15 you must confirm the cost with the user BEFORE calling awake(), not after. times= has no such
    floor (daily clock times are inherently low-frequency — 3 fires/day max), no cost confirmation needed.
      ❌ awake(action="every", interval_minutes=1, task="สรุปข่าวให้หน่อย")  — registered without asking, 1-min interval = 60 turns/hour
      ✅ ask user "ทุก 1 ชั่วโมงพอไหม (ทุก 1 นาทีจะเปลืองมาก)" → user ตกลง → awake(action="every", interval_minutes=60, task="สรุปข่าวให้หน่อย")
      (interval_minutes=15 or more needs no extra confirmation beyond what the user already asked for)
    """
    try:
        action = action.strip().lower()
        if action not in {"watch_file", "every", "once", "watch_screen", "list", "stop"}:
            return _error(f"unknown action: {action} (ใช้ watch_file/every/once/watch_screen/list/stop)")

        if action == "list":
            return _do_list()
        if action == "stop":
            return _do_stop(awake_id)

        if action == "once":
            if not task.strip():
                return _error("once ต้องระบุ task=<งานที่ต้องทำครั้งเดียว>")
            resolved, schedule_error = _resolve_once_run_at(run_at, delay_minutes)
            if schedule_error:
                return _error(schedule_error)
            return _register("once", "", task.strip(), 0, state={"run_at": resolved})

        if action == "watch_file":
            if not target.strip():
                return _error("watch_file ต้องระบุ target=<path ที่จะเฝ้า>")
            if not task.strip():
                return _error("watch_file ต้องระบุ task=<สิ่งที่ต้องทำเมื่อไฟล์เปลี่ยน>")
            return _register("file", target.strip(), task.strip(), 60)

        if action == "every":
            if not task.strip():
                return _error("every ต้องระบุ task=<งานที่ต้องทำซ้ำ>")
            parsed_times, times_error = _parse_times(times)
            if times_error:
                return _error(times_error)
            if parsed_times:
                return _register("every", "", task.strip(), interval_minutes, times=parsed_times)
            if interval_minutes < _MIN_EVERY_MINUTES:
                return _error(
                    f"every ต้องมี interval_minutes >= {_MIN_EVERY_MINUTES} นาที "
                    f"(แต่ละครั้งที่ fire คือ LLM turn เต็ม 1 ครั้ง ไม่ฟรี) หรือระบุ times=\"HH:MM\" แทน"
                )
            return _register("every", "", task.strip(), interval_minutes)

        # watch_screen
        if not task.strip():
            return _error("watch_screen ต้องระบุ task=<สิ่งที่ต้องทำเมื่อหน้าจอเปลี่ยน>")
        return _register("screen", "", task.strip(), 60)
    except Exception as exc:
        return _error(str(exc))
