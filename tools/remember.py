# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

import datetime
import json
import logging
import os
import re
import threading
from filelock import FileLock
from langchain_core.tools import tool

log = logging.getLogger(__name__)

_MEMORY_FILE       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "memory.md"))
_MEMORY_MAX_CHARS  = 5_000   # ~50-100 facts; keeps system prompt bounded
_MEMORY_COMPACT_TO = 3_500   # LLM compaction target — headroom below max so it doesn't re-fire every append
_MEMORY_LOCK       = FileLock(_MEMORY_FILE + ".lock")  # cross-process; reentrant within one process
_THREAD_LOCK       = threading.Lock()                  # in-process: FileLock alone doesn't serialize threads (O-31 class)
_FACT_MAX_CHARS    = 1_000   # single-fact cap — also guarantees drop-oldest fallback can always fit ≥1 entry
_CATEGORY_MAX      = 30
_LLM_TIMEOUT       = 45
_MAX_CONFLICT_REMOVALS = 3   # LLM asking to remove more existing lines than this = judged garbage, ignore

_LEADING_TAGS_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)*")


def _today() -> str:
    return datetime.date.today().isoformat()


def _entry(fact: str, category: str) -> str:
    tag = f" [{category}]" if category else ""
    return f"- [{_today()}]{tag} {fact}\n"


def _normalize(line: str) -> str:
    """Comparable form of an entry: drop '- ', leading [date]/[category] tags, casefold."""
    s = line.strip()
    if s.startswith("- "):
        s = s[2:]
    s = _LEADING_TAGS_RE.sub("", s)
    return re.sub(r"\s+", " ", s).casefold().strip()


def _read_lines(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return f.readlines()
    except FileNotFoundError:
        return []


def _llm_chat(prompt: str, max_tokens: int = 800) -> str:
    """One non-streaming completion against the production model. Raises on any failure —
    every caller has a deterministic fallback; a memory save must never fail because of the LLM."""
    import requests
    from config import MLX_BASE_URL, MODEL

    r = requests.post(
        f"{MLX_BASE_URL}/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=_LLM_TIMEOUT,
    )
    if not r.ok:
        raise ValueError(f"HTTP {r.status_code}")
    raw = r.json()["choices"][0]["message"].get("content") or ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    if not raw:
        raise ValueError("empty LLM response")
    return raw


def _conflict_scan(lines: list[str], fact: str) -> list[int]:
    """Indexes of existing lines the new fact DUPLICATES or CONTRADICTS (newest wins).
    Raises on LLM failure/garbage — caller falls back to plain append."""
    numbered = "\n".join(f"{i}. {l.strip()}" for i, l in enumerate(lines))
    prompt = (
        "A user memory file contains these numbered entries:\n"
        f"{numbered}\n\n"
        f'A new entry is being added:\n"{fact}"\n\n'
        "Which existing entries does the new entry DUPLICATE (same information) or "
        "CONTRADICT (outdated information the new entry supersedes)? Newest information wins. "
        "Only list an entry when it is clearly the SAME topic — different topics are never conflicts. "
        "When unsure, do not list it.\n"
        'Answer with JSON only: {"remove": [<entry numbers>]} — empty list if none.'
    )
    parsed = json.loads(_llm_chat(prompt, max_tokens=200))
    idxs = parsed.get("remove", []) if isinstance(parsed, dict) else None
    if not isinstance(idxs, list):
        raise ValueError("LLM returned non-list 'remove'")
    idxs = sorted({int(i) for i in idxs})
    if any(i < 0 or i >= len(lines) for i in idxs):
        raise ValueError("LLM index out of range")
    if len(idxs) > _MAX_CONFLICT_REMOVALS:
        raise ValueError(f"LLM wants to remove {len(idxs)} lines (> {_MAX_CONFLICT_REMOVALS}) — judged garbage")
    return idxs


def _compact_memory_llm(content: str) -> str:
    """Summarize/merge memory content down to _MEMORY_COMPACT_TO chars.
    Raises on failure/garbage — caller falls back to drop-oldest."""
    prompt = (
        "Rewrite this user memory file so it fits within "
        f"{_MEMORY_COMPACT_TO} characters WITHOUT losing distinct facts:\n"
        "- Merge entries about the same topic into one line.\n"
        "- When entries contradict, keep only the one with the newest [YYYY-MM-DD] date "
        "(undated entries are oldest).\n"
        "- Keep the format: one entry per line, '- [YYYY-MM-DD] [category] fact' "
        "(keep existing dates/categories where present).\n"
        "- Keep the original language of each entry. No explanation, output the file content only.\n\n"
        f"{content}"
    )
    out = _llm_chat(prompt, max_tokens=1_500).strip()
    if not out or "- " not in out:
        raise ValueError("compaction output has no entries")
    if len(out) > _MEMORY_MAX_CHARS:
        raise ValueError(f"compaction output too long ({len(out)} chars)")
    return out + "\n"


def _backup_memory(content: str) -> None:
    """memory.md is user-authored — never LLM-rewrite/delete from it without a recoverable copy."""
    try:
        with open(_MEMORY_FILE + ".bak", "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        log.warning("memory.md backup failed: %s", e)


def _trim_memory(path: str, max_chars: int) -> None:
    """Deterministic fallback: drop oldest lines until file fits. Caller must hold _MEMORY_LOCK."""
    try:
        lines = _read_lines(path)
        while lines and sum(len(l) for l in lines) > max_chars:
            lines.pop(0)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        log.warning("memory.md trim failed (file may grow unbounded): %s", e)


def _enforce_cap(path: str, max_chars: int) -> str:
    """Over cap → backup then LLM-summarize; LLM fails → drop-oldest.
    Returns a short user-facing note ('' if under cap).
    Manages its own brief locks around the read and the write — the LLM compaction call
    (up to _LLM_TIMEOUT seconds) runs with no lock held, so it doesn't stall other
    remember()/forget() callers on the network round-trip. Re-reads before writing —
    if the file changed underneath us during that round-trip, the compacted-from-stale-
    snapshot result is discarded instead of clobbering the concurrent write; the next
    remember() call retries the cap check from scratch."""
    with _THREAD_LOCK, _MEMORY_LOCK:
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return ""
    if len(content) <= max_chars:
        return ""
    _backup_memory(content)
    try:
        compacted = _compact_memory_llm(content)
        with _THREAD_LOCK, _MEMORY_LOCK:
            with open(path, encoding="utf-8") as f:
                current = f.read()
            if current != content:
                log.warning("memory.md changed during compaction — discarding stale result, will retry next call")
                return ""
            with open(path, "w", encoding="utf-8") as f:
                f.write(compacted)
        return f" (memory เต็ม — สรุปย่อแล้ว {len(content)}→{len(compacted)} chars, สำรองไว้ที่ memory.md.bak)"
    except Exception as e:
        log.warning("memory compaction failed (%s) — dropping oldest lines", type(e).__name__)
        with _THREAD_LOCK, _MEMORY_LOCK:
            _trim_memory(path, max_chars)
        return " (memory เต็ม — ตัดรายการเก่าสุดออก, สำรองไว้ที่ memory.md.bak)"


@tool
def remember(
    fact: str = "",
    replace: str = "",
    forget: str = "",
    category: str = "",
    insert_after: str = "",
    insert_before: str = "",
    insert_at: int = 0,
) -> str:
    """Save, update, delete, or insert a fact in permanent memory (memory.md). Four modes:

    MODE 1 — APPEND (only fact given):
      Use when: user says 'จำไว้ว่า / จำด้วย / บันทึกว่า / จดว่า / remember that'
      Effect: adds a new line. Duplicates/contradictions with existing memory are
      auto-resolved — the NEW fact wins, outdated lines are removed automatically.
      Optional category: short topic tag, e.g. category="การลงทุน" / "ส่วนตัว" / "งาน".
      Example: remember(fact="ชอบกาแฟดำ", category="ส่วนตัว")

    MODE 2 — REPLACE (fact + replace given):
      Use when: user says 'แก้จาก X เป็น Y / เปลี่ยน X เป็น Y / แก้ไข memory ที่บอกว่า X'
      Effect: finds the line containing `replace` text and overwrites it with `fact`.
      Returns error if `replace` text not found — do NOT silently append instead.
      Example: remember(fact="ชอบกาแฟนม", replace="ชอบกาแฟดำ")

    MODE 3 — FORGET (only forget given):
      Use when: user says 'ลบ memory / ไม่ต้องจำ X แล้ว / ลืม X ไปเลย / forget X'
      Effect: deletes EVERY line containing `forget` text. Returns error if not found.
      Example: remember(forget="ชอบกาแฟดำ")

    MODE 4 — INSERT (fact + exactly one insert anchor):
      Use when: user asks to insert/add a memory line at a specific place.
      Effect: inserts a new dated memory line before/after the first line containing anchor text,
      or before 1-based line number `insert_at` (insert_at=1 means top, len+1 means bottom).
      This is a precise edit: it does NOT run duplicate/conflict cleanup.
      Examples:
        remember(fact="ใช้ model Qwen3", insert_after="LLM servers")
        remember(fact="Section note", insert_before="Project rules")
        remember(fact="Top note", insert_at=1)

    KEY DISTINCTION — NEVER guess the mode:
      - Only fact → MODE 1 (append; conflicts auto-resolved, newest wins)
      - fact + replace → MODE 2 (replace); if not found → return error, ask user to confirm
      - Only forget → MODE 3 (delete); NEVER pass fact together with forget
      - fact + one insert anchor → MODE 4 (insert); if anchor not found → return error
    """
    try:
        fact = re.sub(r"\s*\n\s*", " ", fact).strip()  # entries are line-based — a newline would split one
        replace, forget = replace.strip(), forget.strip()
        insert_after, insert_before = insert_after.strip(), insert_before.strip()
        category = re.sub(r"[\[\]|\n]", "", category).strip()[:_CATEGORY_MAX]
        insert_mode_count = sum(bool(x) for x in (insert_after, insert_before, insert_at))

        if forget:
            if replace or insert_mode_count:
                return "[error] ใช้ forget กับ replace/insert พร้อมกันไม่ได้ — เลือกโหมดเดียว"
            if fact:
                return "[error] โหมด forget ต้องส่ง forget อย่างเดียว — ถ้าต้องการแก้ไขให้ใช้ fact+replace"
            with _THREAD_LOCK, _MEMORY_LOCK:
                lines = _read_lines(_MEMORY_FILE)
                kept = [l for l in lines if forget not in l]
                removed = [l.strip() for l in lines if forget in l]
                if not removed:
                    return f"[error] ไม่พบข้อความ '{forget}' ใน memory.md — ไม่มีอะไรถูกลบ"
                with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
                    f.writelines(kept)
            shown = "; ".join(removed)[:300]
            return f"ลบแล้ว {len(removed)} รายการ: {shown}"

        if not fact:
            return "[error] fact ว่างเปล่า — ระบุข้อมูลที่ต้องการจำ"
        if len(fact) > _FACT_MAX_CHARS:
            return f"[error] fact ยาวเกิน {_FACT_MAX_CHARS} chars — สรุปให้สั้นลงก่อนบันทึก"
        if replace and insert_mode_count:
            return "[error] ใช้ replace กับ insert พร้อมกันไม่ได้ — เลือกโหมดเดียว"
        if insert_mode_count > 1:
            return "[error] ระบุ insert_after, insert_before, หรือ insert_at ได้อย่างใดอย่างหนึ่งเท่านั้น"
        if insert_at < 0:
            return "[error] insert_at ต้องเป็นเลขบรรทัดแบบ 1-based และต้องไม่ติดลบ"

        if replace:
            # No LLM call on this path — read+write stay in one brief lock.
            with _THREAD_LOCK, _MEMORY_LOCK:
                lines = _read_lines(_MEMORY_FILE)
                matched = False
                new_lines = []
                for line in lines:
                    if not matched and replace in line:
                        new_lines.append(_entry(fact, category))
                        matched = True
                    else:
                        new_lines.append(line)
                if not matched:
                    return f"[error] ไม่พบข้อความ '{replace}' ใน memory.md — ใช้ remember(fact) เพื่อเพิ่มใหม่แทน"
                with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            note = _enforce_cap(_MEMORY_FILE, _MEMORY_MAX_CHARS)
            return f"แก้ไขแล้ว: '{replace}' → '{fact}'{note}"

        if insert_mode_count:
            # Precise edit path: no LLM conflict scan, read+write stay in one brief lock.
            with _THREAD_LOCK, _MEMORY_LOCK:
                lines = _read_lines(_MEMORY_FILE)
                if insert_at:
                    if insert_at > len(lines) + 1:
                        return f"[error] insert_at={insert_at} เกินจำนวนบรรทัดใน memory.md ({len(lines)})"
                    idx = insert_at - 1
                    where = f"บรรทัด {insert_at}"
                else:
                    anchor = insert_after or insert_before
                    matches = [i for i, line in enumerate(lines) if anchor in line]
                    if not matches:
                        return f"[error] ไม่พบข้อความ '{anchor}' ใน memory.md — แทรกบรรทัดไม่ได้"
                    idx = matches[0] + (1 if insert_after else 0)
                    where = ("หลัง" if insert_after else "ก่อน") + f" '{anchor}'"
                lines.insert(idx, _entry(fact, category))
                with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
                    f.writelines(lines)
            note = _enforce_cap(_MEMORY_FILE, _MEMORY_MAX_CHARS)
            return f"แทรกแล้วที่{where}: {fact}{note}"

        # MODE 1 — append with auto conflict resolution (newest wins)
        with _THREAD_LOCK, _MEMORY_LOCK:
            lines = _read_lines(_MEMORY_FILE)
        entries = [l for l in lines if l.strip()]
        norm_new = _normalize(fact)
        if any(_normalize(l) == norm_new for l in entries):
            return f"มีข้อมูลนี้อยู่แล้วใน memory — ไม่บันทึกซ้ำ: {fact}"

        # _conflict_scan is an LLM call (up to _LLM_TIMEOUT s) — runs with no lock held so it
        # doesn't stall other remember()/forget() callers on the network round-trip.
        removed_note = ""
        removed_texts: set[str] = set()
        if entries:
            try:
                idxs = _conflict_scan(entries, fact)
                if idxs:
                    removed = [entries[i].strip() for i in idxs]
                    removed_texts = {entries[i] for i in idxs}
                    _backup_memory("".join(lines))  # LLM-driven deletion — never without a recoverable copy
                    removed_note = (
                        f" (แทนที่ข้อมูลเดิมที่ซ้ำ/ขัดแย้ง {len(removed)} รายการ: "
                        + "; ".join(removed)[:300] + ")"
                    )
            except Exception as e:
                log.warning("conflict scan failed (%s) — plain append", type(e).__name__)

        with _THREAD_LOCK, _MEMORY_LOCK:
            # Re-read current content instead of writing back the pre-LLM-call
            # snapshot — a concurrent remember()/forget() landing during
            # _conflict_scan's network round-trip must not be clobbered.
            kept_lines = [l for l in _read_lines(_MEMORY_FILE) if l not in removed_texts]
            kept_lines = kept_lines + [_entry(fact, category)]
            with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)
        note = _enforce_cap(_MEMORY_FILE, _MEMORY_MAX_CHARS)
        return f"บันทึกแล้ว: {fact}{removed_note}{note}"
    except Exception as e:
        return f"[error] {e}"
