# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""web_cache.py — process-level in-memory cache for web tool outputs

Scope: per-session (process), in-memory only. Closing the program → cache gone.

Two separate caches:
  _RAW_CACHE     url → raw body (fetcher result, tool-independent)
  _SUMMARY_CACHE url|query_hash → query-aware summary

TTL: raw entries expire after CACHE_TTL_SECONDS (default 30 min). On expiry,
both the raw entry and its associated summaries are treated as a miss → re-fetch.

Eviction: LRU by last_accessed, capped by entry count and total bytes (raw only).

Query-aware summaries: same URL + different user query → different summary key,
so the agent gets a summary focused on *this* query rather than a stale one from
an earlier unrelated fetch.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WEB_CACHE_MAX_ENTRIES,
    WEB_CACHE_MAX_BYTES,
    WEB_CACHE_PER_ENTRY_MAX,
)

# Raw body cache: url → {raw, fetched_at, last_accessed, size_bytes}
_RAW_CACHE: dict[str, dict] = {}
# Summary cache: "{url}|{query_hash}" → summary string  (max 200 entries, simple FIFO)
_SUMMARY_CACHE: OrderedDict[str, str] = OrderedDict()
_SUMMARY_MAX = WEB_CACHE_MAX_ENTRIES  # mirrors raw-cache entry cap

_LOCK = threading.Lock()

CACHE_TTL_SECONDS = int(os.getenv("V2_WEB_CACHE_TTL", "1800"))  # 30 min default

# ── per-turn web call counter ─────────────────────────────────────────────────
_WEB_COUNT: list[int] = [0]
_WEB_MAX = int(os.getenv("V2_MAX_WEB_CALLS", "20"))
# Per-turn cap, defaults to _WEB_MAX. graph.py lowers this for simple-search
# turns (S2: code-enforced "≤2 web_search") and web_count_reset() restores it.
_WEB_LIMIT: list[int] = [_WEB_MAX]


def web_count_reset() -> None:
    _WEB_COUNT[0] = 0
    _WEB_LIMIT[0] = _WEB_MAX


def web_count_set_limit(n: int) -> None:
    """Lower this turn's web-call cap to n (used for simple-search turns)."""
    _WEB_LIMIT[0] = n


def web_count_check() -> str | None:
    if _WEB_COUNT[0] >= _WEB_LIMIT[0]:
        return f"[web_limit] ค้นครบ {_WEB_LIMIT[0]} ครั้งแล้ว — หยุดค้นและสรุปจากข้อมูลที่มีได้เลย"
    return None


def web_count_inc() -> None:
    _WEB_COUNT[0] += 1


# ── helpers ───────────────────────────────────────────────────────────────────

def _query_hash(query: str) -> str:
    """8-char MD5 hex of normalised query. Empty query → empty string."""
    if not query or not query.strip():
        return ""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()[:8]


def _summary_key(url: str, query: str) -> str:
    h = _query_hash(query)
    return f"{url}|{h}" if h else url


def _is_expired(entry: dict) -> bool:
    return (time.time() - entry["fetched_at"]) > CACHE_TTL_SECONDS


# ── raw body cache ────────────────────────────────────────────────────────────

def get(url: str) -> str | None:
    """Return cached raw body for url, or None on miss / TTL expiry."""
    with _LOCK:
        entry = _RAW_CACHE.get(url)
        if entry is None:
            return None
        if _is_expired(entry):
            _RAW_CACHE.pop(url, None)
            # Also purge all summary entries for this URL
            stale = [k for k in _SUMMARY_CACHE if k == url or k.startswith(f"{url}|")]
            for k in stale:
                _SUMMARY_CACHE.pop(k, None)
            return None
        entry["last_accessed"] = time.time()
        return entry["raw"]


def put(url: str, raw: str) -> None:
    """Store raw body. Refuses errors/empty. Truncates + evicts LRU if needed."""
    if not url or not raw:
        return
    if raw.lstrip().startswith("[error]"):
        return
    if len(raw) > WEB_CACHE_PER_ENTRY_MAX:
        raw = raw[:WEB_CACHE_PER_ENTRY_MAX] + "\n...[cache-truncated]"
    size = len(raw.encode("utf-8", errors="ignore"))
    now = time.time()
    with _LOCK:
        _RAW_CACHE[url] = {
            "raw": raw,
            "fetched_at": now,
            "last_accessed": now,
            "size_bytes": size,
        }
        _evict_if_needed_locked()


# ── query-aware summary cache ─────────────────────────────────────────────────

def get_summary(url: str, query: str = "") -> str | None:
    """Return cached summary for (url, query), or None on miss / raw TTL expiry."""
    with _LOCK:
        # If raw has expired, the summary is stale too
        raw_entry = _RAW_CACHE.get(url)
        if raw_entry is None or _is_expired(raw_entry):
            return None
        key = _summary_key(url, query)
        if key not in _SUMMARY_CACHE:
            return None
        _SUMMARY_CACHE.move_to_end(key)
        return _SUMMARY_CACHE[key]


def put_summary(url: str, query: str, summary: str) -> None:
    """Store query-aware summary. LRU evict when over _SUMMARY_MAX."""
    if not url or not summary:
        return
    key = _summary_key(url, query)
    with _LOCK:
        if key in _SUMMARY_CACHE:
            _SUMMARY_CACHE.move_to_end(key)
        elif len(_SUMMARY_CACHE) >= _SUMMARY_MAX:
            _SUMMARY_CACHE.popitem(last=False)
        _SUMMARY_CACHE[key] = summary


# ── LRU eviction (raw cache) ──────────────────────────────────────────────────

def _evict_if_needed_locked() -> None:
    while len(_RAW_CACHE) > WEB_CACHE_MAX_ENTRIES:
        _evict_one_locked()
    while True:
        total = sum(e["size_bytes"] for e in _RAW_CACHE.values())
        if total <= WEB_CACHE_MAX_BYTES or not _RAW_CACHE:
            break
        _evict_one_locked()


def _evict_one_locked() -> None:
    if not _RAW_CACHE:
        return
    oldest_url = min(_RAW_CACHE.keys(), key=lambda u: _RAW_CACHE[u]["last_accessed"])
    _RAW_CACHE.pop(oldest_url, None)
    stale = [k for k in _SUMMARY_CACHE if k == oldest_url or k.startswith(f"{oldest_url}|")]
    for k in stale:
        _SUMMARY_CACHE.pop(k, None)


def evict_if_needed() -> None:
    with _LOCK:
        _evict_if_needed_locked()


def clear() -> None:
    with _LOCK:
        _RAW_CACHE.clear()
        _SUMMARY_CACHE.clear()


def stats() -> dict:
    with _LOCK:
        total = sum(e["size_bytes"] for e in _RAW_CACHE.values())
        return {
            "raw_entries": len(_RAW_CACHE),
            "summary_entries": len(_SUMMARY_CACHE),
            "bytes": total,
            "max_entries": WEB_CACHE_MAX_ENTRIES,
            "max_bytes": WEB_CACHE_MAX_BYTES,
            "ttl_seconds": CACHE_TTL_SECONDS,
        }


def split_summary(content: str) -> tuple[str, str]:
    """Legacy helper — kept for any callers that still pass combined content."""
    if "\n---RAW---\n" in content:
        summary, raw = content.split("\n---RAW---\n", 1)
        return summary, raw
    return content, ""
