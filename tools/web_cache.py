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
import hashlib
import threading
import time
import sys
import os
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WEB_CACHE_MAX_ENTRIES,
    WEB_CACHE_MAX_BYTES,
    WEB_CACHE_PER_ENTRY_MAX,
)

# Raw body cache: url → {raw, fetched_at, last_accessed, size_bytes}
_RAW_CACHE: dict[str, dict] = {}
# Summary cache: "{url}|{query_hash}" → summary string  (max 200 entries, LRU)
_SUMMARY_CACHE: OrderedDict[str, str] = OrderedDict()
_SUMMARY_MAX = WEB_CACHE_MAX_ENTRIES  # mirrors raw-cache entry cap

_LOCK = threading.Lock()

CACHE_TTL_SECONDS = int(os.getenv("V2_WEB_CACHE_TTL", "1800"))  # 30 min default

# ── per-turn web call counter ─────────────────────────────────────────────────
_WEB_COUNT: list[int] = [0]
_WEB_MAX = int(os.getenv("V2_MAX_WEB_CALLS", "20"))
# Per-turn cap, defaults to _WEB_MAX. graph.py lowers this for simple-search
# turns (S2: code-enforced "≤5 web_search") and raises it for create_plan/complex
# turns ("≤30 web_search") unless a handbook/current-facts turn requires the
# tighter finance cap; web_count_reset() restores the default each turn.
_WEB_LIMIT: list[int] = [_WEB_MAX]
# Some turns have a non-negotiable ceiling rather than an ordinary working
# limit.  `tool_loop` may legitimately raise an ordinary limit for a large
# batch, but it must never expand a handbook/current-facts turn past the cap
# that graph.py set for it.
_WEB_HARD_LIMIT: list[int | None] = [None]

# Source provenance scope for the active turn.  The model may omit "official"
# from a follow-up tool argument even when the user's original request required
# primary sources, so this constraint must live beside the atomic turn counter
# rather than be inferred independently from each tool call.
_WEB_OFFICIAL_ONLY: list[bool] = [False]


def web_count_reset() -> None:
    with _LOCK:
        _WEB_COUNT[0] = 0
        _WEB_LIMIT[0] = _WEB_MAX
        _WEB_HARD_LIMIT[0] = None
        _WEB_OFFICIAL_ONLY[0] = False


def web_count_set_limit(n: int) -> None:
    """Lower this turn's web-call cap to n (used for simple-search turns)."""
    with _LOCK:
        limit = max(0, int(n))
        hard_limit = _WEB_HARD_LIMIT[0]
        _WEB_LIMIT[0] = min(limit, hard_limit) if hard_limit is not None else limit


def web_count_set_hard_limit(n: int) -> None:
    """Set a per-turn ceiling that batch helpers cannot raise past."""
    with _LOCK:
        limit = max(0, int(n))
        _WEB_HARD_LIMIT[0] = limit
        _WEB_LIMIT[0] = min(_WEB_LIMIT[0], limit)


def web_source_scope_set_official_only(enabled: bool = True) -> None:
    """Latch the active turn to primary/issuer sources until the next reset."""
    if not enabled:
        return
    with _LOCK:
        _WEB_OFFICIAL_ONLY[0] = True


def web_source_scope_official_only() -> bool:
    with _LOCK:
        return _WEB_OFFICIAL_ONLY[0]


def web_source_scope_url_error(url: str) -> str | None:
    """Return a source-policy error for a disallowed URL in this turn.

    URL-consuming tools must call this before consulting their cache as well as
    before network I/O: an old non-official cache entry is not valid evidence
    merely because the current user asked for primary sources.
    """
    if not web_source_scope_official_only():
        return None
    try:
        # Lazy import avoids a module-import cycle: web_search itself imports
        # this cache module, while the policy helper is only needed at runtime.
        from tools.web_search import _is_official_domain
    except Exception:
        return "[source_blocked] official-only turn — unable to validate the source domain"
    if _is_official_domain(url):
        return None
    return "[source_blocked] official-only turn — browse an official/issuer domain instead"


def web_count_check() -> str | None:
    """Read-only peek — does not reserve a slot. Safe for fast-path bail-outs
    that aren't paired 1:1 with an increment (e.g. _batch_browse's pre-fetch
    early exit). Do NOT use this immediately before a real fetch; use
    web_count_check_and_inc() there to avoid the TOCTOU race below."""
    with _LOCK:
        if _WEB_COUNT[0] >= _WEB_LIMIT[0]:
            return f"[web_limit] ค้นครบ {_WEB_LIMIT[0]} ครั้งแล้ว — หยุดค้นและสรุปจากข้อมูลที่มีได้เลย"
        return None


def web_count_check_and_inc() -> str | None:
    """Atomically check the per-turn cap and reserve a slot in one step.

    ToolNode (langgraph.prebuilt) runs multiple tool_calls from a single
    AIMessage concurrently via a real ThreadPoolExecutor — a separate
    check-then-increment lets two threads both pass the check before either
    increments, silently overrunning the cap (audit.md CONC-2). Call this
    immediately before doing the actual fetch; if it returns an error, no
    slot was reserved.
    """
    with _LOCK:
        if _WEB_COUNT[0] >= _WEB_LIMIT[0]:
            return f"[web_limit] ค้นครบ {_WEB_LIMIT[0]} ครั้งแล้ว — หยุดค้นและสรุปจากข้อมูลที่มีได้เลย"
        _WEB_COUNT[0] += 1
        return None


def web_count_reserve_up_to(n: int) -> int:
    """Atomically reserve up to ``n`` outbound web attempts for a batch.

    Batch tools must reserve before starting their worker threads.  Checking
    now and incrementing after a fetch lets a concurrent tool consume the
    final slot while this batch is already on the network.
    """
    wanted = max(0, int(n))
    with _LOCK:
        granted = min(wanted, max(0, _WEB_LIMIT[0] - _WEB_COUNT[0]))
        _WEB_COUNT[0] += granted
        return granted


def web_count_release_reservation(n: int = 1) -> None:
    """Release slots reserved for work that could not be submitted.

    Only call this immediately after ``web_count_reserve_up_to`` when no
    outbound attempt was started (for example, a bounded executor was busy).
    Started attempts stay counted even if their fetch later fails, matching
    the single-URL tools' budget semantics.
    """
    release = max(0, int(n))
    with _LOCK:
        _WEB_COUNT[0] = max(0, _WEB_COUNT[0] - release)


def web_count_inc() -> None:
    with _LOCK:
        _WEB_COUNT[0] += 1


def web_count_remaining() -> int:
    """Return how many web slots remain this turn (never negative)."""
    with _LOCK:
        return max(0, _WEB_LIMIT[0] - _WEB_COUNT[0])


def web_count_ensure_headroom(extra: int, hard_max: int) -> int:
    """Atomically raise the per-turn limit to fit `extra` more calls if needed.

    Used by tool_loop before a large batch: reads current count + limit under
    _LOCK in one step so there is no TOCTOU gap between reading _WEB_COUNT and
    calling web_count_set_limit (which previously let two concurrent ToolNode
    tool-calls each see a stale count and over-raise the limit independently).

    Returns the remaining slots after the potential limit raise (never negative).
    """
    with _LOCK:
        requested_hard_max = max(0, int(hard_max))
        turn_hard_limit = _WEB_HARD_LIMIT[0]
        effective_hard_max = min(requested_hard_max, turn_hard_limit) \
            if turn_hard_limit is not None else requested_hard_max
        _WEB_LIMIT[0] = min(_WEB_LIMIT[0], effective_hard_max)
        needed = _WEB_COUNT[0] + max(0, int(extra))
        if needed > _WEB_LIMIT[0]:
            _WEB_LIMIT[0] = min(needed, effective_hard_max)
        return max(0, _WEB_LIMIT[0] - _WEB_COUNT[0])


# ── adaptive DDG rate-limit (F3) ───────────────────────────────────────────────
# Process-wide last-DDG-call timestamp. Sleeps only the remaining gap to
# _DDG_MIN_INTERVAL instead of an unconditional 1s before every call.
# Separate lock from _LOCK so a 1-second sleep here doesn't block all cache ops.
_LAST_DDG_CALL: list[float] = [0.0]
_DDG_MIN_INTERVAL = 1.0
_DDG_LOCK = threading.Lock()


def ddg_wait() -> None:
    with _DDG_LOCK:
        # Pacing is about elapsed process time. A wall-clock correction must
        # never turn a sub-second gap into an hours-long sleep.
        elapsed = time.monotonic() - _LAST_DDG_CALL[0]
        if elapsed < _DDG_MIN_INTERVAL:
            time.sleep(_DDG_MIN_INTERVAL - elapsed)
        _LAST_DDG_CALL[0] = time.monotonic()


# ── helpers ───────────────────────────────────────────────────────────────────

def _query_hash(query: str) -> str:
    """Full SHA-256 of a normalised query. Empty query → empty string.

    The old 8-hex MD5 prefix had only 32 bits of key space, so unrelated user
    questions could realistically alias within a long-lived process and reuse
    the wrong query-aware summary. This cache is in-memory only, so there is no
    persisted-key compatibility cost to using the full digest.
    """
    if not query or not query.strip():
        return ""
    return hashlib.sha256(query.lower().strip().encode()).hexdigest()


def _summary_key(url: str, query: str) -> str:
    h = _query_hash(query)
    return f"{url}|{h}" if h else url


def _age_seconds(entry: dict) -> float:
    return max(0.0, time.monotonic() - entry["fetched_at"])


def _is_expired(entry: dict) -> bool:
    return _age_seconds(entry) > CACHE_TTL_SECONDS


# ── raw body cache ────────────────────────────────────────────────────────────

def _purge_url_locked(url: str) -> None:
    """Remove url's raw entry (if present) and every query-variant summary keyed
    off it. Caller must already hold _LOCK. Shared by get(), get_summary(), and
    _evict_one_locked() so the summary-key scan isn't triplicated."""
    _RAW_CACHE.pop(url, None)
    stale = [k for k in _SUMMARY_CACHE if k == url or k.startswith(f"{url}|")]
    for k in stale:
        _SUMMARY_CACHE.pop(k, None)


def get(url: str, *, max_age_seconds: float | None = None) -> str | None:
    """Return cached raw body for url, or None on miss / expiry.

    ``max_age_seconds`` is a caller-specific freshness ceiling. If the entry is
    older than that ceiling but still inside the ordinary cache TTL, treat it as
    a miss without purging it; a live quote can then refetch while stable readers
    can still reuse the older body if the refresh fails.

    Official-only turns also require provenance recorded by the fetch path. A
    cache entry from an earlier ordinary turn may have followed a redirect, so
    its initial URL alone is not enough to reuse it as primary evidence.
    """
    with _LOCK:
        entry = _RAW_CACHE.get(url)
        if entry is None:
            return None
        age = _age_seconds(entry)
        if age > CACHE_TTL_SECONDS:
            _purge_url_locked(url)
            return None
        if max_age_seconds is not None and age > max(0.0, float(max_age_seconds)):
            return None
        if _WEB_OFFICIAL_ONLY[0] and not entry.get("official_verified", False):
            return None
        entry["last_accessed"] = time.monotonic()
        return entry["raw"]


def put(url: str, raw: str, *, official_verified: bool = False,
        page_dates: tuple[str, str] | None = None) -> None:
    """Store raw body. Refuses errors/empty. Truncates + evicts LRU if needed.

    ``official_verified`` is set only by a fetch path that checked the final
    effective URL. It lets a later official-only turn reuse genuinely primary
    cached evidence without trusting legacy or ordinary-turn cache entries.
    ``page_dates`` is the page's own (published, modified) ISO metadata; a
    same-body re-put without it keeps the dates already recorded.
    """
    if not url or not raw:
        return
    if raw.lstrip().startswith(("[error]", "[source_blocked]", "[web_limit]")):
        return
    if len(raw) > WEB_CACHE_PER_ENTRY_MAX:
        raw = raw[:WEB_CACHE_PER_ENTRY_MAX] + "\n...[cache-truncated]"
    size = len(raw.encode("utf-8", errors="ignore"))
    # Both TTL and LRU timestamps are process-local elapsed-time values. Using
    # wall time here can keep stale entries alive or corrupt LRU order after an
    # NTP/manual clock adjustment.
    now = time.monotonic()
    with _LOCK:
        previous = _RAW_CACHE.get(url)
        same_generation = previous is not None and previous["raw"] == raw
        if not same_generation:
            # A summary describes one exact raw generation. Never let it
            # survive a refetch whose body changed.
            _purge_url_locked(url)
        verified = bool(official_verified) or bool(
            same_generation and previous and previous.get("official_verified", False)
        )
        if page_dates is None and same_generation and previous:
            page_dates = previous.get("page_dates")
        _RAW_CACHE[url] = {
            "raw": raw,
            "fetched_at": now,
            "last_accessed": now,
            "size_bytes": size,
            "official_verified": verified,
            "page_dates": tuple(page_dates) if page_dates else ("", ""),
        }
        _evict_if_needed_locked()


def get_page_dates(url: str) -> tuple[str, str]:
    """(published, modified) ISO dates recorded with url's raw body, or ("", "")."""
    with _LOCK:
        entry = _RAW_CACHE.get(url)
        if entry is None or _is_expired(entry):
            return ("", "")
        return tuple(entry.get("page_dates") or ("", ""))


# ── query-aware summary cache ─────────────────────────────────────────────────

def get_summary(url: str, query: str = "") -> str | None:
    """Return cached summary for (url, query), or None on miss / raw TTL expiry."""
    with _LOCK:
        # If raw has expired, the summary is stale too — purge both (matches get()'s
        # expiry behavior; previously only get() purged, leaving a dead raw entry +
        # its summaries sitting in cache until something else touched that URL).
        raw_entry = _RAW_CACHE.get(url)
        if raw_entry is None:
            return None
        if _is_expired(raw_entry):
            _purge_url_locked(url)
            return None
        if _WEB_OFFICIAL_ONLY[0] and not raw_entry.get("official_verified", False):
            return None
        key = _summary_key(url, query)
        if key not in _SUMMARY_CACHE:
            return None
        _SUMMARY_CACHE.move_to_end(key)  # LRU: mark as recently used
        # A summary hit is a use of its parent raw generation. Summary-producing
        # workflows such as research_orchestrator can return straight from here
        # without calling get(), so without this the URL ranks as the oldest raw
        # entry and _evict_one_locked() drops it —
        # together with every summary keyed off it — while genuinely cold URLs
        # survive. Same lock, same clock source as get().
        raw_entry["last_accessed"] = time.monotonic()
        return _SUMMARY_CACHE[key]


def put_summary(url: str, query: str, summary: str, *, raw: str | None = None) -> bool:
    """Store a summary only while its raw generation is still current.

    ``raw`` closes the race where a slow summarizer for generation A finishes
    after a refetch has installed generation B. Returns whether it was stored.
    """
    if not url or not summary:
        return False
    key = _summary_key(url, query)
    with _LOCK:
        entry = _RAW_CACHE.get(url)
        if entry is None or _is_expired(entry):
            if entry is not None:
                _purge_url_locked(url)
            return False
        if raw is not None:
            expected = raw
            if len(expected) > WEB_CACHE_PER_ENTRY_MAX:
                expected = expected[:WEB_CACHE_PER_ENTRY_MAX] + "\n...[cache-truncated]"
            if entry["raw"] != expected:
                return False
        if key in _SUMMARY_CACHE:
            _SUMMARY_CACHE.move_to_end(key)
        elif len(_SUMMARY_CACHE) >= _SUMMARY_MAX:
            _SUMMARY_CACHE.popitem(last=False)  # evict least-recently-used
        _SUMMARY_CACHE[key] = summary
        return True


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
    _purge_url_locked(oldest_url)


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
