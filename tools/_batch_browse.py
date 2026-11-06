# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Internal URL-batch engine used only by the public browse_url tool.

Architecture:
  Phase 1 (parallel): HTTP fetch only — no LLM in worker threads.
  Phase 2 (local):    the same deterministic evidence selection browse_url uses.

Cache-aware and LLM-free: a raw-cache hit skips the network; a miss fetches and
caches the raw body once. No query-aware summary is read or written. Every
outbound fetch reserves web budget before its worker is submitted, so a batch
cannot race another tool past the per-turn ceiling; cache hits stay free.
"""
from __future__ import annotations
import sys
import os
from contextvars import copy_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.browse_url import (
    _append_table_evidence,
    _fetch_body,
    _get_browse_cache,
    _select_browse_evidence,
)
from config import (
    BATCH_BROWSE_EVIDENCE_MAX_CHARS as _EVIDENCE_CAP,
    BATCH_BROWSE_MAX_URLS as _MAX_URLS,
    BATCH_BROWSE_MAX_WORKERS as _MAX_WORKERS,
    WEB_LIVE_CACHE_TTL_SECONDS,
)
from tools._bounded_executor import BoundedExecutor
from tools._progress import (
    cancelable_as_completed as _cancelable_as_completed,
    check_cancel as _check_cancel,
    phase as _phase,
    progress as _progress,
)
from tools import web_cache
from tools.web_cache import (
    web_count_check as _wc_check,
    web_count_release_reservation as _wc_release_reservation,
    web_count_reserve_up_to as _wc_reserve_up_to,
)

_HTTP_EXECUTOR = BoundedExecutor(_MAX_WORKERS, thread_name_prefix="batch-http")


def _http_only(url: str, query: str) -> tuple[str, str]:
    """HTTP fetch only, no LLM — safe in a worker thread."""
    return url, _fetch_body(url, query)


def _extract_url(entry: str) -> str:
    """ดึง URL จาก entry ที่ model อาจ copy ทั้งบรรทัดจาก browse_url sitemap mode ("date  url"):
    เอา token แรกที่ขึ้นต้น http; ไม่มี → token แรก (bare domain ให้ขั้น https:// prefix จัดการต่อ)"""
    tokens = entry.split()
    if not tokens:
        return ""
    return next((t for t in tokens if t.startswith("http")), tokens[0])


def browse_urls(urls: list, user_query: str = "") -> str:
    """Fetch a bounded URL list in parallel and return per-URL evidence in request order."""
    if not urls:
        return "[error] urls is required"

    _check_cancel()
    clean = [_extract_url(str(u).strip()) for u in urls if u and str(u).strip()]
    clean = [u for u in clean if u][:_MAX_URLS]
    if not clean:
        return "[error] no valid URLs provided"

    # normalize + dedup — a duplicate URL must not fetch/count/process twice
    clean = [u if u.startswith("http") else "https://" + u for u in clean]
    clean = list(dict.fromkeys(clean))

    # Source scope is a per-turn provenance policy, including cache reads.
    results: dict[str, str] = {}
    requested = clean
    clean = []
    for u in requested:
        source_err = web_cache.web_source_scope_url_error(u)
        if source_err:
            results[u] = source_err
        else:
            clean.append(u)

    uq = (user_query or "").strip()
    from tools.web_search import is_live_market_query, runtime_date_grounding_notice
    cache_max_age = WEB_LIVE_CACHE_TTL_SECONDS if is_live_market_query(uq) else None

    # ── raw cache hit / miss ─────────────────────────────────────────
    pending: list[dict] = []  # [{"url", "raw", "is_new"}]
    to_fetch: list[str] = []
    for u in clean:
        hit_raw = _get_browse_cache(u, max_age_seconds=cache_max_age)
        if hit_raw is not None:
            if hit_raw.strip():
                pending.append({"url": u, "raw": hit_raw, "is_new": False})
            else:
                results[u] = f"[web:{u}] [no usable content cached]"
            continue
        to_fetch.append(u)

    # Reserve each outbound attempt before submitting a worker: a read-only check
    # followed by a post-fetch increment lets another tool take the last slot
    # while this batch is already fetching.
    requested_fetches = list(to_fetch)
    to_fetch = []
    if requested_fetches:
        reserved = _wc_reserve_up_to(len(requested_fetches))
        if reserved < len(requested_fetches):
            err = _wc_check() or "[web_limit] ค้นครบแล้ว — หยุดค้นและสรุปจากข้อมูลที่มีได้เลย"
            _progress(f"web budget: capping fetch to {reserved}/{len(requested_fetches)} URLs")
            for u in requested_fetches[reserved:]:
                results[u] = err
        to_fetch = requested_fetches[:reserved]

    _phase(f"📄 batch browse: {len(to_fetch)} fetch + {len(pending)} cached + {len(results)} ready")

    # ── Phase 1: parallel HTTP fetch (no LLM in the workers) ─────────
    raw_map: dict[str, str] = {}
    submitted: list[str] = []
    # BoundedExecutor admits one task per worker; drain bounded waves so every
    # URL is fetched without growing its queue or the parallelism limit.
    wave_size = max(1, int(_MAX_WORKERS))
    for offset in range(0, len(to_fetch), wave_size):
        wave_futures = {}
        wave_urls = to_fetch[offset:offset + wave_size]
        for u in wave_urls:
            try:
                future = _HTTP_EXECUTOR.submit(copy_context().run, _http_only, u, uq)
            except Exception:
                future = None
            if future is not None:
                wave_futures[future] = u
                submitted.append(u)
            else:
                # No worker accepted it, so no network attempt consumes this reservation.
                _wc_release_reservation()
                results[u] = "[error] browse fetch workers are busy — retry later"
        try:
            done = 0
            for future in _cancelable_as_completed(wave_futures, timeout_seconds=35.0):
                try:
                    done += 1
                    url, raw = future.result()
                    raw_map[url] = raw
                    _progress(f"HTTP {done}/{len(wave_futures)}: {url[:55]}")
                except Exception:
                    pass
        except Exception:
            # Reservations for later waves never became network attempts.
            for u in to_fetch[offset + len(wave_urls):]:
                _wc_release_reservation()
                results[u] = "[error] batch fetch stopped before this URL was submitted"
            break  # timeout/cancellation — affected URLs fall through as "fetch missing"

    # Every submitted attempt keeps its reservation, successful or not, so an
    # error path cannot be used to exceed the per-turn hard ceiling.
    for u in submitted:
        raw = raw_map.get(u, "[error] fetch missing")
        if raw.startswith(("[error]", "[source_blocked]", "[web_limit]")):
            results[u] = raw
            continue
        pending.append({"url": u, "raw": raw, "is_new": True})

    # ── Phase 2: deterministic evidence — no inner LLM ───────────────
    for p in pending:
        raw = p["raw"]
        if p["is_new"]:
            web_cache.put(p["url"], raw, official_verified=web_cache.web_source_scope_official_only())
        evidence = _select_browse_evidence(raw, p["url"], uq, cap=_EVIDENCE_CAP)
        if evidence.startswith(("[error]", "[source_blocked]")):
            results[p["url"]] = evidence
            continue
        if not evidence:
            results[p["url"]] = f"[web:{p['url']}] [no usable evidence]"
            continue
        results[p["url"]] = _append_table_evidence(f"[web:{p['url']}] {evidence}", raw)
        _progress(f"evidence ready: {p['url'][:55]} ({len(evidence)} chars)")

    # ── merge in the original request order ──────────────────────────
    n_fetch = len(submitted)
    lines = [f"[browse_url batch] {len(requested)} URLs ({n_fetch} fetched, {len(requested) - n_fetch} cached)"]
    for u in requested:
        lines.append(results.get(u, f"[error] missing: {u}"))
    output = "\n\n".join(lines)
    date_notice = runtime_date_grounding_notice(f"{uq}\n" + "\n".join(results.values()))
    return f"{date_notice}\n{output}" if date_notice else output
