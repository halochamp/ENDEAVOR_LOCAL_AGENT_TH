"""batch_browse.py — fetch หลาย URLs แล้วคืน summaries รวมใน 1 tool call

Architecture (แก้ปัญหา LLM thread pile-up):
  Phase 1 (parallel):   HTTP fetch เท่านั้น — ไม่มี LLM ใน thread
  Phase 2 (sequential): summarize ทีละ URL — MLX server ไม่ต้องรับ concurrent request

Cache-aware: URL ที่ cache hit ข้ามทั้ง 2 phase, ไม่นับ web counter
"""
from __future__ import annotations
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from tools.browse_url import _fetch_jina
from tools._summarize import summarize
from config import SUMMARY_MAX_CHARS, BATCH_BROWSE_MAX_WORKERS as _MAX_WORKERS, BATCH_BROWSE_MAX_URLS as _MAX_URLS
from tools._progress import phase as _phase, progress as _progress
from tools import web_cache
from tools.web_cache import web_count_check as _wc_check, web_count_inc as _wc_inc


def _http_only(url: str) -> tuple[str, str]:
    """HTTP fetch เท่านั้น ไม่มี LLM — safe ใน thread"""
    return url, _fetch_jina(url)


@tool
def batch_browse(urls: list, user_query: str = "") -> str:
    """Fetch หลาย URLs แล้วคืน summaries รวมใน 1 tool call
    ใช้แทน browse_url ทีละตัวเมื่อมี URL list จาก fetch_sitemap หรือ search results

    urls: list ของ URLs ที่ต้องการ fetch (สูงสุด BATCH_BROWSE_MAX_URLS, default 8)
    user_query: คำถามปัจจุบัน เพื่อให้ summary ตรงประเด็น
    คืน: summaries รวมของทุก URL หรือ "[error] reason"
    """
    if not urls:
        return "[error] urls is required"

    clean = [str(u).strip() for u in urls if u and str(u).strip()][:_MAX_URLS]
    if not clean:
        return "[error] no valid URLs provided"

    # normalize URLs
    clean = [u if u.startswith("http") else "https://" + u for u in clean]

    # ── แยก cache hit vs miss ──────────────────────────────────────────
    cached_results: dict[str, str] = {}
    to_fetch: list[str] = []
    for u in clean:
        hit_raw = web_cache.get(u)
        if hit_raw is not None:
            hit_sum = (web_cache.get_summary(u, user_query or "")
                       or web_cache.get_summary(u, ""))
            if hit_sum is None:
                stripped = hit_raw.strip()
                hit_sum = stripped[:SUMMARY_MAX_CHARS] if stripped else "[no usable content cached]"
            cached_results[u] = f"[web:{u}] {hit_sum}"
        else:
            to_fetch.append(u)

    if to_fetch:
        err = _wc_check()
        if err:
            if cached_results:
                partial = "\n\n".join(cached_results[u] for u in clean if u in cached_results)
                return f"[batch_browse] web limit — คืน {len(cached_results)} cached URLs\n\n{partial}"
            return err

    n_fetch = len(to_fetch)
    n_cached = len(cached_results)
    _phase(f"📄 batch browse: {n_fetch} fetch + {n_cached} cached")

    # ── Phase 1: parallel HTTP fetch (ไม่มี LLM ใน thread) ───────────
    raw_map: dict[str, str] = {}
    if to_fetch:
        try:
            with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, n_fetch)) as pool:
                futures = {pool.submit(_http_only, u): u for u in to_fetch}
                done = 0
                for future in as_completed(futures, timeout=35):
                    try:
                        done += 1
                        url, raw = future.result(timeout=35)
                        raw_map[url] = raw
                        _progress(f"HTTP {done}/{n_fetch}: {url[:55]}")
                    except Exception:
                        pass
        except Exception:
            pass  # timeout or executor error — use empty raw for affected URLs

    # ── Phase 2: sequential summarize + cache (LLM ทีละ URL) ─────────
    fetch_results: dict[str, str] = {}
    for url in to_fetch:
        raw = raw_map.get(url, "[error] fetch missing")
        if raw.startswith("[error]"):
            fetch_results[url] = raw
            continue
        _wc_inc()   # นับเฉพาะ fetch สำเร็จ ไม่นับ error/timeout
        _phase(f"📄 summarize: {url[:50]}")
        s = summarize(raw, url=url, user_query=user_query or None)
        web_cache.put(url, raw)
        web_cache.put_summary(url, user_query or "", s)
        fetch_results[url] = f"[web:{url}] {s}"

    # ── รวม results ตาม order เดิม ────────────────────────────────────
    all_results = {**cached_results, **fetch_results}
    lines = [f"[batch_browse] {len(clean)} URLs ({n_fetch} fetched, {n_cached} cached)"]
    for u in clean:
        lines.append(all_results.get(u, f"[error] missing: {u}"))

    return "\n\n".join(lines)
