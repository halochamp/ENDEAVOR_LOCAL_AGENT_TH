# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""web_search.py — DDG search → cached, summarized evidence

Fetches top URLs from DuckDuckGo, summarizes each (query-aware), and caches
the raw body so the agent's message context only carries compact tags:
    [web:url1] summary1

    [web:url2] summary2
    ...
Use recall_web(url) to retrieve a full cached body later.

ต่างจาก V1 Endeavor: ไม่ใช้ gemma4 pre-synthesis — 27B agent อ่าน summary
แล้วตัดสินใจเองว่าต้อง recall_web ตัวไหนต่อ
"""
from __future__ import annotations
import re
import time
import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import web_cache
from tools._summarize import summarize

from tools._progress import progress as _progress, phase as _phase
from tools.web_cache import web_count_check as _wc_check, web_count_inc as _wc_inc
from config import (
    WEB_SEARCH_MAX_RESULTS as _SEARCH_MAX,
    WEB_SEARCH_FETCH_TOP as _FETCH_TOP,
    WEB_SEARCH_MAX_CHARS_URL as _MAX_CHARS_URL,
    WEB_SEARCH_FETCH_TIMEOUT as _FETCH_TIMEOUT,
)

log = logging.getLogger(__name__)

_ADULT_DOMAINS = {"xxx", "porn", "sex", "adult", "hentai", "nude", "xhamster", "xvideos",
                  "pornhub", "betflix", "slot", "casino", "betting", "ufabet", "gclub", "lsm99"}
_ADULT_KW = {"คลิปหลุด", "หนังโป๊", "nude", "porn", "xxx", "explicit", "18+",
             "สล็อต", "คาสิโน", "เดิมพัน", "บาคาร่า", "slot online", "gambling"}


def _filter_adult(results: list[dict]) -> list[dict]:
    out = []
    for r in results:
        url = (r.get("url") or "").lower()
        text = ((r.get("title") or "") + " " + (r.get("snippet") or "")).lower()
        if any(d in url for d in _ADULT_DOMAINS) or any(kw in text for kw in _ADULT_KW):
            continue
        out.append(r)
    return out


def _rank_results(results: list[dict], query: str) -> list[dict]:
    q_words = {w.lower() for w in query.split() if len(w) > 2}
    scored = []
    for r in results:
        text = (r.get("title", "") + " " + r.get("snippet", "")).lower()
        kw = sum(1 for w in q_words if w in text)
        ln = min(len(r.get("snippet", "")), 300) / 300
        scored.append((kw * 2 + ln, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]


def _fetch_one(url: str) -> str:
    try:
        import trafilatura
    except ImportError:
        log.warning("[web_search] 'trafilatura' not installed — content fetch disabled. "
                    "Fix: pip install trafilatura")
        return ""
    try:
        import requests
    except ImportError:
        log.warning("[web_search] 'requests' not installed — content fetch disabled. "
                    "Fix: pip install requests")
        return ""
    try:
        resp = requests.get(url, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False,
                                   include_tables=True, output_format="markdown") or ""
        return text[:_MAX_CHARS_URL] + ("\n...[truncated]" if len(text) > _MAX_CHARS_URL else "")
    except Exception as e:
        log.debug(f"[web_search] fetch error {url[:50]}: {e}")
        return ""


@tool
def web_search(query: str, user_query: str = "") -> str:
    """Search the internet for real-time information: current prices, today's news, recent events, facts to look up.
    Returns compact Thai summaries tagged with each source URL — the full bodies are cached.
    Pass user_query with the user's current question to get query-focused summaries.
    If you need more detail than a summary provides, call recall_web(url) to retrieve up to 20,000 chars of that source."""
    query = query.strip()
    if not query:
        return "[error] query is required"
    err = _wc_check()
    if err:
        return err
    _wc_inc()
    _phase(f"🔍 ค้นหา: {query[:45]}")
    # If the agent forgot user_query, fall back to the search query itself —
    # it's a reasonable proxy for the user's intent.
    effective_uq = (user_query or "").strip() or query

    try:
        from ddgs import DDGS
        _progress(f"DDG search: {query[:60]}")
        time.sleep(1.0)  # min delay กัน DDG rate-limit
        raw = list(DDGS().text(query, max_results=_SEARCH_MAX, region="wt-wt"))
    except Exception as e:
        log.warning(f"[web_search] DDG attempt 1 failed: {e} — retry in 3s")
        _progress(f"DDG attempt 1 failed, retrying in 3s: {e}")
        time.sleep(3.0)
        try:
            from ddgs import DDGS
            raw = list(DDGS().text(query, max_results=_SEARCH_MAX, region="wt-wt"))
        except Exception as e2:
            return f"[error] web search unavailable: {e2}"
    if not raw:
        return "(no results found)"
    results = [{"title": r.get("title", ""), "snippet": r.get("body", ""),
                "url": r.get("href", ""), "content": ""} for r in raw]
    results = _filter_adult(results)
    if not results:
        return "(no results after content filtering)"
    ranked = _rank_results(results, query)[:_FETCH_TOP]

    # Parallel HTTP fetch (network is the bottleneck, safe to parallelize)
    # timeout=12 prevents trafilatura.extract() on heavy pages from blocking indefinitely
    _progress(f"fetching top {len(ranked)} URLs in parallel…")
    try:
        ex = ThreadPoolExecutor(max_workers=_FETCH_TOP)
        try:
            futs = {ex.submit(_fetch_one, r["url"]): i for i, r in enumerate(ranked)}
            for f in as_completed(futs, timeout=12):
                try:
                    ranked[futs[f]]["content"] = f.result(timeout=12)
                except Exception:
                    pass
        finally:
            ex.shutdown(wait=False)
    except Exception:
        pass  # timeout or executor error — use snippet fallback in summarization

    # Summarization is sequential — local mlx server serves one request at a time,
    # parallel calls would just queue up and add overhead.
    parts = []
    summ_t0 = time.time()
    for idx, r in enumerate(ranked, 1):
        url = r["url"]
        if not url:
            continue
        body = r.get("content") or r.get("snippet") or ""
        if not body:
            continue

        # Check query-aware summary first, then raw cache for re-summarize
        summary = web_cache.get_summary(url, effective_uq)
        if summary is not None:
            _progress(f"[{idx}/{len(ranked)}] summary HIT (query-aware): {url[:60]}")
        else:
            raw_cached = web_cache.get(url)
            if raw_cached is not None:
                # Raw exists but no summary for this query → re-summarize from cache
                _progress(f"[{idx}/{len(ranked)}] raw HIT, re-summarize for new query: {url[:60]}")
                summary = summarize(raw_cached, url=url, user_query=effective_uq)
            else:
                _progress(f"[{idx}/{len(ranked)}] summarizing: {url[:60]}")
                raw_with_title = f"# {r.get('title', '')}\n\n{body}"
                summary = summarize(raw_with_title, url=url, user_query=effective_uq)
                web_cache.put(url, raw_with_title)
            web_cache.put_summary(url, effective_uq, summary)
        parts.append(f"[web:{url}] {summary}")
    _progress(f"summarized {len(parts)} URLs in {time.time() - summ_t0:.1f}s")
    log.info(f"[web_search] summarized {len(parts)} URLs in {time.time() - summ_t0:.1f}s")

    if not parts:
        return "(no usable content fetched)"
    return "\n\n".join(parts)
