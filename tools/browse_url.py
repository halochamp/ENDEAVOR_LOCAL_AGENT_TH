"""browse_url.py — fetch any URL via Jina Reader (r.jina.ai) → clean markdown

ไม่ต้องเปิด browser — Jina server render JS ให้แล้วคืน clean text
ใช้สำหรับ: อ่านบทความ, ดึงเนื้อหาจาก URL เฉพาะ, หน้าที่ JS-rendered

Cache-aware: on first fetch summarizes the raw body (query-aware if
user_query is provided) and stores raw + summary separately in the session
cache. Returns only `[web:<url>] <summary>` to keep the message context
small. Use recall_web(url) to retrieve the full body later.
"""
from __future__ import annotations
import urllib.request
import logging
import sys
import os
import time
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import web_cache
from tools._summarize import summarize
from tools._progress import progress as _progress, phase as _phase
from tools.web_cache import web_count_check as _wc_check, web_count_inc as _wc_inc
from config import BROWSE_URL_MAX_CHARS

log = logging.getLogger(__name__)

_JINA = "https://r.jina.ai/"


def _fetch_jina(url: str) -> str:
    """Raw Jina fetch. Returns body or '[error] ...'. No caching, no summarization."""
    try:
        _progress(f"fetching via Jina Reader: {url[:70]}")
        t0 = time.time()
        req = urllib.request.Request(
            _JINA + url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/plain"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        content = content.strip()
        if not content:
            _progress("Jina returned empty")
            return "[error] empty response from Jina Reader"
        if len(content) > BROWSE_URL_MAX_CHARS:
            from tools.read_file import _sample_coverage
            content = _sample_coverage(content, url, max_chars=BROWSE_URL_MAX_CHARS)
        _progress(f"fetched {len(content)} chars in {time.time()-t0:.1f}s")
        return content
    except Exception as e:
        _progress(f"Jina fetch failed: {e}")
        return f"[error] browse_url failed: {e}"


@tool
def browse_url(url: str, user_query: str = "") -> str:
    """Fetch and read a specific URL via Jina Reader. Returns a compact Thai summary tagged with the URL — the full body is cached.
    Pass user_query with the user's current question to get a query-focused summary.
    If you need more detail than the summary provides, call recall_web(url) to retrieve up to 20,000 chars of full body."""
    if not url or not url.strip():
        return "[error] url is required"
    err = _wc_check()
    if err:
        return err
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    _wc_inc()
    _phase(f"📄 อ่านเว็บ: {url[:45]}")

    effective_uq = (user_query or "").strip()

    # Check query-aware summary first
    summary = web_cache.get_summary(url, effective_uq)
    if summary is not None:
        _progress(f"summary HIT (query-aware): {url[:70]}")
        return f"[web:{url}] {summary}"

    # Raw cached but no summary for this query → re-summarize
    raw = web_cache.get(url)
    if raw is not None:
        _progress(f"raw HIT, re-summarize for new query: {url[:70]}")
        summary = summarize(raw, url=url, user_query=effective_uq or None)
        web_cache.put_summary(url, effective_uq, summary)
        return f"[web:{url}] {summary}"

    # Full miss → fetch + summarize + cache both
    _progress(f"cache MISS: {url[:70]}")
    raw = _fetch_jina(url)
    if raw.startswith("[error]"):
        return raw

    summary = summarize(raw, url=url, user_query=effective_uq or None)
    web_cache.put(url, raw)
    web_cache.put_summary(url, effective_uq, summary)
    _progress(f"cached ({len(raw)} raw + {len(summary)} summary)")
    return f"[web:{url}] {summary}"
