# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""recall_web.py — retrieve raw body of a previously fetched URL

Messages only carry compact `[web:<url>] <summary>` tags; when the agent
needs more detail it calls recall_web(url) and gets back the raw body only
(the summary is already in the message history — no need to repeat it),
hard-capped at RECALL_WEB_MAX_CHARS.

Cache miss → fetches the body directly (tools.browse_url._fetch_body) and
caches it. Deliberately does NOT go through the browse_url @tool wrapper —
that path ends in summarize() (an LLM call) whose summary recall_web has no
use for (it only wants the raw body), so routing through it would pay an
LLM call just to throw the result away.
"""
from __future__ import annotations
import logging
import sys
import os
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RECALL_WEB_MAX_CHARS
from tools import web_cache
from tools._progress import progress as _progress, phase as _phase

log = logging.getLogger(__name__)


@tool
def recall_web(url: str, query: str = "") -> str:
    """Retrieve the raw body of a previously fetched URL (the summary is already in your history).
    Use when you need details beyond the summary. Returns up to 20,000 chars of raw content.
    Pass query to focus which parts of a long page are returned (plain head truncation otherwise).
    On cache miss it will auto re-fetch the URL once."""
    if not url or not url.strip():
        return "[error] url is required"
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    _phase(f"📑 ดึง cache: {url[:42]}")

    content = web_cache.get(url)
    if content is None:
        _progress(f"recall_web cache MISS — auto re-fetch: {url[:60]}")
        try:
            # Reserve web budget before the actual network call (matches
            # browse_url's own check_and_inc placement — cache hits stay free).
            err = web_cache.web_count_check_and_inc()
            if err:
                return f"[error] recall_web: {err}"
            # Fetch the raw body directly — bypass browse_url's @tool wrapper
            # so a cache-miss recall doesn't pay for a summarize() LLM call
            # whose output would just be discarded (see module docstring).
            from tools.browse_url import _fetch_body
            raw = _fetch_body(url, query)
            if not raw or raw.startswith("[error]"):
                return f"[error] recall_web: {raw or 'empty fetch result'}"
            web_cache.put(url, raw)
            # Re-read from cache rather than reusing `raw` directly so what we
            # return matches exactly what's now cached (put() truncates raw
            # bodies over WEB_CACHE_PER_ENTRY_MAX).
            content = web_cache.get(url)
            if content is None:
                return f"[error] recall_web: fetch succeeded but cache write failed for {url}"
        except Exception as e:
            return f"[error] recall_web auto-fetch failed: {e}"
    else:
        _progress(f"recall_web cache HIT: {url[:60]} ({len(content)} chars)")

    if len(content) > RECALL_WEB_MAX_CHARS:
        if query and query.strip():
            # Query given — sample coverage across the whole body so relevant
            # sections past the old head-only cutoff aren't silently dropped.
            from tools.read_file import _sample_coverage
            content = _sample_coverage(content, url, max_chars=RECALL_WEB_MAX_CHARS, query=query)
        else:
            # No query — keep the original deterministic head-truncation behavior.
            content = content[:RECALL_WEB_MAX_CHARS] + f"\n...[recall truncated at {RECALL_WEB_MAX_CHARS} chars]"
    return content


# Docstring hardcodes "20,000" for readability but the real cap is the
# env-configurable RECALL_WEB_MAX_CHARS (V2_RECALL_WEB_MAX_CHARS) — patch the
# description post-decoration so the model-facing number always matches config.
recall_web.description = recall_web.description.replace("20,000", f"{RECALL_WEB_MAX_CHARS:,}")
