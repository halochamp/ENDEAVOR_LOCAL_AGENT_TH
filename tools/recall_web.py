"""recall_web.py — retrieve raw body of a previously fetched URL

Messages only carry compact `[web:<url>] <summary>` tags; when the agent
needs more detail it calls recall_web(url) and gets back the raw body only
(the summary is already in the message history — no need to repeat it),
hard-capped at RECALL_WEB_MAX_CHARS.

Cache miss → auto re-fetches via browse_url, then reads raw from cache.
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
def recall_web(url: str) -> str:
    """Retrieve the raw body of a previously fetched URL (the summary is already in your history).
    Use when you need details beyond the summary. Returns up to 20,000 chars of raw content.
    On cache miss it will auto re-fetch the URL once."""
    if not url or not url.strip():
        return "[error] url is required"
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    _phase(f"📑 ดึง cache: {url[:42]}")

    content = web_cache.get(url)
    if content is None:
        _progress(f"recall_web cache MISS — auto re-fetch: {url[:60]}")
        # Cache miss — re-fetch via browse_url (populates cache as side effect).
        # Lazy import to avoid circular dep at module load.
        try:
            from tools.browse_url import browse_url
            fetch_result = browse_url.invoke({"url": url})
        except Exception as e:
            return f"[error] recall_web auto-fetch failed: {e}"
        content = web_cache.get(url)
        if content is None:
            if isinstance(fetch_result, str) and fetch_result.startswith("[error]"):
                return f"[error] recall_web: {fetch_result}"
            return f"[error] recall_web: could not fetch or cache {url}"
    else:
        _progress(f"recall_web cache HIT: {url[:60]} ({len(content)} chars)")

    if len(content) > RECALL_WEB_MAX_CHARS:
        content = content[:RECALL_WEB_MAX_CHARS] + f"\n...[recall truncated at {RECALL_WEB_MAX_CHARS} chars]"
    return content
