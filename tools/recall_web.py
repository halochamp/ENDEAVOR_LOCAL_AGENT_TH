# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""recall_web.py — retrieve raw body of a previously fetched URL

web_search messages carry compact `[web:<url>]` evidence passages; when the
agent needs more detail it calls recall_web(url) and gets back the raw body only
(the extracted evidence is already in message history — no need to repeat it),
hard-capped at RECALL_WEB_MAX_CHARS.

Cache miss → fetches the body directly (tools.browse_url._fetch_body) and
caches it. Deliberately does NOT go through the browse_url @tool wrapper because
recall_web needs raw/deeper content, while browse_url intentionally returns only
compact deterministic evidence.
"""
from __future__ import annotations
import logging
import sys
import os
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RECALL_WEB_MAX_CHARS, WEB_LIVE_CACHE_TTL_SECONDS
from tools import web_cache
from tools._progress import progress as _progress, phase as _phase

log = logging.getLogger(__name__)


@tool
def recall_web(url: str, query: str = "") -> str:
    """Read deeper raw content for a URL already fetched by web_search/browse_url; query focuses long pages, up to 20k chars. Prefer this over re-fetching the same URL. Cache miss auto-refetches once.
    """
    if not url or not url.strip():
        return "[error] url is required"
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    source_err = web_cache.web_source_scope_url_error(url)
    if source_err:
        return source_err
    _phase(f"📑 ดึง cache: {url[:42]}")

    from tools.web_search import is_live_market_query
    cache_max_age = WEB_LIVE_CACHE_TTL_SECONDS if is_live_market_query(query) else None
    from tools.browse_url import _get_browse_cache
    content = _get_browse_cache(url, max_age_seconds=cache_max_age)
    if content is None:
        _progress(f"recall_web cache MISS — auto re-fetch: {url[:60]}")
        try:
            # Reserve web budget before the actual network call (matches
            # browse_url's own check_and_inc placement — cache hits stay free).
            err = web_cache.web_count_check_and_inc()
            if err:
                return f"[error] recall_web: {err}"
            # Fetch the raw body directly — bypass browse_url's @tool wrapper
            # because recall_web needs the raw/deeper body, not compact evidence.
            from tools.browse_url import _fetch_body
            raw = _fetch_body(url, query)
            if raw.startswith("[source_blocked]"):
                return raw
            if not raw or raw.startswith("[error]"):
                return f"[error] recall_web: {raw or 'empty fetch result'}"
            web_cache.put(url, raw, official_verified=web_cache.web_source_scope_official_only())
            # Re-read from cache rather than reusing `raw` directly so what we
            # return matches exactly what's now cached (put() truncates raw
            # bodies over WEB_CACHE_PER_ENTRY_MAX).
            content = web_cache.get(url, max_age_seconds=cache_max_age)
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
    from tools.web_search import runtime_date_grounding_notice
    date_notice = runtime_date_grounding_notice(f"{query}\n{content}")
    return f"{date_notice}\n{content}" if date_notice else content


# Docstring hardcodes "20,000" for readability but the real cap is the
# env-configurable RECALL_WEB_MAX_CHARS (V2_RECALL_WEB_MAX_CHARS) — patch the
# description post-decoration so the model-facing number always matches config.
recall_web.description = recall_web.description.replace("20,000", f"{RECALL_WEB_MAX_CHARS:,}")
