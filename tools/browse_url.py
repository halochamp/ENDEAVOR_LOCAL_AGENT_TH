# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""browse_url.py — fetch any URL → clean markdown

Direct fetch (requests + trafilatura) first — fast and local; falls back to
Jina Reader (r.jina.ai) for JS-rendered pages (thin direct extract) and
direct-fetch failures. ใช้สำหรับ: อ่านบทความ, ดึงเนื้อหาจาก URL เฉพาะ

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
from tools.web_cache import web_count_check_and_inc as _wc_check_and_inc
from config import BROWSE_URL_MAX_CHARS, RECALL_WEB_MAX_CHARS

log = logging.getLogger(__name__)

_JINA = "https://r.jina.ai/"

# Direct extract shorter than this usually means a JS-shell page whose real
# content needs rendering — fall through to Jina Reader for those.
_MIN_DIRECT_CHARS = 800


def _fetch_jina(url: str, query: str = "") -> str:
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
            content = _sample_coverage(content, url, max_chars=BROWSE_URL_MAX_CHARS, query=query)
        _progress(f"fetched {len(content)} chars in {time.time()-t0:.1f}s")
        return content
    except Exception as e:
        _progress(f"Jina fetch failed: {e}")
        return f"[error] browse_url failed: {e}"


def _fetch_body(url: str, query: str = "") -> str:
    """Direct fetch (requests + trafilatura — fast, local) first; Jina Reader
    fallback for JS-heavy pages and fetch failures. Returns body or '[error] ...'."""
    from tools.web_search import _fetch_one  # lazy — avoids import cycle at boot
    direct = _fetch_one(url)
    if len(direct) >= _MIN_DIRECT_CHARS:
        _progress(f"direct fetch OK ({len(direct)} chars) — skip Jina")
        return direct
    jina = _fetch_jina(url, query)
    if jina.startswith("[error]") and direct:
        _progress(f"Jina failed — keeping thin direct content ({len(direct)} chars)")
        return direct
    return jina


def _split_browse_result(result: str) -> tuple[str, str, bool]:
    """Parse browse_url's output contract into (url, body, ok)."""
    if result.startswith("[web_limit]"):
        return "", result, False
    if not result.startswith("[web:"):
        return "", result, False
    bracket_end = result.find("] ")
    if bracket_end == -1:
        return "", result, False
    return result[5:bracket_end], result[bracket_end + 2:], True


@tool
def browse_url(url: str, user_query: str = "") -> str:
    """Fetch and read one specific URL; return a compact Thai summary tagged with the URL.

    Use when the user gives a URL, or after web_search returns a result whose full
    page is needed. Do not use browser_use for normal article/page reading; browser_use
    is only for login/forms/click interaction or when browse_url clearly cannot read
    enough content.

    Always pass user_query with the current user question so the summary is focused.
    The full cleaned body is cached; if the summary lacks detail, call recall_web(url)
    instead of re-calling browse_url on the same URL.
    """
    if not url or not url.strip():
        return "[error] url is required"
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
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
        web_cache.put_summary(url, effective_uq, summary, raw=raw)
        return f"[web:{url}] {summary}"

    # Full miss → actual network fetch — only now consume web budget
    # (cache hits above are free, matching batch_browse's counting semantics)
    err = _wc_check_and_inc()
    if err:
        return err
    _progress(f"cache MISS: {url[:70]}")
    raw = _fetch_body(url, effective_uq)
    if raw.startswith("[error]"):
        return raw

    summary = summarize(raw, url=url, user_query=effective_uq or None)
    web_cache.put(url, raw)
    web_cache.put_summary(url, effective_uq, summary, raw=raw)
    _progress(f"cached ({len(raw)} raw + {len(summary)} summary)")
    return f"[web:{url}] {summary}"


# Docstring is model-facing and hardcodes "20,000" for readability, but the
# actual cap is recall_web's RECALL_WEB_MAX_CHARS (env-configurable via
# V2_RECALL_WEB_MAX_CHARS) — patch the description post-decoration so the
# number the model sees always matches the real config, not a stale literal.
browse_url.description = browse_url.description.replace("20,000", f"{RECALL_WEB_MAX_CHARS:,}")
