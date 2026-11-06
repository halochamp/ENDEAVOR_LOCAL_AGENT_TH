# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""browse_url.py — fetch any URL → compact deterministic evidence

Direct fetch (requests + trafilatura) first — fast and local; falls back to
Jina Reader (r.jina.ai) for JS-rendered pages (thin direct extract) and
direct-fetch failures. ใช้สำหรับ: อ่านบทความ, ดึงเนื้อหาจาก URL เฉพาะ

Cache-aware and LLM-free: cache the cleaned raw body once, then select the
passages that answer user_query with the same deterministic evidence selector
web_search uses. Returns `[web:<url>] <evidence>` to keep the message context
small. Use recall_web(url) to retrieve deeper raw content later.
"""
from __future__ import annotations
import json
import urllib.request
import logging
import sys
import os
import re
import time
from typing import Literal
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import web_cache
from tools._progress import progress as _progress, phase as _phase
from tools.web_cache import web_count_check_and_inc as _wc_check_and_inc
from config import (
    BROWSE_URL_MAX_CHARS,
    RECALL_WEB_MAX_CHARS,
    WEB_DEEP_CACHE_MIN_CHARS,
    WEB_EVIDENCE_MAX_CHARS,
    WEB_LIVE_CACHE_TTL_SECONDS,
)

log = logging.getLogger(__name__)

_JINA = "https://r.jina.ai/"

# Direct extract shorter than this usually means a JS-shell page whose real
# content needs rendering — fall through to Jina Reader for those. Shared with
# web_search's raw-cache admission so the two tools cannot drift.
_MIN_DIRECT_CHARS = WEB_DEEP_CACHE_MIN_CHARS


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
    if direct.startswith("[source_blocked]"):
        return direct
    if web_cache.web_source_scope_official_only():
        # Jina is a server-side reader: it does not expose the target's final
        # redirect URL to us, so using it here would turn an initial allowlist
        # check into an unverifiable provenance claim. Keep a short verified
        # direct body, or fail transparently if the endpoint cannot be read
        # directly.
        if direct:
            _progress(f"official source direct fetch ({len(direct)} chars) — skip Jina")
            return direct
        return "[error] official-only source could not be read directly; Jina fallback is disabled for provenance safety"
    if len(direct) >= _MIN_DIRECT_CHARS:
        _progress(f"direct fetch OK ({len(direct)} chars) — skip Jina")
        return direct
    jina = _fetch_jina(url, query)
    if jina.startswith("[error]") and direct:
        _progress(f"Jina failed — keeping thin direct content ({len(direct)} chars)")
        return direct
    return jina


def _get_browse_cache(url: str, max_age_seconds: float | None = None) -> str | None:
    return web_cache.get(url, max_age_seconds=max_age_seconds)


def _select_browse_evidence(raw: str, url: str, query: str, cap: int = WEB_EVIDENCE_MAX_CHARS) -> str:
    """The one evidence selector shared by single reads and URL batches."""
    from tools.web_search import select_web_evidence  # lazy — avoids import cycle at boot
    return select_web_evidence(raw, query, cap=cap, fallback="coverage")


def _normalise_http_url(value: str) -> str:
    url = (value or "").strip()
    if not url:
        return ""
    if "://" in url and not url.startswith(("http://", "https://")):
        return ""
    return url if url.startswith(("http://", "https://")) else "https://" + url


_TABLE_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,.]*(?:%|\b)")


def _markdown_table_evidence(raw: str) -> str:
    """Return a bounded useful Markdown table already present in fetched text."""
    rows: list[list[str]] = []
    candidates: list[list[str]] = []

    def flush() -> None:
        nonlocal rows
        if len(rows) >= 2 and len(rows[0]) >= 2:
            width = len(rows[0])
            clean = [row for row in rows if len(row) == width]
            body = [row for row in clean[1:] if not all(re.fullmatch(r"[:\-\s]*", c or "") for c in row)]
            if body and any(_TABLE_NUMBER_RE.search(cell) for row in body for cell in row):
                candidates.append(clean[:31])
        rows = []

    for line in (raw or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            rows.append(cells)
        else:
            flush()
    flush()
    if not candidates:
        return ""
    table = "\n".join("| " + " | ".join(row) + " |" for row in candidates[0])
    if len(table) > 3000:
        table = table[:3000].rsplit("\n", 1)[0]
    return f"\n\n[table evidence]\n{table}"


def _append_table_evidence(result: str, raw: str) -> str:
    evidence = _markdown_table_evidence(raw)
    return result + evidence if evidence else result


@tool
def browse_url(
    url: str = "",
    user_query: str = "",
    urls: list[str] | str | None = None,
    mode: Literal["read", "sitemap"] = "read",
    filter_keyword: str = "",
    table_index: int | None = None,
) -> str:
    """Read web pages through one consolidated web tool.

    Normal mode reads one URL. Pass urls=[...] to read a bounded batch of pages in
    one call; a JSON-encoded array of strings is also accepted. Set mode="sitemap"
    with url and optional filter_keyword to list URLs
    from a site's sitemap. For a rendered page table, pass table_index=-1 to list
    tables or a non-negative table_index to extract one table as bounded CSV.

    Always pass user_query with the current question: pages are returned as
    compact verbatim evidence selected for it (no inner LLM summarizer). Use
    recall_web(url) for deeper raw content from a page already read.
    Use browser_use only for interaction such as login/forms or when direct reading
    cannot provide the requested content. Useful numeric Markdown tables in a normal
    read are attached as bounded evidence automatically.
    """
    selected_mode = (mode or "read").strip().lower()
    if selected_mode == "sitemap":
        if urls is not None or table_index is not None:
            return "[error] sitemap mode accepts url and filter_keyword only"
        sitemap_url = _normalise_http_url(url)
        if not sitemap_url:
            return "[error] a valid http(s) url is required for sitemap mode"
        from tools._fetch_sitemap import read_sitemap
        return read_sitemap(sitemap_url, filter_keyword)
    if selected_mode != "read":
        return "[error] mode must be 'read' or 'sitemap'"
    if filter_keyword:
        return "[error] filter_keyword is only valid in sitemap mode"

    if urls is not None:
        if url.strip():
            return "[error] provide url or urls, not both"
        if table_index is not None:
            return "[error] table_index is only valid when reading one url"
        if isinstance(urls, str):
            try:
                decoded_urls = json.loads(urls)
            except (TypeError, ValueError):
                return "[error] urls must be a list of strings or a JSON-encoded list of strings"
            if not isinstance(decoded_urls, list) or any(not isinstance(item, str) for item in decoded_urls):
                return "[error] urls must be a list of strings or a JSON-encoded list of strings"
            urls = decoded_urls
        if not isinstance(urls, list):
            return "[error] urls must be a list of strings"
        from tools._batch_browse import browse_urls
        return browse_urls(urls, user_query)

    url = _normalise_http_url(url)
    if not url:
        return "[error] a valid http(s) url is required"

    if table_index is not None:
        from tools._scrape_table import extract_table
        return extract_table(url, table_index)

    source_err = web_cache.web_source_scope_url_error(url)
    if source_err:
        return source_err
    _phase(f"📄 อ่านเว็บ: {url[:45]}")

    effective_uq = (user_query or "").strip()
    from tools.web_search import is_live_market_query, runtime_date_grounding_notice
    cache_max_age = WEB_LIVE_CACHE_TTL_SECONDS if is_live_market_query(effective_uq) else None

    # Raw cache is the only browse_url cache layer: no LLM summary is made or stored.
    raw = _get_browse_cache(url, max_age_seconds=cache_max_age)
    if raw is not None:
        _progress(f"raw cache HIT: {url[:70]}")
    else:
        # Full miss → actual network fetch — only now consume web budget
        # (cache hits are free, matching the URL batch's counting semantics)
        err = _wc_check_and_inc()
        if err:
            return err
        _progress(f"cache MISS: {url[:70]}")
        raw = _fetch_body(url, effective_uq)
        if raw.startswith(("[source_blocked]", "[error]")):
            return raw
        web_cache.put(url, raw, official_verified=web_cache.web_source_scope_official_only())
        _progress(f"cached raw ({len(raw)} chars)")

    evidence = _select_browse_evidence(raw, url, effective_uq)
    if evidence.startswith(("[error]", "[source_blocked]")):
        return evidence
    if not evidence:
        return "[error] page contained no readable evidence"
    _progress(f"evidence ready ({len(evidence)} chars)")
    date_notice = runtime_date_grounding_notice(f"{effective_uq}\n{evidence}")
    payload = _append_table_evidence(f"[web:{url}] {evidence}", raw)
    return f"{date_notice}\n{payload}" if date_notice else payload


# Docstring is model-facing and hardcodes "20,000" for readability, but the
# actual cap is recall_web's RECALL_WEB_MAX_CHARS (env-configurable via
# V2_RECALL_WEB_MAX_CHARS) — patch the description post-decoration so the
# number the model sees always matches the real config, not a stale literal.
browse_url.description = browse_url.description.replace("20,000", f"{RECALL_WEB_MAX_CHARS:,}")
