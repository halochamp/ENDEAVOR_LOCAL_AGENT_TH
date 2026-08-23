# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""web_search.py — multi-engine search → cached, summarized evidence

Fetches top URLs from search engines (ddgs: auto engine rotation, with an
explicit bing/brave/google fallback pass when the primary attempts fail),
summarizes each (query-aware), and caches the FULL raw body so the agent's
message context only carries compact tags:
    [web:url1] summary1

    [web:url2] summary2
    ...
Use recall_web(url) to retrieve a full cached body later (up to 20k chars —
the cache stores the full fetched body, same cap as browse_url).

ต่างจาก V1 Endeavor: ไม่ใช้ gemma4 pre-synthesis — 27B agent อ่าน summary
แล้วตัดสินใจเองว่าต้อง recall_web ตัวไหนต่อ
"""
from __future__ import annotations
import re
import time
import logging
import socket
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import web_cache
from tools._summarize import summarize, summarize_batch, _query_terms, _score_chunk

from tools._progress import progress as _progress, phase as _phase
from tools.web_cache import web_count_check_and_inc as _wc_check_and_inc
from config import (
    WEB_SEARCH_MAX_RESULTS as _SEARCH_MAX,
    WEB_SEARCH_FETCH_TOP as _FETCH_TOP,
    WEB_SEARCH_FETCH_TIMEOUT as _FETCH_TIMEOUT,
    # Full-body cap shared with browse_url — recall_web promises up to this
    # much from cache, so web_search must store full bodies, not a 1.5k cut.
    BROWSE_URL_MAX_CHARS as _FULL_BODY_CAP,
)

log = logging.getLogger(__name__)

_ADULT_DOMAINS = {"xxx", "porn", "sex", "adult", "hentai", "nude", "xhamster", "xvideos",
                  "pornhub", "betflix", "slot", "casino", "betting", "ufabet", "gclub", "lsm99"}
_ADULT_KW = {"คลิปหลุด", "หนังโป๊", "nude", "porn", "xxx", "explicit", "18+",
             "สล็อต", "คาสิโน", "เดิมพัน", "บาคาร่า", "slot online", "gambling"}

_THAI_RE = re.compile(r"[฀-๿]")

# Sites 403 the default python-requests UA; a browser UA fixes most of them.
_FETCH_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept-Language": "th,en;q=0.8",
}

# Explicit time words only — broader triggers (e.g. "ราคา"/"price") risk
# timelimit dropping undated-but-fresh quote pages. English needs \b word
# boundaries ("now" is inside "snowflake"/"known"); Thai has no word
# boundaries, so substring match is intended there.
_FRESH_EN_RE = re.compile(r"\b(?:today|latest|now|current|breaking|this week)\b")
_FRESH_TH = ("วันนี้", "ล่าสุด", "ตอนนี้", "เมื่อวาน", "สัปดาห์นี้", "เดือนนี้")
_RECENCY_VALUES = {"d", "w", "m", "y"}


def _is_time_sensitive(text: str) -> bool:
    t = text.lower()
    return bool(_FRESH_EN_RE.search(t)) or any(kw in t for kw in _FRESH_TH)

# (backend, pre-delay). "auto" rotates engines; the last pass skips
# duckduckgo entirely in case it is the rate-limited one.
_SEARCH_ATTEMPTS = [("auto", 0.0), ("auto", 3.0), ("bing, brave, google", 3.0)]


def _is_no_results_error(exc: Exception) -> bool:
    return "No results found" in str(exc)


def _ddg_reachable(host: str = "duckduckgo.com", port: int = 443, timeout: float = 3.0) -> bool:
    """TCP-connect check — the sandboxed-process equivalent of a ping, used
    only when every backend attempt already failed, to tell a network-level
    block apart from a rate-limit/API error."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _filter_adult(results: list[dict]) -> list[dict]:
    out = []
    for r in results:
        url = (r.get("url") or "").lower()
        text = ((r.get("title") or "") + " " + (r.get("snippet") or "")).lower()
        if any(d in url for d in _ADULT_DOMAINS) or any(kw in text for kw in _ADULT_KW):
            continue
        out.append(r)
    return out


# Date tokens ("23", "สิงหาคม", "2569") are not topical discriminators: every
# page published that day carries them in its title, so a topically-unrelated
# dated page outscores the right one. Observed twice on this exact shape — a
# daily-horoscope page (2026-08-23) and a TV programme listing (2026-08-21)
# each beat the actual forecast for "สภาพอากาศ… <date>", and each cost a full
# extra web_search round-trip when the agent re-searched. Freshness is enforced
# upstream by the engine's own time filter (timelimit / tbs=qdr), never by this
# ranker, so ranking drops them.
_TH_MONTHS = frozenset((
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
))
_EN_MONTHS = frozenset((
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
))


def _is_numeric_date(term: str) -> bool:
    """A bare number that reads as a day (1-31), a CE year, or a BE year."""
    if not term.isdigit():
        return False
    n = int(term)
    return 1 <= n <= 31 or 1900 <= n <= 2100 or 2400 <= n <= 2600


def _is_date_term(term: str, has_numeric_date: bool) -> bool:
    """True for a token that carries only a date. English month names double as
    ordinary words ("may", "march"), so they count as a date only alongside a
    numeric day/year token; Thai month names are unambiguous on their own."""
    if _is_numeric_date(term) or term in _TH_MONTHS:
        return True
    return has_numeric_date and term in _EN_MONTHS


def _rank_results(results: list[dict], query: str) -> list[dict]:
    # _query_terms/_score_chunk from _summarize handle Thai queries (no word
    # boundaries) via 4-gram fragments — plain query.split() scores 0 on Thai.
    terms = _query_terms(query)
    has_numeric_date = any(_is_numeric_date(t) for t, _ in terms)
    topic_terms = [(t, g) for t, g in terms if not _is_date_term(t, has_numeric_date)]
    # A query that is nothing but a date leaves no topical signal — there the
    # date tokens are all we have, so keep them rather than ranking on length.
    if topic_terms:
        terms = topic_terms
    scored = []
    for r in results:
        text = (r.get("title", "") + " " + r.get("snippet", "")).lower()
        kw = _score_chunk(text, terms)
        ln = min(len(r.get("snippet", "")), 300) / 300
        scored.append((kw + ln, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]


def _domain(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return url


def _select_top(ranked: list[dict], n: int) -> tuple[list[dict], list[dict]]:
    """Pick top n with unique domains; remainder (rank order) is the backfill pool.

    If there are fewer unique domains than n, fill the shortfall with
    duplicate-domain results rather than fetching fewer URLs.
    """
    top: list[dict] = []
    rest: list[dict] = []
    seen: set[str] = set()
    for r in ranked:
        dom = _domain(r["url"])
        if len(top) < n and dom not in seen:
            seen.add(dom)
            top.append(r)
        else:
            rest.append(r)
    while len(top) < n and rest:
        top.append(rest.pop(0))
    return top, rest


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
        resp = requests.get(url, timeout=_FETCH_TIMEOUT, headers=_FETCH_HEADERS)
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False,
                                   include_tables=True, output_format="markdown") or ""
        return text[:_FULL_BODY_CAP] + ("\n...[truncated]" if len(text) > _FULL_BODY_CAP else "")
    except Exception as e:
        log.debug(f"[web_search] fetch error {url[:50]}: {e}")
        return ""


def _fetch_batch(rs: list[dict]) -> None:
    """Parallel HTTP fetch into each r["content"] (network is the bottleneck).

    timeout=12 prevents trafilatura.extract() on heavy pages from blocking
    indefinitely; failures leave content empty (snippet fallback downstream).
    """
    if not rs:
        return
    _progress(f"fetching {len(rs)} URLs in parallel…")
    try:
        ex = ThreadPoolExecutor(max_workers=len(rs))
        try:
            futs = {ex.submit(_fetch_one, r["url"]): r for r in rs}
            for f in as_completed(futs, timeout=12):
                try:
                    futs[f]["content"] = f.result(timeout=12)
                except Exception:
                    pass
        finally:
            # wait=False — don't let __exit__ block on slow fetches past the 12s budget
            ex.shutdown(wait=False)
    except Exception:
        pass  # timeout or executor error — use snippet fallback in summarization


def _search_engines(query: str, region: str, timelimit: str | None) -> list[dict] | str:
    """Run the search with engine fallback. Returns raw result list, or an
    "[error] ..." / "(no results found)" string when every attempt fails."""
    try:
        from ddgs import DDGS
    except Exception as e:  # Tool Output Contract: never raise out of the tool
        log.warning(f"[web_search] ddgs unavailable: {e}")
        return f"[error] web search unavailable: {e}"
    last_err: Exception | None = None
    got_empty = False
    for backend, delay in _SEARCH_ATTEMPTS:
        if delay:
            time.sleep(delay)
        try:
            web_cache.ddg_wait()  # adaptive min-gap delay กัน rate-limit
            raw = list(DDGS().text(query, max_results=_SEARCH_MAX, region=region,
                                   timelimit=timelimit, backend=backend))
            if raw:
                return raw
            got_empty = True  # engine answered but found nothing — worth a second opinion
            _progress(f"backend '{backend}' returned 0 results — trying next")
            if timelimit:
                break
        except Exception as e:
            last_err = e
            if _is_no_results_error(e):
                got_empty = True
                _progress(f"backend '{backend}' returned 0 results")
                if timelimit:
                    break
                continue
            log.warning(f"[web_search] backend '{backend}' failed: {e}")
            _progress(f"backend '{backend}' failed: {e}")
    if got_empty:
        # timelimit can zero out results (undated pages get filtered) — one
        # unfiltered pass before giving up, so freshness never costs recall.
        if timelimit:
            _progress(f"0 results with timelimit={timelimit} — retrying unfiltered")
            try:
                web_cache.ddg_wait()
                raw = list(DDGS().text(query, max_results=_SEARCH_MAX, region=region,
                                       timelimit=None, backend="auto"))
                if raw:
                    return raw
            except Exception as e:
                if not _is_no_results_error(e):
                    log.warning(f"[web_search] unfiltered retry failed: {e}")
        return "(no results found)"
    if not _ddg_reachable():
        log.warning("[web_search] duckduckgo.com unreachable — network block suspected (check VPN/WARP)")
        return f"[error] web search unavailable (DuckDuckGo unreachable — network block suspected, check VPN/WARP): {last_err}"
    return f"[error] web search unavailable: {last_err}"


def _search_urls(query: str, user_query: str = "", recency: str = "", max_results: int | None = None) -> list[str]:
    """Internal helper: run the same search pipeline as web_search(), but return
    only ranked/selected URLs. No summarization, no cache writes, no web-budget
    increment — callers must enforce their own budget if needed."""
    query = (query or "").strip()
    if not query:
        return []

    effective_uq = (user_query or "").strip() or query
    timelimit = recency.strip().lower() if recency.strip().lower() in _RECENCY_VALUES else None
    if timelimit is None and (_is_time_sensitive(query) or _is_time_sensitive(effective_uq)):
        timelimit = "w"

    # ddgs 9.x no longer handles the old "wt-wt" pseudo-region reliably:
    # it can route the Wikipedia backend to wt.wikipedia.org, which does not
    # exist and causes slow DNS retries. Use a real English region instead.
    region = "th-th" if _THAI_RE.search(query) else "us-en"
    raw = _search_engines(query, region, timelimit)
    if isinstance(raw, str):
        return []

    results = [{"title": r.get("title", ""), "snippet": r.get("body", ""),
                "url": r.get("href", ""), "content": ""} for r in raw]
    results = _filter_adult(results)
    if not results:
        return []

    ranked = _rank_results(results, query)
    take_n = max_results if isinstance(max_results, int) and max_results > 0 else _FETCH_TOP
    selected, _backlog = _select_top(ranked, take_n)
    return [r["url"] for r in selected if r.get("url")]


@tool
def web_search(query: str, user_query: str = "", recency: str = "") -> str:
    """Search the internet for real-time information: current prices, today's news, recent events, facts to look up.
    Returns compact Thai summaries tagged with each source URL — the full bodies are cached.
    Pass user_query with the user's current question to get query-focused summaries.
    Pass recency to restrict result age when the question is time-sensitive: "d" (past day), "w" (past week), "m" (past month), "y" (past year).
    If you need more detail than a summary provides, call recall_web(url) to retrieve up to 20,000 chars of that source.
    If the result is "(no results found)" or "[error]", broaden the same intent once at most; then stop searching and answer with caveats."""
    query = query.strip()
    if not query:
        return "[error] query is required"
    err = _wc_check_and_inc()
    if err:
        return err
    _phase(f"🔍 ค้นหา: {query[:45]}")
    # If the agent forgot user_query, fall back to the search query itself —
    # it's a reasonable proxy for the user's intent.
    effective_uq = (user_query or "").strip() or query

    # Freshness: explicit recency arg wins; otherwise auto-detect time words.
    timelimit = recency.strip().lower() if recency.strip().lower() in _RECENCY_VALUES else None
    if timelimit is None and (_is_time_sensitive(query) or _is_time_sensitive(effective_uq)):
        timelimit = "w"
        _progress("time-sensitive query detected — restricting results to past week")

    # Thai query → Thai region gets Thai-language sources ranked properly.
    # Keep this in sync with _search_urls(): English/global queries need a real
    # region code; "wt-wt" breaks ddgs 9.x Wikipedia fallback.
    region = "th-th" if _THAI_RE.search(query) else "us-en"

    _progress(f"search: {query[:60]} (region={region}, timelimit={timelimit})")
    raw = _search_engines(query, region, timelimit)
    if isinstance(raw, str):
        return raw
    results = [{"title": r.get("title", ""), "snippet": r.get("body", ""),
                "url": r.get("href", ""), "content": ""} for r in raw]
    results = _filter_adult(results)
    if not results:
        return "(no results after content filtering)"
    ranked = _rank_results(results, query)
    selected, backlog = _select_top(ranked, _FETCH_TOP)

    _fetch_batch(selected)
    # Backfill: replace fetch failures with next-ranked URLs that do fetch;
    # slots that still fail keep their snippet fallback as before.
    empties = [r for r in selected if not r["content"]]
    if empties and backlog:
        candidates = backlog[:len(empties)]
        _fetch_batch(candidates)
        replacements = [c for c in candidates if c["content"]]
        for slot, repl in zip(empties, replacements):
            selected[selected.index(slot)] = repl
            _progress(f"backfill: {_domain(repl['url'])} replaces {_domain(slot['url'])} (fetch failed)")

    # Summarization: cache hits resolve immediately; everything else needing
    # an LLM call is batched into a single request when there are ≥2, falling
    # back to per-URL summarize() for singletons or unparseable batches.
    summ_t0 = time.time()
    summaries: dict[str, str] = {}
    pending = []  # [{"url", "raw", "is_new"}]
    for idx, r in enumerate(selected, 1):
        url = r["url"]
        if not url:
            continue
        body = r.get("content") or r.get("snippet") or ""
        if not body:
            continue

        # Check query-aware summary first, then raw cache for re-summarize
        summary = web_cache.get_summary(url, effective_uq)
        if summary is not None:
            _progress(f"[{idx}/{len(selected)}] summary HIT (query-aware): {url[:60]}")
            summaries[url] = summary
            continue

        raw_cached = web_cache.get(url)
        if raw_cached is not None:
            # Raw exists but no summary for this query → re-summarize from cache
            _progress(f"[{idx}/{len(selected)}] raw HIT, re-summarize for new query: {url[:60]}")
            pending.append({"url": url, "raw": raw_cached, "is_new": False})
        else:
            _progress(f"[{idx}/{len(selected)}] queued for summarizing: {url[:60]}")
            raw_with_title = f"# {r.get('title', '')}\n\n{body}"
            pending.append({"url": url, "raw": raw_with_title, "is_new": True})

    # Split pending by whether summarize() would call the LLM at all
    from config import SUMMARY_SKIP_LLM_BELOW
    need_llm = [p for p in pending if len(p["raw"].strip()) > SUMMARY_SKIP_LLM_BELOW]
    skip_llm = [p for p in pending if p not in need_llm]

    for p in skip_llm:
        summary = summarize(p["raw"], url=p["url"], user_query=effective_uq)
        if p["is_new"]:
            web_cache.put(p["url"], p["raw"])
        web_cache.put_summary(p["url"], effective_uq, summary, raw=p["raw"])
        summaries[p["url"]] = summary

    if len(need_llm) >= 2:
        batch_result = summarize_batch([(p["url"], p["raw"]) for p in need_llm], effective_uq)
        if batch_result is not None:
            for p in need_llm:
                summary = batch_result[p["url"]]
                if p["is_new"]:
                    web_cache.put(p["url"], p["raw"])
                web_cache.put_summary(p["url"], effective_uq, summary, raw=p["raw"])
                summaries[p["url"]] = summary
            need_llm = []

    for p in need_llm:
        summary = summarize(p["raw"], url=p["url"], user_query=effective_uq)
        if p["is_new"]:
            web_cache.put(p["url"], p["raw"])
        web_cache.put_summary(p["url"], effective_uq, summary, raw=p["raw"])
        summaries[p["url"]] = summary

    parts = [f"[web:{r['url']}] {summaries[r['url']]}" for r in selected if r["url"] in summaries]
    _progress(f"summarized {len(parts)} URLs in {time.time() - summ_t0:.1f}s")
    log.info(f"[web_search] summarized {len(parts)} URLs in {time.time() - summ_t0:.1f}s")

    if not parts:
        return "(no usable content fetched)"
    return "\n\n".join(parts)
