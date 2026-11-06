# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""web_search.py — multi-query search → cached raw evidence passages

One tool call can run up to four complementary search queries. Each query uses
the existing raced DDGS backend policy, then fetches only its allocated top
sources. Full page bodies are cached for recall_web(); the agent context gets
only deterministic query-relevant passages copied from source text — no LLM
summarizer sits between the web page and the main agent.
"""
from __future__ import annotations
import json
import re
import logging
import socket
import sys
import os
import threading
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from urllib.parse import urlparse
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import web_cache
from tools._summarize import _query_terms, _score_chunk, _split_chunks
from tools._freshness import extract_dates as _extract_dates, url_date as _url_date

from tools._progress import (
    cancelable_as_completed as _cancelable_as_completed,
    check_cancel as _check_cancel,
    progress as _progress,
    phase as _phase,
)
from tools._bounded_executor import BoundedExecutor
from config import (
    WEB_SEARCH_MAX_RESULTS as _SEARCH_MAX,
    WEB_SEARCH_FETCH_TOP as _FETCH_TOP,
    WEB_SEARCH_FETCH_TIMEOUT as _FETCH_TIMEOUT,
    WEB_DEEP_CACHE_MIN_CHARS as _DEEP_CACHE_MIN_CHARS,
    WEB_EVIDENCE_MAX_CHARS as _EVIDENCE_MAX_CHARS,
    WEB_LIVE_CACHE_TTL_SECONDS as _LIVE_CACHE_TTL,
    # Full-body cap shared with browse_url — recall_web promises up to this
    # much from cache, so web_search stores sufficiently deep bodies, not a 1.5k cut.
    BROWSE_URL_MAX_CHARS as _FULL_BODY_CAP,
)

log = logging.getLogger(__name__)
_FETCH_EXECUTOR = BoundedExecutor(_FETCH_TOP, thread_name_prefix="web-fetch")
# trafilatura delegates HTML parsing/freeing to lxml.  On macOS the lxml
# document deallocator is not safe when several fetch workers release parser
# trees concurrently; long multi-search walks can otherwise abort the Python
# process with malloc invalid-free (SIGABRT).  Keep HTTP fetches parallel, but
# serialize only the short extraction/deallocation critical section.
_TRAFILATURA_EXTRACT_LOCK = threading.Lock()
# url -> (published, modified) ISO dates read from the page's own metadata by
# _fetch_one. A side channel keeps _fetch_one's str contract for its other
# callers (browse_url, research_orchestrator) and for tests that patch it.
_PAGE_DATES: dict[str, tuple[str, str]] = {}
_PAGE_DATES_LOCK = threading.Lock()
_PAGE_DATES_MAX = 256

_ADULT_DOMAINS = {"xxx", "porn", "sex", "adult", "hentai", "nude", "xhamster", "xvideos",
                  "pornhub", "betflix", "slot", "casino", "betting", "ufabet", "gclub", "lsm99",
                  "ukdevilz", "gratisreifefrauen", "thaihub", "thaiyesthai"}
_ADULT_KW = {"คลิปหลุด", "หนังโป๊", "nude", "porn", "xxx", "explicit", "18+",
             "สล็อต", "คาสิโน", "เดิมพัน", "บาคาร่า", "slot online", "gambling",
             "หลุดเด็ก", "แตกคาปาก", "เย็ด", "bareback", "shemale"}

# A user asking for an official/primary source is making a provenance
# constraint, not merely a ranking preference.  Search engines can return a
# high-ranked unrelated page (including adult/spam pages) after a freshness
# retry, so keep those results out of the evidence channel entirely.
_OFFICIAL_REQUEST_RE = re.compile(
    r"(?:แหล่งทางการ|เว็บทางการ|แหล่งข้อมูลทางการ|ประกาศทางการ|"
    r"official|primary\s+source|issuer|factsheet|prospectus|fund\s+page|"
    r"ก\.ล\.ต\.|กรมสรรพากร|สำนักงาน\s*ก\.ล\.ต\.)",
    re.IGNORECASE,
)
_TAX_QUERY_RE = re.compile(
    r"(?:rmf|ssf|thai\s*esg|thai\s*esgx|กองทุน|ลดหย่อน|ภาษี|"
    r"ก\.ล\.ต\.|กรมสรรพากร)",
    re.IGNORECASE,
)

# Domains that publish primary regulatory, exchange, index-provider, issuer,
# or fund-manager material.  This is intentionally an allowlist for explicit
# official-source requests; ordinary searches keep the broader ranked result
# behavior.  Subdomains are accepted through _is_official_domain().
_OFFICIAL_DOMAINS = {
    "sec.or.th", "rd.go.th", "ratchakitcha.soc.go.th", "set.or.th", "aimc.or.th",
    "bot.or.th", "oic.or.th",
    "vanguard.com", "vanguard.co.uk", "blackrock.com", "ishares.com", "ssga.com",
    "spglobal.com", "msci.com", "ftserussell.com", "nasdaq.com", "nyse.com",
    "londonstockexchange.com",
    # Common Thai fund-manager/issuer domains.
    "principal.th", "krungsriasset.com", "kasikornasset.com", "scbam.com",
    "ktam.co.th", "bblam.co.th", "mfcfund.com", "tiscoasset.com", "uobam.co.th",
    "one-asset.com", "aberdeenstandard.com", "lhfund.co.th", "kasset.co.th",
}

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
_LIVE_MARKET_RE = re.compile(
    r"(?:ราคาหุ้น|ราคาหลักทรัพย์|หุ้น.{0,24}ราคา|"
    r"\b(?:stock|share|equity|ticker)\b.{0,24}\b(?:price|quote)\b|"
    r"\b(?:price|quote)\b.{0,24}\b(?:stock|share|equity|ticker)\b)",
    re.IGNORECASE,
)
_LIVE_VALUE_RE = re.compile(
    r"(?:\$\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:USD|THB|EUR|GBP|JPY)\b|"
    r"\b\d{1,6}\.\d{2}\b)",
    re.IGNORECASE,
)
_LIVE_CONTEXT_RE = re.compile(
    r"(?:current\s+price|real[- ]?time|rt\s+quote|after\s*hours|at\s+close|"
    r"(?:market|previous|prev|last)\s+close|closed(?:\s+at)?|last\s+trade|"
    r"trading\s+at|shares?\s+(?:are\s+)?valued\s+at|updated|as\s+of|today|"
    r"ปัจจุบัน|ล่าสุด|วันนี้|หลังตลาด|ปิดตลาด|ปิดแล้ว)",
    re.IGNORECASE,
)
_LIVE_NEGATIVE_RE = re.compile(
    r"(?:analyst|target|forecast|prediction|estimate|price\s+target|upside|downside|"
    r"คาดการณ์|ราคาเป้าหมาย|นักวิเคราะห์)",
    re.IGNORECASE,
)
_LIVE_BAD_LABEL_RE = re.compile(
    r"(?:open\s+price|opening\s+price|day\s+high|day\s+low|52[- ]week|"
    r"market\s+cap|volume|range|price\s+target|target|forecast|estimate)",
    re.IGNORECASE,
)
_LIVE_SOURCE_GOOD_RE = re.compile(
    r"(?:real[- ]?time|stock\s+price\s+today|current\s+(?:stock\s+)?price|"
    r"stock\s+quote|live\s+(?:stock\s+)?price|quote\s*&\s*news|market[- ]activity)",
    re.IGNORECASE,
)
_LIVE_PREFIX_REL_RE = re.compile(
    r"(?:current\s+price(?:\s+of\s+[^\n]{0,40})?(?:\s+is|\s*:)?|"
    r"trading\s+at|shares?\s+(?:are\s+)?valued\s+at|after\s*hours|"
    r"at\s+close|(?:market|previous|prev|last)\s+close|last\s+trade|updated|as\s+of|"
    r"ปัจจุบัน|ล่าสุด|หลังตลาด|ปิดตลาด)[^\n]{0,90}$",
    re.IGNORECASE,
)
_LIVE_SUFFIX_REL_RE = re.compile(
    r"^.{0,110}(?:at\s+close|after\s*hours|closed(?:\s+at)?|last\s+trade|updated|"
    r"ปิดแล้ว|หลังตลาด|ปิดตลาด)",
    re.IGNORECASE | re.DOTALL,
)
_LIVE_TIME_RE = re.compile(
    r"(?:\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,\s*20\d{2})?|"
    r"\b\d+\s+(?:minutes?|hours?|days?)\s+ago\b|\btoday\b|\byesterday\b|"
    r"วันนี้|เมื่อวาน|ล่าสุด|\d+\s*(?:นาที|ชั่วโมง|วัน)ที่ผ่านมา)",
    re.IGNORECASE,
)
_MONTH_NUM = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_LIVE_TICKER_EXCLUDE = {"USD", "THB", "EUR", "GBP", "JPY", "NYSE", "NASDAQ", "ETF"}


def _is_time_sensitive(text: str) -> bool:
    t = text.lower()
    return bool(_FRESH_EN_RE.search(t)) or any(kw in t for kw in _FRESH_TH)


def runtime_date_grounding_notice(text: str) -> str:
    """Return a compact calendar anchor for date-sensitive web evidence.

    Thai sources commonly print Buddhist Era years (for example 2569). Local
    models can otherwise reinterpret that number as a Gregorian year after a
    tool round trip. Keep source text verbatim and prepend an explicit runtime
    equivalence instead of rewriting evidence bytes.
    """
    value = text or ""
    be_years = sorted({int(y) for y in re.findall(r"\b(25\d{2})\b", value)})
    if not (
        _is_time_sensitive(value)
        or be_years
        or re.search(r"(?:วันนี้|พรุ่งนี้|เมื่อวาน|today|tomorrow|yesterday)", value, re.IGNORECASE)
    ):
        return ""
    current = date.today()
    buddhist_year = current.year + 543
    mappings = [f"B.E. {year} = {year - 543} CE" for year in be_years[:4]]
    if buddhist_year not in be_years:
        mappings.insert(0, f"B.E. {buddhist_year} = {current.year} CE")
    mapping_text = "; ".join(mappings)
    return (
        f"[runtime_date] TODAY = {current.isoformat()} = Thai Buddhist Era {buddhist_year}. "
        f"Calendar mapping: {mapping_text}. Never interpret a Thai B.E. year as the same-numbered CE year. "
        "Source-relative today/tomorrow/yesterday belongs to the source's publication/forecast date; "
        "do not relabel a different source date as runtime TODAY."
    )


def is_live_market_query(text: str) -> bool:
    """Narrow classifier for facts whose page body can become stale within minutes."""
    return bool(_LIVE_MARKET_RE.search(text or ""))


def _has_live_market_fact(text: str) -> bool:
    """Require a market value and live/session context in the same local window.

    This deliberately rejects analyst targets/forecasts even when the sentence
    compares them with a phrase like "current price". It also avoids treating
    conversational "how close is ..." as a market-close marker.
    """
    value = text or ""
    for match in _LIVE_VALUE_RE.finditer(value):
        lo = max(0, match.start() - 220)
        hi = min(len(value), match.end() + 220)
        window = value[lo:hi]
        prefix = value[max(0, match.start() - 140):match.start()]
        suffix = value[match.end():min(len(value), match.end() + 120)]
        attachment_prefix = value[max(0, match.start() - 70):match.start()]
        attachment_suffix = value[match.end():min(len(value), match.end() + 70)]

        # Bare numbers are accepted only when the local quote block identifies
        # a currency; this keeps percentages/ratios from becoming fake prices.
        token = match.group(0)
        if not re.search(r"(?:\$|USD|THB|EUR|GBP|JPY)", token, re.IGNORECASE):
            if not re.search(r"\b(?:USD|THB|EUR|GBP|JPY)\b", window, re.IGNORECASE):
                continue

        if (_LIVE_NEGATIVE_RE.search(attachment_prefix)
                or _LIVE_NEGATIVE_RE.search(attachment_suffix)
                or _LIVE_BAD_LABEL_RE.search(attachment_prefix)):
            continue

        # Require an explicit relation between this value and the market state,
        # not merely a generic word somewhere in the same evidence chunk.
        if _LIVE_PREFIX_REL_RE.search(prefix) or _LIVE_SUFFIX_REL_RE.search(suffix):
            return True
    return False


def _recent_live_timestamp(text: str, *, today: date | None = None, max_days: int = 7) -> bool:
    """Return True only when evidence carries a timestamp/date recent enough for a quote.

    Seven days intentionally spans weekends and long market holidays. Month/day
    stamps without a year are interpreted as the most recent non-future occurrence.
    """
    value = text or ""
    low = value.lower()
    current = today or date.today()
    if re.search(r"\b(?:today|yesterday)\b", low) or any(x in value for x in ("วันนี้", "เมื่อวาน")):
        return True
    rel = re.search(r"\b(\d+)\s+(minutes?|hours?|days?)\s+ago\b", low)
    if rel:
        amount = int(rel.group(1))
        unit = rel.group(2)
        return unit.startswith(("minute", "hour")) or amount <= max_days
    rel_th = re.search(r"(\d+)\s*(นาที|ชั่วโมง|วัน)ที่ผ่านมา", value)
    if rel_th:
        return rel_th.group(2) != "วัน" or int(rel_th.group(1)) <= max_days

    candidates: list[date] = []
    for y, m, d in re.findall(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", value):
        try:
            candidates.append(date(int(y), int(m), int(d)))
        except ValueError:
            pass
    for m, d, y in re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", value):
        try:
            year = int(y) + (2000 if len(y) == 2 else 0)
            candidates.append(date(year, int(m), int(d)))
        except ValueError:
            pass
    month_pat = re.compile(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{1,2})(?:,\s*(20\d{2}))?\b",
        re.IGNORECASE,
    )
    for mon, day_s, year_s in month_pat.findall(value):
        month = _MONTH_NUM.get(mon.lower())
        if not month:
            continue
        year = int(year_s) if year_s else current.year
        try:
            candidate = date(year, month, int(day_s))
            if not year_s and candidate > current + timedelta(days=1):
                candidate = date(year - 1, month, int(day_s))
            candidates.append(candidate)
        except ValueError:
            pass
    return any(0 <= (current - candidate).days <= max_days for candidate in candidates)


def _has_live_market_quote(text: str) -> bool:
    """A verified quote needs the price relation plus a recent timestamp/date."""
    value = text or ""
    return _has_live_market_fact(value) and _recent_live_timestamp(value)


def _extract_live_ticker(query: str) -> str:
    for token in re.findall(r"\b[A-Z]{1,5}\b", query or ""):
        if token not in _LIVE_TICKER_EXCLUDE:
            return token
    return ""

# brave/google/startpage are 3 scraper engines (ddgs/engines/
# {brave,google,startpage}.py) hitting 3 different origins directly. Brave and
# Google are independent indexes; startpage serves Google's index from its own
# origin, which is what the raced set actually needs here — the observed
# failure mode is per-origin rate limiting (brave.com and google.com both
# answering 429 within the same second), not a bad index. Passed
# together via ddgs's own comma-separated backend syntax, DDGS.text()'s
# internal _search_sync() races them with its own ThreadPoolExecutor. Current
# ddgs 9.14.4/network behavior can still collapse that batch to "No results
# found" while an individual backend returns results, so keep one fixed
# individual fallback rather than duplicating DDGS's threading here.
# duckduckgo itself is
# deliberately NOT in this set: it's the historically rate-limited/flaky
# origin (see network_ddg_blocked finding; the old serial code's own comment
# said "the last pass skips duckduckgo entirely in case it is the
# rate-limited one") — dropping it from the hot path is a reliability
# improvement, not a capability loss, since the raced set already fully
# covered the old code's own last-resort fallback.
#
# 2026-08-23: this set was "bing, brave, google" with a "bing" fallback, but
# ddgs no longer ships a bing engine — every search logged "bing - backends do
# not exist or are disabled" (162 times since 2026-07-28) while silently
# racing 2 engines instead of 3, and the individual fallback resolved to
# ddgs's remaining available set (which readmits duckduckgo) instead of the
# one engine it names. ddgs never raises on an unknown backend name, so the
# drift stayed invisible; _warn_unknown_backends() below now surfaces it.
# yandex was measured as a replacement and rejected: its ddgs engine queries
# yandex.com/search/site/ (Yandex *Site* Search), which answers 200 with
# results unrelated to the query — for "พยากรณ์อากาศ กรมอุตุนิยมวิทยา ฝนตกหนัก"
# it returned YouTube, google.com and an adult site. Result count is not
# evidence of a usable engine; these three were picked on result quality.
#
# The fallback is brave rather than startpage because the two roles need
# different kinds of diversity: the raced set wants separate ORIGINS (so one
# host's rate limit cannot take out the batch), while the fallback runs after
# that batch already collapsed and wants a separate INDEX. startpage would
# retry Google's index through another door; brave is the only independent
# index in the set.
#
# 2026-10-20: re-measured on a healthy link behind the WARP tunnel this host
# needs: brave/google/startpage answered 0/5 (google /sorry bot page, brave
# 429/decode), so every search paid a dead race before `auto`. duckduckgo now
# succeeds 3/3 in ~2s and yahoo (Bing index) 2/3. ddgs runs this list in
# width-2 waves, so duckduckgo+yahoo share the first wave: if WARP is off and
# the ISP blocks duckduckgo it fails fast while yahoo still answers. A/B on the
# same 5 queries: old set 0/5, this set 4/5 at a similar mean latency.
_RACE_BACKENDS = "duckduckgo, yahoo, startpage"
# `auto` is intentionally only the bounded second attempt. It broadens across
# the remaining installed engines after the fast 3-origin race collapses under
# transient rate limits; downstream ranking/evidence policy still decides what
# can be used.
_FALLBACK_BACKEND = "auto"

# One-shot diagnostic; a lost race here only costs a duplicate WARNING line.
_BACKEND_CHECK_DONE = False


def _warn_unknown_backends() -> None:
    """Warn once if a configured backend name no longer exists in ddgs.

    ddgs accepts an unknown name silently — it emits its own library-level
    WARNING and proceeds with whatever is left — so a renamed or removed
    engine degrades the raced set without ever failing a search.
    """
    global _BACKEND_CHECK_DONE
    if _BACKEND_CHECK_DONE:
        return
    _BACKEND_CHECK_DONE = True
    try:
        from ddgs.engines import ENGINES
        available = set(ENGINES["text"])
    except Exception:
        return  # ddgs internals moved — never break search over a diagnostic
    configured = {b.strip() for b in
                  f"{_RACE_BACKENDS},{_FALLBACK_BACKEND}".split(",")
                  if b.strip() and b.strip() not in {"auto", "all"}}
    missing = sorted(configured - available)
    if missing:
        log.warning(
            f"[web_search] configured backend(s) missing from ddgs: {', '.join(missing)} "
            f"— available: {', '.join(sorted(available))}. Searches silently run with "
            f"fewer engines until _RACE_BACKENDS/_FALLBACK_BACKEND are updated."
        )


def _run_search(query: str, region: str, timelimit: str | None,
                 backends: str) -> tuple[list[dict], bool, Exception | None]:
    """One DDGS().text() call against the given (possibly comma-separated)
    backend set. Returns ([], got_empty, last_err) on failure — got_empty
    distinguishes "engine(s) answered, 0 hits" (raised as DDGSException
    "No results found." when every engine in the batch returned nothing,
    per ddgs's _search_sync: `err` only holds a real exception when one was
    actually raised, so a genuine connection/API failure's own message
    reaches here untouched) from a real connection/API failure — same
    distinction the old serial loop made, verified against the installed
    ddgs 9.14.4 source 2026-08-04."""
    from ddgs import DDGS
    try:
        raw = list(DDGS().text(query, max_results=_SEARCH_MAX, region=region,
                                timelimit=timelimit, backend=backends))
    except Exception as e:
        return [], _is_no_results_error(e), e
    if raw:
        return raw, False, None
    return [], True, None


def _is_no_results_error(exc: Exception) -> bool:
    return "No results found" in str(exc)


def _ddg_reachable(host: str = "duckduckgo.com", port: int = 443, timeout: float = 3.0) -> bool:
    """TCP-connect check — the sandboxed-process equivalent of a ping, used
    only when every backend attempt already failed, to tell a network-level
    block (see network_ddg_blocked memory — WARP fixed it last time) apart
    from a rate-limit/API error."""
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


def _is_official_request(query: str, user_query: str = "") -> bool:
    return bool(_OFFICIAL_REQUEST_RE.search(f"{query or ''} {user_query or ''}"))


def _is_official_domain(url: str) -> bool:
    d = _domain(url)
    return any(d == base or d.endswith("." + base) for base in _OFFICIAL_DOMAINS)


def _filter_official(results: list[dict]) -> list[dict]:
    """Keep only primary/issuer domains after an explicit official request."""
    return [r for r in results if _is_official_domain(r.get("url", ""))]


def _normalise_results(raw: list[dict] | str) -> list[dict]:
    if isinstance(raw, str):
        return []
    return [{"title": r.get("title", ""), "snippet": r.get("body", ""),
             "url": r.get("href", ""), "content": "",
             "freshness_relaxed": bool(r.get("_freshness_relaxed", False))}
            for r in raw]


def _official_query_variants(query: str, user_query: str = "") -> list[str]:
    """Return bounded domain-restricted retries for official-only searches."""
    text = f"{query or ''} {user_query or ''}"
    if _TAX_QUERY_RE.search(text):
        # One domain per query is more reliable with DDGS than a parenthesized
        # OR expression.  The first usually gives the SEC handbook landing
        # page; the second covers tax-law material held by the Revenue Dept.
        return [f"{query} site:sec.or.th", f"{query} site:rd.go.th"]
    issuer_sites = []
    for pattern, domains in (
        (r"vanguard", ("vanguard.com", "vanguard.co.uk")),
        (r"(?:ishares|blackrock)", ("ishares.com", "blackrock.com")),
        (r"\bmsci\b", ("msci.com",)),
        (r"(?:ftse|all.world)", ("ftserussell.com",)),
        (r"(?:s&p|sp500|s\s*and\s*p)", ("spglobal.com",)),
    ):
        if re.search(pattern, text, re.IGNORECASE):
            issuer_sites.extend(domains)
    return [f"{query} site:{domain}" for domain in dict.fromkeys(issuer_sites)]


def _official_results_or_retry(results: list[dict], query: str, user_query: str,
                               region: str) -> list[dict]:
    """Apply the official-source boundary, with at most two site retries."""
    accepted = _filter_official(_filter_adult(results))
    if accepted:
        return accepted
    for variant in _official_query_variants(query, user_query):
        raw = _search_engines(variant, region, None)
        retry_results = _filter_official(_filter_adult(_normalise_results(raw)))
        if retry_results:
            _progress(f"official-source retry accepted {len(retry_results)} result(s)")
            return retry_results
    return []


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


# Capitalized query tokens (SET, GDP, S&P). Lowercased term overlap cannot tell
# the Stock Exchange of Thailand from "set" (live 2026-10-19: a Pokémon-card
# "Price Index: Set Market Trend" page ranked first for "SET index").
_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9&]{1,9})(?![A-Za-z0-9])")


def _query_acronyms(query: str) -> list[str]:
    found = [a for a in _ACRONYM_RE.findall(query or "") if not a[1:].isdigit()]
    return list(dict.fromkeys(found))


def _acronym_case(text: str, acronym: str) -> int:
    """+1 written as in the query, -1 present only as an ordinary word, 0 absent.

    Absent is neutral: Thai pages often transliterate (GDP → จีดีพี)."""
    pattern = rf"(?<![A-Za-z0-9]){re.escape(acronym)}(?![A-Za-z0-9])"
    if re.search(pattern, text):
        return 1
    if re.search(pattern, text, re.IGNORECASE):
        return -1
    return 0


def _off_topic_acronym(results: list[dict], query: str) -> tuple[str, str]:
    """(acronym, example title) when every candidate uses a query acronym only
    as an ordinary word; ("", "") otherwise."""
    for acronym in _query_acronyms(query):
        signals = [
            (_acronym_case("\n".join(str(r.get(k) or "") for k in ("title", "snippet", "content")), acronym), r)
            for r in results
        ]
        if signals and not any(s > 0 for s, _ in signals) and any(s < 0 for s, _ in signals):
            example = next(r for s, r in signals if s < 0)
            return acronym, str(example.get("title") or example.get("url") or "")
    return "", ""


# Topics whose answers change over time. Ranking only — a tiebreaker, never a
# filter: undated-but-fresh quote pages must still compete (see _FRESH_EN_RE).
_FRESH_TOPIC_RE = re.compile(
    r"\b(?:news|prices?|quotes?|index|indices|rates?|markets?|stocks?|shares|earnings|"
    r"election|war|ceasefire|forecast|weather|outlook|release[ds]?|updates?)\b"
    r"|ข่าว|ราคา|ดัชนี|อัตรา|ตลาด|หุ้น|ผลประกอบการ|เลือกตั้ง|สงคราม|พยากรณ์|อากาศ",
    re.IGNORECASE,
)


def _recency_bonus(result: dict, target: date | None, today: date) -> float:
    """Tiebreaker from dates visible before fetch (title, snippet, URL).

    ``target`` is the date the query names; without one, nearer to today wins.
    Weights stay below one full query-term hit (3) so topic decides first.
    """
    text = f"{result.get('title', '')} {result.get('snippet', '')}"
    dates = [d for d in _extract_dates(text, today) if d <= today + timedelta(days=1)]
    u = _url_date(result.get("url", ""))
    if u and u <= today + timedelta(days=1):
        dates.append(u)
    if not dates:
        return 0.0
    if target is not None:
        gap = min(abs((target - d).days) for d in dates)
        return 2.5 if gap <= 1 else (1.0 if gap <= 7 else (-2.5 if gap > 365 else 0.0))
    gap = (today - max(dates)).days
    return 2.5 if gap <= 3 else (1.0 if gap <= 30 else (-2.5 if gap > 365 else 0.0))


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
    acronyms = _query_acronyms(query)
    today = date.today()
    named = _extract_dates(query, today)
    target = max(named) if named else None
    fresh = bool(named or _is_time_sensitive(query) or _FRESH_TOPIC_RE.search(query))
    scored = []
    for r in results:
        raw_text = r.get("title", "") + " " + r.get("snippet", "")
        text = raw_text.lower()
        # The title names the page's subject; a title hit counts again.
        kw = _score_chunk(text, terms) + _score_chunk(r.get("title", "").lower(), terms)
        for acronym in acronyms:
            case = _acronym_case(raw_text, acronym)
            kw += 3 if case > 0 else (-8 if case < 0 else 0)
        if is_live_market_query(query):
            if _has_live_market_quote(text):
                kw += 100
            elif _LIVE_SOURCE_GOOD_RE.search(text):
                kw += 24
            if _LIVE_NEGATIVE_RE.search(text):
                kw -= 18
        if fresh:
            kw += _recency_bonus(r, target, today)
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


def _html_page_dates(html: str) -> tuple[str, str]:
    """(published, modified) from page metadata only. extensive_search=False:
    the text-scanning fallback guessed 2010-01-01 for maps.google and today's
    header clock for news hubs in a 40-URL probe. Caller holds the lxml lock."""
    try:
        import htmldate
        published = htmldate.find_date(html, original_date=True, extensive_search=False) or ""
        modified = htmldate.find_date(html, original_date=False, extensive_search=False) or ""
        return published, modified
    except Exception as e:
        log.debug(f"[web_search] page date error: {e}")
        return "", ""


def _take_page_dates(url: str) -> tuple[str, str]:
    with _PAGE_DATES_LOCK:
        return _PAGE_DATES.pop(url, ("", ""))


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
        final_url = str(getattr(resp, "url", "") or url)
        source_err = web_cache.web_source_scope_url_error(final_url)
        if source_err:
            log.info("[web_search] blocked off-scope redirect %s -> %s", url[:80], final_url[:80])
            return source_err
        with _TRAFILATURA_EXTRACT_LOCK:
            text = trafilatura.extract(resp.text, include_comments=False,
                                       include_tables=True, output_format="markdown") or ""
            dates = _html_page_dates(resp.text) if text else ("", "")
        if any(dates):
            with _PAGE_DATES_LOCK:
                _PAGE_DATES[url] = dates
                while len(_PAGE_DATES) > _PAGE_DATES_MAX:
                    _PAGE_DATES.pop(next(iter(_PAGE_DATES)))
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
        futs = {}
        for result in rs:
            future = _FETCH_EXECUTOR.submit(_fetch_one, result["url"])
            if future is not None:
                futs[future] = result
        for future in as_completed(futs, timeout=12):
            try:
                result = futs[future]
                raw = future.result()
                if raw.startswith("[source_blocked]"):
                    result["source_error"] = raw
                    result["content"] = ""
                else:
                    result["content"] = raw
                    result["page_dates"] = _take_page_dates(result["url"])
                    result["official_verified"] = bool(
                        raw and web_cache.web_source_scope_official_only()
                    )
            except Exception:
                pass
    except Exception:
        pass  # timeout or executor error — use snippet fallback in summarization


def _search_engines(query: str, region: str, timelimit: str | None) -> list[dict] | str:
    """Run the raced backend set, then at most one individual fallback. Returns raw result
    list, or an "[error] ..." / "(no results found)" string when the
    attempt fails.

    2026-08-04: was 3 SERIAL attempts (auto, sleep 3s, auto, sleep 3s,
    3 named engines), each waiting out a full ~14-20s round trip before
    trying the next — a failure on attempt 1 cost up to 3 round trips plus
    6s of sleep. Collapsed to one call: 'auto' (which can route to the
    flaky duckduckgo backend, plus wikipedia/grokipedia that ddgs's
    _get_engines() prepends for the text category) is replaced by the named
    _RACE_BACKENDS set, passed as ddgs's own comma-separated syntax so
    its internal _search_sync() races them. A bounded individual fallback
    covers current DDGS batches that discard a working engine's results — see
    _run_search / _RACE_BACKENDS."""
    try:
        import ddgs  # noqa: F401  — Tool Output Contract: never raise out of the tool
    except Exception as e:
        log.warning(f"[web_search] ddgs unavailable: {e}")
        return f"[error] web search unavailable: {e}"
    _warn_unknown_backends()  # ddgs is importable here — safe to read its registry
    web_cache.ddg_wait()  # adaptive min-gap delay — paces our own outbound search volume
    raw, got_empty, last_err = _run_search(query, region, timelimit, _RACE_BACKENDS)
    if raw:
        return raw

    # Exactly one fallback keeps latency bounded at two DDGS calls. If a
    # freshness-filtered query produced no hits, the fallback may broaden to
    # unfiltered recall, but every returned row is marked so downstream output
    # cannot silently pretend the freshness guarantee survived.
    freshness_relaxed = bool(got_empty and timelimit)
    fallback_timelimit = None if freshness_relaxed else timelimit
    if freshness_relaxed:
        _progress(f"0 results with timelimit={timelimit} — "
                  f"retrying unfiltered via {_FALLBACK_BACKEND} (freshness relaxed)")
    elif got_empty:
        _progress(f"combined backends returned 0 results — retrying via {_FALLBACK_BACKEND}")
    else:
        _progress(f"combined backends failed — retrying via {_FALLBACK_BACKEND}")
    web_cache.ddg_wait()
    raw2, fallback_empty, fallback_err = _run_search(
        query, region, fallback_timelimit, _FALLBACK_BACKEND
    )
    if raw2:
        if freshness_relaxed:
            return [{**row, "_freshness_relaxed": True} for row in raw2]
        return raw2
    if fallback_empty:
        if timelimit:
            return "[error] fresh web search unavailable — filtered and fallback providers returned no usable results; freshness could not be verified"
        return "(no results found)"
    last_err = fallback_err or last_err
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

    results = _normalise_results(raw)
    results = _filter_adult(results)
    if (web_cache.web_source_scope_official_only()
            or _is_official_request(query, effective_uq)):
        results = _official_results_or_retry(results, query, effective_uq, region)
    if not results:
        return []

    ranked = _rank_results(results, query)
    take_n = max_results if isinstance(max_results, int) and max_results > 0 else _FETCH_TOP
    selected, _backlog = _select_top(ranked, take_n)
    return [r["url"] for r in selected if r.get("url")]


_MAX_BATCH_QUERIES = 4
_EVIDENCE_CHUNK_CHARS = 550


def _split_bracketed_queries(inner: str) -> list[str]:
    """Split a non-JSON list body on commas outside double quotes; keep quotes."""
    parts, buf, quoted = [], [], False
    for ch in inner:
        if ch == '"':
            quoted = not quoted
        if ch == "," and not quoted:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return [part.strip() for part in parts if part.strip()]


def _normalise_query_batch(query: str | list[str]) -> tuple[list[str], bool]:
    """Normalize one string or a model-friendly list into up to four searches."""
    if isinstance(query, str):
        stripped = query.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                decoded = json.loads(stripped)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, list):
                items = [part.strip() for part in decoded if isinstance(part, str) and part.strip()]
            elif decoded is None:
                # Qwen writes phrase operators inside a list it cannot JSON-escape:
                # ["a" site:x, "b" site:x]. Searching that whole string as one
                # bracketed query silently returns nothing and burns the budget.
                items = _split_bracketed_queries(stripped[1:-1])
            else:
                items = [part.strip() for part in query.split("|||") if part.strip()]
        else:
            items = [part.strip() for part in query.split("|||") if part.strip()]
    elif isinstance(query, list):
        items = [part.strip() for part in query if isinstance(part, str) and part.strip()]
    else:
        return [], False
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[:_MAX_BATCH_QUERIES], len(out) > _MAX_BATCH_QUERIES


def select_web_evidence(
    raw: str,
    focus_query: str,
    cap: int = _EVIDENCE_MAX_CHARS,
    *,
    fallback: str = "leading",
) -> str:
    """Select compact verbatim source passages without calling an LLM.

    ``leading`` preserves web_search's conservative no-match behavior. ``coverage``
    keeps evenly distributed passages across a known page, which is safer for
    browse_url when the query and page use different languages or the user asks
    for a general article summary with little lexical overlap.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    if len(text) <= cap:
        return text

    # _split_chunks preserves paragraph order but deliberately does not split a
    # single very long paragraph. Split those oversized chunks into deterministic
    # windows so a relevant fact deep in one paragraph can still be selected.
    chunks: list[str] = []
    for chunk in _split_chunks(text, size=_EVIDENCE_CHUNK_CHARS):
        if len(chunk) <= _EVIDENCE_CHUNK_CHARS:
            chunks.append(chunk)
        else:
            chunks.extend(
                chunk[i:i + _EVIDENCE_CHUNK_CHARS]
                for i in range(0, len(chunk), _EVIDENCE_CHUNK_CHARS)
            )
    if not chunks:
        return text[:cap]

    terms = _query_terms((focus_query or "").strip())
    scores = [_score_chunk(chunk.lower(), terms) for chunk in chunks] if terms else [0] * len(chunks)
    if is_live_market_query(focus_query):
        # Exact current-quote passages must outrank analyst targets/history that
        # repeat the words "price" and the ticker more often. This remains fully
        # deterministic and only changes ordering inside the same evidence cap.
        for idx, chunk in enumerate(chunks):
            if _has_live_market_fact(chunk):
                scores[idx] += 100
            elif _LIVE_CONTEXT_RE.search(chunk):
                scores[idx] += 12

    # Always keep the lede for title/date/entity context. Then add the strongest
    # matching passage plus its immediate neighbors. This keeps evidence compact
    # while preserving qualifiers that often sit just before/after the hit.
    selected: set[int] = {0}
    used = min(len(chunks[0]), cap)
    if any(scores):
        for idx in sorted(range(len(chunks)), key=lambda i: (-scores[i], i)):
            if scores[idx] <= 0 or used >= cap:
                break
            for neighbor in (idx, idx - 1, idx + 1):
                if neighbor < 0 or neighbor >= len(chunks) or neighbor in selected:
                    continue
                need = len(chunks[neighbor]) + 8
                if used + need <= cap:
                    selected.add(neighbor)
                    used += need
    else:
        if fallback == "coverage" and len(chunks) > 1:
            # Direct URL reading must remain useful even when lexical scoring is
            # weak (e.g. Thai question, English page, or "summarize this"). Fill
            # the same compact budget with deterministic document-wide coverage:
            # lede + evenly spaced interior/end passages, still in source order.
            target_count = max(2, cap // (_EVIDENCE_CHUNK_CHARS + 8))
            target_count = min(target_count, len(chunks))
            if target_count == 1:
                targets = [0]
            else:
                targets = [
                    round(i * (len(chunks) - 1) / (target_count - 1))
                    for i in range(target_count)
                ]
            for idx in targets[1:]:
                if idx in selected:
                    continue
                need = len(chunks[idx]) + 8
                if used + need <= cap:
                    selected.add(idx)
                    used += need
        else:
            # Search results have their own ranking/snippet context, so a
            # conservative leading fallback is safer than arbitrary coverage.
            for idx in range(1, len(chunks)):
                need = len(chunks[idx]) + 8
                if used + need > cap:
                    break
                selected.add(idx)
                used += need

    parts: list[str] = []
    previous: int | None = None
    for idx in sorted(selected):
        if previous is not None and idx != previous + 1:
            parts.append("[...]")
        parts.append(chunks[idx])
        previous = idx
    return "\n".join(parts)[:cap]


_TABLE_SEP_RE = re.compile(r"^\|(?:\s*:?-{2,}:?\s*\|)+$")
_HAS_LETTER_RE = re.compile(r"[^\W\d_]")
_ROW_JOIN_MAX_LINES = 12
# Only long prose repeats are dropped. Short labels, lines with digits
# (timestamps, prices) and code repeat per item and carry that item's facts;
# a repeated headline followed by its own timestamp line is kept so the
# timestamp is not re-attached to the previous item.
_DEDUPE_MIN_CHARS = 40
_NOT_PROSE_RE = re.compile(r"\d|^[\"'{}\[\]<>#`$-]")


def _dedupe_row_cells(row: str) -> str:
    """Drop the repeats a multi-line cell leaves: "| AAV | AAV Asia Aviation |
    Asia Aviation | 0.82 |" -> "| AAV | Asia Aviation | 0.82 |". Text cells
    only, so equal adjacent numbers (open = high) are kept."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    kept: list[str] = []
    for cell in cells:
        prev = kept[-1] if kept else ""
        if prev and cell.startswith(prev + " ") and _HAS_LETTER_RE.search(prev):
            cell = cell[len(prev):].strip()
        if prev and cell == prev and _HAS_LETTER_RE.search(cell):
            continue
        kept.append(cell)
    return "| " + " | ".join(kept) + " |"


def _tidy_page_text(text: str) -> str:
    """Make an extracted page cheaper to read before evidence selection.

    trafilatura renders a table cell that holds several block elements as a
    row broken over 4-5 lines (live: aslan.ai SET Index table), so each row
    costs several lines of the evidence budget and its values no longer sit
    beside their name. Rejoin such rows when the joined pipe count matches
    the header (else leave them untouched), and drop repeated prose lines.
    """
    lines = text.split("\n")
    out: list[str] = []
    seen: set[str] = set()
    width = 0
    in_code = False
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("```"):
            in_code = not in_code
        if in_code or s.startswith("```"):
            out.append(lines[i].rstrip())
            i += 1
            continue
        if _TABLE_SEP_RE.match(s) and out and out[-1].startswith("|"):
            width = s.count("|")
            out.append(s)
            i += 1
            continue
        if width and s.startswith("|") and s.count("|") < width:
            buf, count, j = [s], s.count("|"), i + 1
            while j < len(lines) and count < width and j - i < _ROW_JOIN_MAX_LINES:
                nxt = lines[j].strip()
                # The last row often omits the trailing empty cell: stop
                # before the next row, separator, or end of table.
                if count == width - 1 and (not nxt or nxt.startswith("|") and nxt.count("|") >= 2):
                    break
                if not nxt or _TABLE_SEP_RE.match(nxt):
                    break
                buf.append(nxt)
                count += nxt.count("|")
                j += 1
            if len(buf) > 1 and width - 1 <= count <= width:
                out.append(_dedupe_row_cells(" ".join(buf)))
                i = j
                continue
        if not s.startswith("|"):
            width = 0
            if len(s) >= _DEDUPE_MIN_CHARS and not _NOT_PROSE_RE.search(s):
                after = [l.strip() for l in lines[i + 1:i + 6] if l.strip()][:3]
                if s in seen and not any(re.search(r"\d", a) and len(a) < 80 for a in after):
                    i += 1
                    continue
                seen.add(s)
        if s or (out and out[-1]):
            out.append(s)
        i += 1
    return "\n".join(out).strip()


def _hydrate_results(results: list[dict], *, max_cache_age_seconds: float | None = None) -> None:
    """Load cached raw bodies, fetch misses, and cache only deep-enough pages.

    Thin ordinary direct extracts are still valid evidence for this search call,
    but are intentionally not admitted to the shared raw cache. That leaves a
    later recall_web/browse_url free to perform its deeper direct+Jina fetch.
    Official-only direct bodies are cached regardless of length because Jina is
    deliberately disabled there and the verified direct response is the safest
    available provenance-preserving evidence.
    """
    misses: list[dict] = []
    for result in results:
        url = result.get("url", "")
        if not url:
            continue
        cached = web_cache.get(url, max_age_seconds=max_cache_age_seconds)
        if cached is not None:
            result["content"] = cached
            result["page_dates"] = web_cache.get_page_dates(url)
            result["cache_hit"] = True
            _progress(f"raw cache HIT: {url[:60]}")
        else:
            misses.append(result)

    _fetch_batch(misses)
    for result in misses:
        body = result.get("content") or ""
        if not body:
            continue
        official_verified = bool(result.get("official_verified", False))
        if official_verified or len(body) >= _DEEP_CACHE_MIN_CHARS:
            web_cache.put(
                result.get("url", ""),
                body,
                official_verified=official_verified,
                page_dates=result.get("page_dates"),
            )
        else:
            _progress(
                f"thin direct body ({len(body)} chars) — evidence only, not raw-cached: "
                f"{result.get('url', '')[:60]}"
            )


def _result_evidence(result: dict, focus: str, *, live_market: bool = False) -> tuple[str, str]:
    """Return (evidence, label) for one search result.

    A fetched body normally wins over a search snippet. For live-market queries,
    however, a readable page can omit the live quote while the index snippet has
    it; in that narrow case prefer the quote-bearing snippet instead of silently
    losing the requested fact.
    """
    body = _tidy_page_text((result.get("content") or "").strip())
    snippet = (result.get("snippet") or "").strip()
    if body:
        evidence = select_web_evidence(body, focus)
        if live_market and not _has_live_market_quote(evidence) and snippet:
            snippet_evidence = select_web_evidence(snippet, focus)
            if _has_live_market_quote(snippet_evidence):
                return snippet_evidence, "Search snippet fallback (live fact)"
        return evidence, "Evidence (verbatim page passages)"
    if snippet:
        return select_web_evidence(snippet, focus), "Search snippet fallback"
    return "", ""


def _age_text(value: str, today: date) -> str:
    days = (today - date.fromisoformat(value)).days
    if days <= 0:
        return "today"
    if days < 730:
        return f"{days} day{'s' if days != 1 else ''} old"
    return f"{days // 365} years old"


def _page_date_line(result: dict, today: date | None = None) -> str:
    """One line stating the page's own dates and their age.

    Ages are computed at render time (cached bodies get older). Runtime TODAY
    is never printed in ISO here: the Goal verifier reads every ISO date in a
    source block as evidence, so it would make stale sources look current.
    """
    today = today or date.today()
    url = result.get("url", "")
    u = _url_date(url)
    kept: list[str] = []
    for value in result.get("page_dates") or ("", ""):
        try:
            d = date.fromisoformat(value) if value else None
        except ValueError:
            d = None
        # htmldate falls back to YYYY-01-01 when it only finds a year.
        if d is None or d > today + timedelta(days=1) or (
                (d.month, d.day) == (1, 1) and d != u):
            kept.append("")
        else:
            kept.append(d.isoformat())
    published, modified = kept
    if not published and u and u <= today + timedelta(days=1):
        published = u.isoformat()
    if not published and not modified:
        return "Page date: unknown — rely on dates inside the text"
    path = urlparse(url).path.strip("/")
    # Root or a bare section word (/economy/); one-segment article slugs carry
    # hyphens or digits.
    is_hub = not path or (len(path) <= 24 and not re.search(r"[/\-_\d.]", path))
    hub = "; home/section page — items carry their own dates" if is_hub else ""
    if published and modified and modified > published:
        return (f"Page date: published {published} ({_age_text(published, today)}), "
                f"updated {modified} ({_age_text(modified, today)}){hub}")
    value = published or modified
    return f"Page date: {value} ({_age_text(value, today)}){hub}"


def _web_search_one(
    query: str,
    user_query: str,
    recency: str,
    take_n: int,
    *,
    _source_urls: list[str] | None = None,
) -> str:
    """One reserved query: search, fetch, and return deterministic raw evidence."""
    effective_uq = (user_query or "").strip() or query
    focus = f"{effective_uq}\n{query}".strip()
    live_market = is_live_market_query(focus)
    cache_max_age = _LIVE_CACHE_TTL if live_market else None
    official_only = (
        web_cache.web_source_scope_official_only()
        or _is_official_request(query, effective_uq)
    )

    # Exact ticker quote is a data-retrieval problem, not a web-ranking problem.
    # Try one canonical quote URL first; only fall back to DDGS when it cannot
    # provide a recent price+timestamp. This is faster, avoids provider rate
    # limits, and prevents conflicting search sources from confusing synthesis.
    if live_market and not official_only:
        ticker = _extract_live_ticker(query)
        if ticker:
            direct_url = f"https://finance.yahoo.com/quote/{ticker}/"
            direct = {
                "url": direct_url,
                "title": f"{ticker} Stock Price, News, Quote & History - Yahoo Finance",
                "snippet": "",
                "content": "",
                "freshness_relaxed": False,
            }
            _hydrate_results([direct], max_cache_age_seconds=cache_max_age)
            evidence, evidence_label = _result_evidence(direct, focus, live_market=True)
            if _has_live_market_quote(evidence):
                _progress(f"live-quote direct verified: {ticker} via finance.yahoo.com")
                if _source_urls is not None:
                    _source_urls.append(direct_url)
                return (
                    "[live_quote_verified]\n"
                    "[market_status_unverified] Report only the source price/date/session labels below; "
                    "do not infer the current market open/closed state, holiday/weekend reason, next-open time, or weekday name.\n"
                    f"[web:{direct_url}]\nTitle: {direct['title']}\n"
                    f"{evidence_label}:\n{evidence}"
                )

    timelimit = recency.strip().lower() if recency.strip().lower() in _RECENCY_VALUES else None
    if timelimit is None and (_is_time_sensitive(query) or _is_time_sensitive(effective_uq)):
        timelimit = "w"
        _progress(f"{query[:36]}: time-sensitive → past week")

    region = "th-th" if _THAI_RE.search(query) else "us-en"
    _progress(f"search: {query[:60]} (region={region}, timelimit={timelimit})")
    raw = _search_engines(query, region, timelimit)
    results = _normalise_results(raw)
    freshness_relaxed = any(r.get("freshness_relaxed") for r in results)

    if official_only:
        results = _official_results_or_retry(results, query, effective_uq, region)
        if not results:
            return "[no_official_sources] ไม่พบแหล่งทางการที่เกี่ยวข้อง — ห้ามใช้ผลลัพธ์ที่ไม่ใช่ official เป็นหลักฐาน"
    else:
        results = _filter_adult(results)
        if isinstance(raw, str):
            return raw
    if not results:
        return "(no results after content filtering)"

    ranked = _rank_results(results, query)
    selected, backlog = _select_top(ranked, max(1, take_n))
    _hydrate_results(selected, max_cache_age_seconds=cache_max_age)

    # Replace failed fetches with next-ranked pages when possible. If all fetches
    # fail, the original selected row still contributes its search snippet below.
    empties = [r for r in selected if not r.get("content") and not r.get("source_error")]
    if empties and backlog:
        candidates = backlog[:len(empties)]
        _hydrate_results(candidates, max_cache_age_seconds=cache_max_age)
        replacements = [c for c in candidates if c.get("content")]
        for slot, repl in zip(empties, replacements):
            selected[selected.index(slot)] = repl
            _progress(
                f"backfill: {_domain(repl['url'])} replaces "
                f"{_domain(slot['url'])} (fetch failed)"
            )

    # Four-query batches intentionally begin with one source per query. For a
    # live market fact, deepen only the weak query when its first source does
    # not expose a quote/session marker. This preserves the one-tool-call budget
    # while avoiding a second model search turn just to recover one missing price.
    if live_market and not any(
        _has_live_market_quote(_result_evidence(r, focus, live_market=True)[0])
        for r in selected
    ):
        snippet_hits = [r for r in backlog if _has_live_market_quote(r.get("snippet", ""))]
        candidates = snippet_hits[:3] if snippet_hits else backlog[:3]
        for candidate in candidates:
            _hydrate_results([candidate], max_cache_age_seconds=cache_max_age)
            candidate_evidence, _ = _result_evidence(candidate, focus, live_market=True)
            if _has_live_market_quote(candidate_evidence):
                if selected:
                    replaced = selected[-1]
                    selected[-1] = candidate
                    _progress(
                        f"live-fact backfill: {_domain(candidate.get('url', ''))} replaces "
                        f"{_domain(replaced.get('url', ''))}"
                    )
                else:
                    selected.append(candidate)
                break

    # Search-engine ranking/rate limits must not be the last word for a common
    # US ticker. If the bounded search/backfill still has no verified quote,
    # fetch the canonical Yahoo quote URL directly once. This is still inside
    # the same web_search invocation/budget unit and avoids another model turn.
    if live_market and not official_only and not any(
        _has_live_market_quote(_result_evidence(r, focus, live_market=True)[0])
        for r in selected
    ):
        ticker = _extract_live_ticker(query)
        if ticker:
            direct_url = f"https://finance.yahoo.com/quote/{ticker}/"
            if not any(r.get("url") == direct_url for r in selected):
                direct = {
                    "url": direct_url,
                    "title": f"{ticker} Stock Price, News, Quote & History - Yahoo Finance",
                    "snippet": "",
                    "content": "",
                    "freshness_relaxed": False,
                }
                _hydrate_results([direct], max_cache_age_seconds=cache_max_age)
                direct_evidence, _ = _result_evidence(direct, focus, live_market=True)
                if _has_live_market_quote(direct_evidence):
                    if selected:
                        replaced = selected[-1]
                        selected[-1] = direct
                        _progress(
                            f"live-quote direct fallback: finance.yahoo.com replaces "
                            f"{_domain(replaced.get('url', ''))}"
                        )
                    else:
                        selected.append(direct)

    off_topic, example_title = _off_topic_acronym(selected + backlog, query)
    if off_topic:
        _progress(f"low relevance: no result writes {off_topic!r} as in the query")
        return (
            f'[low_relevance] No result uses "{off_topic}" as written in the query; they use it as an '
            f'ordinary word (e.g. "{example_title[:90]}"), so they are unrelated and not evidence. '
            "Search again with context that disambiguates it (full name, country, or market)."
        )

    parts: list[str] = []
    included_source_urls: list[str] = []
    live_fact_verified = False
    for result in selected:
        url = result.get("url", "")
        if not url or result.get("source_error"):
            continue
        title = (result.get("title") or "").strip()
        evidence, evidence_label = _result_evidence(result, focus, live_market=live_market)
        if not evidence:
            continue
        if live_market and _has_live_market_quote(evidence):
            live_fact_verified = True

        title_line = f"\nTitle: {title}" if title else ""
        parts.append(
            f"[web:{url}]{title_line}\n{_page_date_line(result)}\n{evidence_label}:\n{evidence}"
        )
        included_source_urls.append(url)

    if not parts:
        if any(r.get("source_error") for r in selected):
            return "[source_blocked] official-only turn — selected source redirected outside the official/issuer allowlist"
        return "(no usable content fetched)"

    notices: list[str] = []
    if freshness_relaxed:
        notices.append(
            "[freshness_relaxed] Freshness-filtered search returned no results; "
            "the sources below came from an unfiltered fallback. Verify the source "
            "timestamp/session before treating changing values as current."
        )
    if live_market and not live_fact_verified:
        notices.append(
            "[live_fact_unverified] No selected source exposed both a market value and "
            "a current/session marker. Exact current quote is unavailable; do not infer "
            "it from open/high/low/range/forecast/target fields."
        )
        _progress(f"evidence ready: 0 verified live source(s) for {query[:40]}")
        # Fail closed: do not expose numeric near-misses that the outer model can
        # accidentally relabel as the current quote (observed with OPEN PRICE).
        return "\n".join(notices)
    if live_market:
        notices.append(
            "[live_quote_verified] [market_status_unverified] Report only the source price/date/session labels; "
            "do not infer current market open/closed state, holiday/weekend reason, next-open time, or weekday name."
        )
    _progress(f"evidence ready: {len(parts)} source(s) for {query[:40]}")
    payload = "\n\n".join(parts)
    if _source_urls is not None:
        _source_urls.extend(included_source_urls)
    return "\n".join(notices + [payload]) if notices else payload


def _web_search_one_with_sources(
    query: str,
    user_query: str,
    recency: str,
    take_n: int,
) -> tuple[str, list[str]]:
    source_urls: list[str] = []
    content = _web_search_one(
        query,
        user_query,
        recency,
        take_n,
        _source_urls=source_urls,
    )
    return content, list(dict.fromkeys(source_urls))


def _web_search_tool_artifact(source_urls: list[str]) -> dict[str, object]:
    """Return producer-known result URLs outside the model-visible content."""
    return {
        "provenance_type": "web_search_results_v1",
        "external_source_urls": list(dict.fromkeys(source_urls)),
    }


@tool(response_format="content_and_artifact")
def web_search(
    query: str | list[str],
    user_query: str = "",
    recency: str = "",
) -> tuple[str, dict[str, object]]:
    """Search the web and return compact verbatim evidence + source URLs. query may be one search or a list of 2-4 focused subqueries run concurrently; prefer one batch for distinct facts/angles. user_query focuses deterministic passage selection. recency: d/w/m/y. Date-sensitive queries use runtime [Today], not a guessed year.

    Full fetched pages are cached; no inner LLM summarizes them. Thin passage => use recall_web(url, query=...) for more raw evidence. Budget is per invocation, so a 1-4 query batch costs one call; max 3 web_search calls/turn."""

    batch, query_limit_hit = _normalise_query_batch(query)
    if not batch:
        return "[error] query is required", _web_search_tool_artifact([])

    _check_cancel()
    err = web_cache.web_count_check_and_inc()
    if err:
        return err, _web_search_tool_artifact([])

    active = batch
    _phase(
        f"🔍 ค้นหา {len(active)} คำค้น"
        if len(active) > 1
        else f"🔍 ค้นหา: {active[0][:45]}"
    )

    # Preserve the old total evidence breadth: one query may inspect four
    # sources; a 2-4 query batch divides those four source slots across queries.
    base, remainder = divmod(_FETCH_TOP, len(active))
    source_slots = [
        max(1, base + (1 if idx < remainder else 0))
        for idx in range(len(active))
    ]

    outputs: list[str] = [""] * len(active)
    source_urls: list[str] = []
    if len(active) == 1:
        outputs[0], source_urls = _web_search_one_with_sources(
            active[0], user_query, recency, source_slots[0]
        )
    else:
        # DDGS starts are still globally paced by web_cache.ddg_wait(); the
        # queries can overlap after those staggered starts, while page fetches
        # share the existing bounded fetch executor.
        pool = ThreadPoolExecutor(
            max_workers=len(active),
            thread_name_prefix="web-query",
        )
        try:
            futures = {}
            for idx, q in enumerate(active):
                ctx = copy_context()
                future = pool.submit(
                    ctx.run,
                    _web_search_one_with_sources,
                    q,
                    user_query,
                    recency,
                    source_slots[idx],
                )
                futures[future] = idx
            for future in _cancelable_as_completed(futures):
                idx = futures[future]
                try:
                    outputs[idx], query_sources = future.result()
                    source_urls.extend(query_sources)
                except Exception as exc:
                    log.warning("[web_search] batch query failed: %s", exc)
                    outputs[idx] = f"[error] web search unavailable: {exc}"
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    notes: list[str] = []
    control_sentinel = (
        len(active) == 1
        and outputs[0].startswith((
            "[error]", "[web_limit]", "[no_official_sources]", "[source_blocked]",
        ))
    )
    date_notice = "" if control_sentinel else runtime_date_grounding_notice(
        "\n".join([str(user_query or "")] + active + outputs)
    )
    if date_notice:
        notes.append(date_notice)
    if query_limit_hit:
        notes.append(
            f"[query_limit] web_search ใช้สูงสุด {_MAX_BATCH_QUERIES} คำค้นต่อครั้ง — ใช้ 4 คำค้นแรก"
        )
    if len(active) == 1:
        content = "\n".join(notes + [outputs[0]]) if notes else outputs[0]
        return content, _web_search_tool_artifact(source_urls)

    blocks = [
        f"[query:{q}]\n{outputs[idx]}"
        for idx, q in enumerate(active)
    ]
    return "\n\n".join(notes + blocks), _web_search_tool_artifact(source_urls)
