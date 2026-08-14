# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""fetch_sitemap.py — ดึง URL list จาก sitemap.xml ของเว็บไซต์

รองรับ:
  - <sitemapindex> (ตรวจจาก root element ไม่ใช่ substring ทั้ง doc; fetch สูงสุด 3 child —
    เลือก child ที่ match filter keyword ก่อน (เรียง lastmod ใหม่→เก่าภายในกลุ่ม)
    แล้วตาม <lastmod> ใหม่→เก่า แล้วตาม original order)
  - <urlset>       (direct URL list พร้อม <lastmod> ถ้ามี — parse แบบ linear str.find scan
    กัน quadratic regex บน XML ที่ truncated/malformed; fallback เป็น <loc>-only ถ้าไม่มี wrapper tag;
    unescape XML entities เช่น &amp; ใน loc)
  - robots.txt fallback เพื่อหา Sitemap: location (ลองทุกบรรทัด Sitemap: จนกว่าจะเจอ)
  - gzip-compressed sitemap (.xml.gz / Content-Encoding: gzip)
  - web_cache (raw XML ต่อ URL; ข้ามการ cache ถ้า XML ใหญ่เกิน WEB_CACHE_PER_ENTRY_MAX —
    entry ที่โดน truncate จะ parse ขาด) + web budget (นับ 1 slot ต่อ 1 tool call ไม่ว่าจะ fetch กี่ URL)
  - multi-keyword filter คั่นด้วย space/comma/+ (AND ก่อน, fallback เป็น OR ถ้า AND ไม่มี match)
  - ผลลัพธ์เรียงตาม lastmod ใหม่→เก่า (เฉพาะ lastmod รูปแบบ ISO YYYY-MM-DD — รูปแบบอื่นถือว่า
    ไม่มีวันที่), จำกัดจำนวนที่แสดงต่อ call ด้วย FETCH_SITEMAP_SHOWN
"""
from __future__ import annotations
import gzip
import html
import logging
import re
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from tools._progress import phase as _phase
from tools import web_cache
from tools.web_cache import web_count_check_and_inc as _wc_check_and_inc
from config import FETCH_SITEMAP_MAX_URLS, FETCH_SITEMAP_SHOWN, WEB_CACHE_PER_ENTRY_MAX

log = logging.getLogger(__name__)

_GZIP_MAGIC = b"\x1f\x8b"


def _curl_bytes(url: str) -> bytes:
    """Raw fetch as bytes — needed for gzip magic-byte detection.
    --compressed lets curl auto-decode Content-Encoding: gzip when the
    server sets the header; the magic-byte check below covers .xml.gz
    bodies served without that header."""
    try:
        r = subprocess.run(
            ["curl", "-s", "-L", "--compressed", "--max-time", "20", url],
            capture_output=True, text=False, timeout=25,
        )
        if r.returncode != 0:
            log.warning(f"[fetch_sitemap] curl rc={r.returncode} for {url}: {r.stderr.decode(errors='replace')[:200]}")
        return r.stdout or b""
    except FileNotFoundError:
        log.error("[fetch_sitemap] curl binary not found — cannot fetch sitemap")
        return b""
    except Exception as e:
        log.warning(f"[fetch_sitemap] curl error for {url}: {e}")
        return b""


def _decode_body(raw: bytes) -> str:
    """gzip-aware decode: magic bytes → gunzip ก่อน decode; ล้มเหลว → treat as empty"""
    if not raw:
        return ""
    if raw[:2] == _GZIP_MAGIC:
        try:
            raw = gzip.decompress(raw)
        except Exception as e:
            log.warning(f"[fetch_sitemap] gzip decompress failed: {e}")
            return ""
    return raw.decode("utf-8", errors="replace")


def _cached_fetch(url: str, state: dict) -> str:
    """Cache-aware fetch. Cache hit → free (no budget). Cache miss → check the
    per-turn web budget EXACTLY ONCE per tool call (state["budget_checked"]);
    every subsequent miss within the same call (children, robots.txt) reuses
    the already-reserved slot. `state` is a fresh dict created per tool call
    (never module-level) so this is safe across concurrent tool invocations.

    XML ที่ใหญ่เกิน WEB_CACHE_PER_ENTRY_MAX จะไม่ cache — web_cache.put() จะ
    truncate แล้ว XML ขาด ทำให้ call ถัดไป parse ได้ URL ไม่ครบแบบเงียบๆ;
    hit ที่ลงท้าย marker truncate (poisoned ก่อน fix นี้/จาก writer อื่น) ถือเป็น miss"""
    cached = web_cache.get(url)
    if cached is not None and not cached.endswith("...[cache-truncated]"):
        return cached
    if state.get("budget_err"):
        return ""
    if not state["budget_checked"]:
        state["budget_checked"] = True
        err = _wc_check_and_inc()
        if err:
            state["budget_err"] = err
            return ""
    xml = _decode_body(_curl_bytes(url))
    if xml and len(xml) <= WEB_CACHE_PER_ENTRY_MAX:
        web_cache.put(url, xml)
    return xml


def _find_sitemap_via_robots(base: str, state: dict) -> list[str]:
    """ตรวจ robots.txt เพื่อหา Sitemap: URL ทั้งหมด (บาง robots.txt มีหลายบรรทัด)"""
    robots = _cached_fetch(f"{base}/robots.txt", state)
    if state.get("budget_err") or not robots:
        return []
    return [m.strip() for m in re.findall(r'(?im)^Sitemap:\s*(\S+)', robots) if m.strip()]


def _parse_locs(xml: str) -> list[str]:
    """Flat <loc> extraction — fallback for sitemaps without <url>/<sitemap> wrapper tags."""
    return [html.unescape(u) for u in re.findall(r'<loc>\s*([^<\s]+)\s*</loc>', xml, re.IGNORECASE)]


_LOC_RE = re.compile(r'<loc>\s*([^<\s]+)\s*</loc>', re.IGNORECASE)
_LASTMOD_RE = re.compile(r'<lastmod>\s*([^<\s]+)\s*</lastmod>', re.IGNORECASE)


def _iter_blocks(xml: str):
    """Linear str.find() scan ของ <url>/<sitemap> blocks — regex non-greedy DOTALL
    เป็น quadratic บน XML ที่ truncated/malformed (unclosed tags × body size, block GIL
    ทั้ง process). "<sitemap>" ไม่ false-match <sitemapindex เพราะ find รวม '>' ปิด tag;
    tag เปิด/ปิดต้องชนิดเดียวกัน — <url>...</sitemap> ไม่ cross-match"""
    low = xml.lower()
    pos = 0
    while True:
        starts = [(i, t) for i, t in ((low.find(t, pos), t) for t in ("<url>", "<sitemap>")) if i != -1]
        if not starts:
            return
        i, t = min(starts)
        close = "</" + t[1:]
        j = low.find(close, i + len(t))
        if j == -1:
            return
        yield xml[i + len(t):j]
        pos = j + len(close)


def _parse_entries(xml: str) -> list[tuple[str, str]]:
    """Parse <url>/<sitemap> blocks → [(loc, lastmod)], lastmod="" if missing.
    XML entities ใน loc ถูก unescape (&amp; → &) ให้ fetch ปลายทางเจอ URL จริง.
    Falls back to flat <loc>-only extraction (old _parse_locs behavior) when
    no <url>/<sitemap> wrapper tags are found."""
    entries: list[tuple[str, str]] = []
    for block in _iter_blocks(xml):
        loc_m = _LOC_RE.search(block)
        if not loc_m:
            continue
        lm_m = _LASTMOD_RE.search(block)
        entries.append((html.unescape(loc_m.group(1).strip()), lm_m.group(1).strip() if lm_m else ""))
    if not entries:
        return [(u, "") for u in _parse_locs(xml)]
    return entries


_ISO_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}')


def _sort_by_lastmod(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """เรียง entries ที่มี lastmod รูปแบบ ISO (YYYY-MM-DD...) ใหม่→เก่า (stable — ties
    คงลำดับเดิม); lastmod ว่าง/รูปแบบอื่น (เช่น RFC 822) ต่อท้ายตามลำดับเดิม —
    string-compare กับ non-ISO จะให้ตัวอักษรชนะตัวเลข (garbage outranks dates)"""
    dated = [e for e in entries if _ISO_DATE_RE.match(e[1])]
    undated = [e for e in entries if not _ISO_DATE_RE.match(e[1])]
    dated.sort(key=lambda e: e[1][:10], reverse=True)
    return dated + undated


def _split_keywords(filter_keyword: str) -> list[str]:
    """แยก filter_keyword ด้วย space/comma/+ → lowercase list ไม่มีค่าว่าง"""
    if not filter_keyword or not filter_keyword.strip():
        return []
    parts = re.split(r'[,\+\s]+', filter_keyword.strip())
    return [p.lower() for p in parts if p]


def _select_children(
    entries: list[tuple[str, str]], filter_keywords: list[str], max_child: int
) -> list[tuple[str, str]]:
    """เลือก child sitemap สูงสุด max_child ตัว: keyword-match ก่อน (เรียง lastmod
    ใหม่→เก่าภายในกลุ่ม — child รายปี fund-2020..2026 ต้องได้ปีใหม่สุด ไม่ใช่
    document order), แล้ว lastmod ใหม่→เก่า, แล้ว original order"""
    if not entries:
        return []
    if filter_keywords:
        matched = _sort_by_lastmod(
            [e for e in entries if any(k in e[0].lower() for k in filter_keywords)]
        )
        matched_urls = {e[0] for e in matched}
        rest = [e for e in entries if e[0] not in matched_urls]
    else:
        matched, rest = [], list(entries)
    return (matched + _sort_by_lastmod(rest))[:max_child]


def _is_sitemapindex(xml: str) -> bool:
    """ตรวจ root element จริง — substring ทั้ง doc จะ misroute urlset ที่มี comment
    พูดถึง sitemapindex"""
    head = re.sub(r'^\s*(?:<\?[^>]*\?>\s*|<!--.*?-->\s*)*', '', xml[:4000], flags=re.DOTALL)
    return head.lower().startswith("<sitemapindex")


def _fetch_all_urls(
    sitemap_url: str, state: dict, filter_keywords: list[str] | None = None, max_child: int = 3
) -> list[tuple[str, str]]:
    """Fetch sitemap; ถ้าเป็น sitemapindex → เลือก+fetch child sitemaps สูงสุด max_child ตัว"""
    _phase(f"📋 sitemap: {sitemap_url}")
    xml = _cached_fetch(sitemap_url, state)
    if state.get("budget_err") or not xml:
        return []
    entries = _parse_entries(xml)
    if not entries:
        return []

    if _is_sitemapindex(xml):
        # entries คือ child sitemap URLs (loc + optional lastmod) — เลือกแล้ว fetch แต่ละตัว
        children = _select_children(entries, filter_keywords or [], max_child)
        all_entries: list[tuple[str, str]] = []
        for child_url, _lm in children:
            _phase(f"📋 child sitemap: {child_url}")
            child_xml = _cached_fetch(child_url, state)
            if state.get("budget_err"):
                return all_entries
            if child_xml:
                all_entries.extend(_parse_entries(child_xml))
        return all_entries

    return entries


@tool
def fetch_sitemap(domain_or_url: str, filter_keyword: str = "") -> str:
    """Fetch the URL list from a website's sitemap.xml — use when you need the COMPLETE list
    of URLs from a single domain (e.g. all funds on finnomena, every product, every article).

    !! Use INSTEAD of web_search when the data lives on one known website:
       sitemap.xml gives the full URL list ready to parse — web_search returns generic articles, not a list.

    domain_or_url: domain or sitemap URL, e.g. "finnomena.com" or "https://example.com/sitemap.xml"
    filter_keyword: (optional) one or more keywords separated by space/comma/+, e.g. "RMF fund".
        Matches ALL keywords first (AND); if that finds nothing and ≥2 keywords were given,
        falls back to ANY keyword (OR).

    Returns: URLs sorted newest→oldest by <lastmod> (up to FETCH_SITEMAP_MAX_URLS kept,
        FETCH_SITEMAP_SHOWN shown per call) — pick ≤8 of the most relevant and call
        batch_browse ONCE with all of them. Or "[error] reason" if no sitemap was found.
    """
    try:
        s = (domain_or_url or "").strip()
        if not s:
            return "[error] domain_or_url is required"
        if not s.startswith("http"):
            s = "https://" + s.lstrip("/")
        # ตัด path ที่ไม่ใช่ sitemap ออกเพื่อหา base
        base = re.sub(r'(/sitemap[^/]*\.xml.*|/robots\.txt.*)$', '', s).rstrip("/")
        # รองรับทั้ง .xml และ .xml.gz — endswith(".xml") อย่างเดียวจะทิ้ง URL .gz ที่ caller ส่งมา
        sitemap_url = s if re.search(r'\.xml(\.gz)?$', s.lower()) else f"{base}/sitemap.xml"

        keywords = _split_keywords(filter_keyword)
        state: dict = {"budget_checked": False, "budget_err": None}

        entries = _fetch_all_urls(sitemap_url, state, filter_keywords=keywords)
        if state.get("budget_err"):
            return state["budget_err"]

        # Fallback: ตรวจ robots.txt — ลองทุกบรรทัด Sitemap: จนกว่าจะเจอ
        if not entries:
            robots_sitemaps = _find_sitemap_via_robots(base, state)
            if state.get("budget_err"):
                return state["budget_err"]
            tried = {sitemap_url}
            for candidate in robots_sitemaps:
                if candidate in tried:
                    continue
                tried.add(candidate)
                entries = _fetch_all_urls(candidate, state, filter_keywords=keywords)
                if state.get("budget_err"):
                    return state["budget_err"]
                if entries:
                    break

        if not entries:
            return (
                f"[error] ไม่พบ sitemap ที่ {sitemap_url} — "
                f"ลอง browse_url '{base}/sitemap_index.xml' หรือ web_search แทน"
            )

        # dedup URL คงลำดับเดิม + เก็บ lastmod แรกที่เจอ
        deduped: dict[str, str] = {}
        for url, lastmod in entries:
            if url not in deduped:
                deduped[url] = lastmod
        all_urls = list(deduped.items())
        total_before_filter = len(all_urls)

        filter_note = ""
        if keywords:
            and_matched = [(u, lm) for u, lm in all_urls if all(k in u.lower() for k in keywords)]
            if and_matched:
                matched = and_matched
            elif len(keywords) >= 2:
                or_matched = [(u, lm) for u, lm in all_urls if any(k in u.lower() for k in keywords)]
                matched = or_matched
                if or_matched:
                    filter_note = (
                        f" (filter: '{filter_keyword}' — ไม่มี URL ที่ match ทุกคำ, แสดงแบบ match บางคำ)"
                    )
            else:
                matched = []

            if not matched:
                sample = [u for u, _ in all_urls[:10]]
                return (
                    f"[fetch_sitemap] sitemap มี {total_before_filter} URLs แต่ไม่มีที่ match '{filter_keyword}'\n"
                    f"ตัวอย่าง 10 URL แรก:\n" + "\n".join(sample)
                )
            all_urls = matched

        # total จริงต้องนับก่อน cap — ไม่งั้น sitemap 3000 URLs รายงานว่า "พบ 200"
        total = len(all_urls)
        # sort ก่อนแล้วค่อย cap — cap ก่อนจะตัด URL ใหม่สุดที่อยู่ท้าย sitemap ทิ้ง
        sorted_urls = _sort_by_lastmod(all_urls)[:FETCH_SITEMAP_MAX_URLS]
        shown_list = sorted_urls[:FETCH_SITEMAP_SHOWN]
        shown = len(shown_list)
        has_lastmod = any(lm for _, lm in sorted_urls)

        suffix = filter_note if filter_note else (f" (filter: '{filter_keyword}')" if keywords else "")
        header = f"[fetch_sitemap] พบ {total} URLs{suffix}"
        if total > shown:
            header += f" — แสดง {shown} รายการแรก"
        if has_lastmod:
            header += " (เรียงใหม่→เก่า)"

        lines = []
        for url, lastmod in shown_list:
            date = lastmod[:10] if lastmod else ""
            lines.append(f"{date}  {url}" if date else url)

        footer = "→ เลือก URL ที่เกี่ยวข้องที่สุด ≤8 รายการ แล้วเรียก batch_browse ในครั้งเดียว"

        return header + "\n" + "\n".join(lines) + "\n" + footer
    except Exception as e:
        log.error(f"[fetch_sitemap] unexpected error: {e}")
        return f"[error] fetch_sitemap failed: {e}"
