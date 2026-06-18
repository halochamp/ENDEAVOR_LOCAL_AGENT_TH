# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""_freshness.py — deterministic date extraction + staleness check for scraped content.

Ported from ENDEAVOR_LOCAL_AGENT_MAX Session 51b: an earlier version asked the
summarization LLM to decide whether content was stale. Live test showed this is
unreliable — the model can invent a date that doesn't appear in the source and
flag staleness relative to a date in the FUTURE. Date comparison must be code,
not an LLM judgment call.
"""
from __future__ import annotations
import datetime
import re

_TH_MONTHS = {
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4, "พ.ค.": 5, "มิ.ย.": 6,
    "ก.ค.": 7, "ส.ค.": 8, "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
}
_EN_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}

_ISO_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b")  # DD/MM/YYYY
_TH_MONTH_PAT = "|".join(re.escape(m) for m in _TH_MONTHS)
_TH_RE = re.compile(rf"\b(\d{{1,2}})\s*(?:{_TH_MONTH_PAT})\s*(20\d{{2}})?")
_TH_MONTH_FIND_RE = re.compile(rf"({_TH_MONTH_PAT})")
_EN_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(20\d{2})\b",
    re.IGNORECASE,
)
_URL_DATE_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")


def _safe_date(y: int, m: int, d: int) -> datetime.date | None:
    try:
        return datetime.date(y, m, d)
    except ValueError:
        return None


def extract_dates(text: str, today: datetime.date) -> list[datetime.date]:
    """Find concrete, year-bearing dates in text. Yearless dates (e.g. a bare
    "27 มิ.ย." with no year) are intentionally NOT resolved by guessing a year —
    a forward-looking sentence ("ประชุมวันที่ 27 มิ.ย. จะแถลงผล") and a same-year
    past date look identical without one, and guessing wrong fabricates a false
    staleness signal. Ambiguous dates are left for react.py's synthesis-time rule."""
    found: list[datetime.date] = []
    for y, m, d in _ISO_RE.findall(text):
        dt = _safe_date(int(y), int(m), int(d))
        if dt:
            found.append(dt)
    for d, m, y in _SLASH_RE.findall(text):
        dt = _safe_date(int(y), int(m), int(d))
        if dt:
            found.append(dt)
    for m in _TH_RE.finditer(text):
        year_str = m.group(2)
        if not year_str:
            continue
        day = int(m.group(1))
        month_text = _TH_MONTH_FIND_RE.search(m.group(0))
        if not month_text:
            continue
        month = _TH_MONTHS[month_text.group(1)]
        dt = _safe_date(int(year_str), month, day)
        if dt:
            found.append(dt)
    for mon, d, y in _EN_RE.findall(text):
        dt = _safe_date(int(y), _EN_MONTHS[mon.lower()], int(d))
        if dt:
            found.append(dt)
    return found


def url_date(url: str) -> datetime.date | None:
    m = _URL_DATE_RE.search(url or "")
    if not m:
        return None
    return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def staleness_note(raw: str, url: str, today: datetime.date, max_age_days: int = 1) -> str | None:
    """Pure date math, no LLM. Returns a Thai staleness warning if the most recent
    date found in the content (or URL) is older than max_age_days. Future dates
    found in text are discarded (can't be the content's own publish date)."""
    dates = [d for d in extract_dates(raw, today) if d <= today]
    u = url_date(url)
    if u and u <= today:
        dates.append(u)
    if not dates:
        return None
    most_recent = max(dates)
    age = (today - most_recent).days
    if age > max_age_days:
        return f"⚠️ ข้อมูล ณ {most_recent.isoformat()} ไม่ใช่ข้อมูลล่าสุด (วันนี้ {today.isoformat()}, เก่ากว่า {age} วัน)"
    return None
