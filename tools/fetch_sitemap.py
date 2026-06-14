"""fetch_sitemap.py — ดึง URL list จาก sitemap.xml ของเว็บไซต์

รองรับ:
  - <sitemapindex> (nested child sitemaps — fetch สูงสุด 3 child)
  - <urlset>       (direct URL list)
  - robots.txt fallback เพื่อหา Sitemap: location
"""
from __future__ import annotations
import logging
import re
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from tools._progress import phase as _phase
from config import FETCH_SITEMAP_MAX_URLS

log = logging.getLogger(__name__)


def _curl(url: str) -> str:
    try:
        r = subprocess.run(
            ["curl", "-s", "-L", "--max-time", "20", url],
            capture_output=True, text=True, timeout=25,
        )
        if r.returncode != 0:
            log.warning(f"[fetch_sitemap] curl rc={r.returncode} for {url}: {r.stderr.strip()[:200]}")
        return r.stdout
    except FileNotFoundError:
        log.error("[fetch_sitemap] curl binary not found — cannot fetch sitemap")
        return ""
    except Exception as e:
        log.warning(f"[fetch_sitemap] curl error for {url}: {e}")
        return ""


def _find_sitemap_via_robots(base: str) -> str:
    """ตรวจ robots.txt เพื่อหา Sitemap: URL"""
    robots = _curl(f"{base}/robots.txt")
    m = re.search(r'(?im)^Sitemap:\s*(\S+)', robots)
    return m.group(1).strip() if m else ""


def _parse_locs(xml: str) -> list[str]:
    return re.findall(r'<loc>\s*([^<\s]+)\s*</loc>', xml, re.IGNORECASE)


def _fetch_all_urls(sitemap_url: str, max_child: int = 3) -> list[str]:
    """Fetch sitemap; ถ้าเป็น sitemapindex → ดึง child sitemaps สูงสุด max_child ตัว"""
    _phase(f"📋 sitemap: {sitemap_url}")
    xml = _curl(sitemap_url)
    if not xml:
        return []
    locs = _parse_locs(xml)
    if not locs:
        return []

    if "<sitemapindex" in xml.lower():
        # locs เป็น URL ของ child sitemaps — fetch แต่ละตัว
        all_urls: list[str] = []
        for child_url in locs[:max_child]:
            _phase(f"📋 child sitemap: {child_url}")
            child_xml = _curl(child_url)
            if child_xml:
                all_urls.extend(_parse_locs(child_xml))
        return all_urls

    return locs


@tool
def fetch_sitemap(domain_or_url: str, filter_keyword: str = "") -> str:
    """ดึง list URL จาก sitemap.xml ของเว็บไซต์ — ใช้เมื่อต้องการรายการ URL ครบถ้วนจาก domain เดียว
    เช่น กองทุนทั้งหมดใน finnomena, สินค้าทุกชิ้น, บทความทุกชิ้น

    !! ใช้แทน web_search เมื่อรู้ว่าข้อมูลอยู่ใน website ใดเว็บหนึ่ง
       sitemap.xml มี URL ครบพร้อม parse ทันที — web_search คืน article ทั่วไปไม่ใช่ list

    domain_or_url: โดเมนหรือ URL ของ sitemap เช่น "finnomena.com" หรือ "https://example.com/sitemap.xml"
    filter_keyword: (optional) กรองเฉพาะ URL ที่มีคีย์เวิร์ดนี้ เช่น "RMF", "/fund/", "2024"

    คืน: รายการ URLs ที่ match (สูงสุด FETCH_SITEMAP_MAX_URLS, default 200) หรือ "[error] reason"
    """
    s = domain_or_url.strip()
    if not s.startswith("http"):
        s = "https://" + s.lstrip("/")
    # ตัด path ที่ไม่ใช่ sitemap ออกเพื่อหา base
    base = re.sub(r'(/sitemap[^/]*\.xml.*|/robots\.txt.*)$', '', s).rstrip("/")
    sitemap_url = s if s.lower().endswith(".xml") else f"{base}/sitemap.xml"

    urls = _fetch_all_urls(sitemap_url)

    # Fallback: ตรวจ robots.txt
    if not urls:
        robots_sitemap = _find_sitemap_via_robots(base)
        if robots_sitemap and robots_sitemap != sitemap_url:
            urls = _fetch_all_urls(robots_sitemap)

    if not urls:
        return (
            f"[fetch_sitemap] ไม่พบ sitemap ที่ {sitemap_url}\n"
            f"ลอง: browse_url '{base}/sitemap_index.xml' หรือ web_search เพื่อหา URL เอง"
        )

    # Filter
    if filter_keyword:
        fk = filter_keyword.lower()
        filtered = [u for u in urls if fk in u.lower()]
        if not filtered:
            return (
                f"[fetch_sitemap] sitemap มี {len(urls)} URLs แต่ไม่มีที่ match '{filter_keyword}'\n"
                f"ตัวอย่าง 10 URL แรก:\n" + "\n".join(urls[:10])
            )
        urls = filtered

    total = len(urls)
    shown = urls[:FETCH_SITEMAP_MAX_URLS]
    suffix = f" (filter: '{filter_keyword}')" if filter_keyword else ""
    return f"[fetch_sitemap] พบ {total} URLs{suffix}\n" + "\n".join(shown)
