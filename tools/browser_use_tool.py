# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""browser_use_tool.py — human-like web browsing via browser-use + Playwright

ใช้สำหรับ: เจาะลึกเว็บ, login, กรอก form, navigate หลายชั้น, JS-heavy ที่ Jina ทำไม่ได้
agent ระบุ URL + task เป็นภาษาธรรมชาติ — browser-use จัดการ click/scroll/type เอง

Cache-aware: after browser-use returns, summarizes the result (query-aware
if user_query provided) and caches it under url. Returns only the compact
`[web:<url>] <summary>` tag. Use recall_web(url) to retrieve full body.
"""
from __future__ import annotations
import asyncio
import logging
import sys
import os
import time
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import web_cache
from tools._summarize import summarize
from tools._progress import progress as _progress, phase as _phase

from config import BROWSER_USE_MAX_CHARS

log = logging.getLogger(__name__)

_TIMEOUT = int(os.getenv("V2_BROWSER_TIMEOUT", "120"))  # seconds


@tool
def browser_use(url: str, task: str, user_query: str = "") -> str:
    """Browse a website like a human — can click, scroll, fill forms, navigate multiple pages.
    Use ONLY when: user explicitly asks to "open" or "browse" a specific site, need to login, need to interact with JS-heavy pages, or browse_url returns insufficient content.
    Returns a compact Thai summary tagged with the URL — full result is cached. Call recall_web(url) for full body.
    Args:
        url: starting URL
        task: what to do or find on the site (in natural language)
        user_query: the user's current question — used to produce a query-focused summary"""
    _phase(f"🌐 เปิด browser: {url[:42]}")
    try:
        from browser_use import Agent as BUAgent
        from langchain_openai import ChatOpenAI
        from config import MLX_BASE_URL
    except ImportError as e:
        return f"[error] browser-use not installed: {e}"

    # Check cache before starting browser
    cached = web_cache.get(url)
    if cached is not None:
        _progress(f"cache HIT (skip browser launch): {url[:60]}")
        from config import SUMMARY_MAX_CHARS
        effective_uq_read = (user_query or "").strip() or task
        cached_sum = (web_cache.get_summary(url, effective_uq_read)
                      or web_cache.get_summary(url, "")
                      or cached[:SUMMARY_MAX_CHARS])
        return f"[web:{url}] {cached_sum}"

    _progress(f"launching Chromium for: {url[:60]}")
    t0 = time.time()

    async def _run() -> str:
        try:
            llm = ChatOpenAI(
                base_url=MLX_BASE_URL,
                api_key="local",
                model="local-model",
                temperature=0.1,
            )
            agent = BUAgent(
                task=f"Go to {url}. {task}",
                llm=llm,
            )
            result = await asyncio.wait_for(agent.run(), timeout=_TIMEOUT)
            text = str(result)
            if len(text) > BROWSER_USE_MAX_CHARS:
                text = text[:BROWSER_USE_MAX_CHARS] + f"\n...[truncated at {BROWSER_USE_MAX_CHARS} chars]"
            return text or "(no result)"
        except asyncio.TimeoutError:
            return f"[error] browser_use timed out after {_TIMEOUT}s"
        except Exception as e:
            return f"[error] browser_use failed: {e}"

    try:
        raw = asyncio.run(_run())
        _progress(f"browser done in {time.time()-t0:.1f}s ({len(raw)} chars)")
    except Exception as e:
        _progress(f"browser failed: {e}")
        return f"[error] browser_use runner failed: {e}"

    if raw.startswith("[error]"):
        return raw  # don't cache errors

    # Use task as supplemental context if user_query is empty.
    effective_uq = (user_query or "").strip() or task
    raw_with_task = f"# Task: {task}\n# URL: {url}\n\n{raw}"
    summary = summarize(raw_with_task, url=url, user_query=effective_uq)
    web_cache.put(url, raw_with_task)
    web_cache.put_summary(url, effective_uq, summary)
    return f"[web:{url}] {summary}"
