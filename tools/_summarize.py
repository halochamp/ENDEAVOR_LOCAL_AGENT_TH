# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""_summarize.py — query-aware summarization of web content

Reuses the same Qwen3-35B used by the agent (via build_llm).
Output is Thai, ≤ SUMMARY_MAX_CHARS chars (hard truncate if model overshoots).

On any LLM failure → graceful fallback: first SUMMARY_MAX_CHARS chars of raw + "...".
Short content (≤ SUMMARY_SKIP_LLM_BELOW) skips the LLM and uses raw as its own summary.
"""
from __future__ import annotations
import logging
import re
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUMMARY_MAX_CHARS, SUMMARY_SKIP_LLM_BELOW, SUMMARY_MAX_TOKENS, SUMMARY_BATCH_MAX_TOKENS
from tools._progress import progress as _progress

log = logging.getLogger(__name__)

_LLM_CACHE: dict = {}
_BATCH_LLM_CACHE: dict = {}

_SOURCE_MARKER_RE = re.compile(r"===\s*SOURCE\s*(\d+)\s*===")


def _get_summarize_llm(temperature: float):
    """Return a cached ChatOpenAI instance for summarization (no-thinking)."""
    if temperature not in _LLM_CACHE:
        from llm import build_llm
        _LLM_CACHE[temperature] = build_llm(
            temperature=temperature,
            max_tokens=SUMMARY_MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    return _LLM_CACHE[temperature]


def _get_batch_summarize_llm(temperature: float):
    """Return a cached ChatOpenAI instance for batch summarization (no-thinking,
    higher max_tokens to fit N sectioned summaries)."""
    if temperature not in _BATCH_LLM_CACHE:
        from llm import build_llm
        _BATCH_LLM_CACHE[temperature] = build_llm(
            temperature=temperature,
            max_tokens=SUMMARY_BATCH_MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    return _BATCH_LLM_CACHE[temperature]


def _hard_truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3].rstrip() + "..."


def _parse_batch_response(text: str, n: int) -> dict[int, str] | None:
    """Split a "===SOURCE i===\\n<summary>" formatted response into {i: summary}.

    Returns None if any of the n expected sections is missing or empty —
    the caller falls back to per-URL summarize() in that case.
    """
    pieces = _SOURCE_MARKER_RE.split(text)
    result: dict[int, str] = {}
    for i in range(1, len(pieces), 2):
        try:
            idx = int(pieces[i])
        except ValueError:
            continue
        body = pieces[i + 1].strip() if i + 1 < len(pieces) else ""
        if body:
            result[idx] = body
    if len(result) != n or any(i not in result for i in range(1, n + 1)):
        return None
    return result


def summarize(raw: str, url: str, user_query: str | None = None) -> str:
    """Summarize raw web content. Query-aware if user_query is non-empty."""
    raw = (raw or "").strip()
    if not raw:
        return "(เนื้อหาว่าง)"

    # Short content → use raw as summary, save an LLM call.
    # SUMMARY_SKIP_LLM_BELOW (default 1500) covers most trafilatura outputs
    if len(raw) <= SUMMARY_SKIP_LLM_BELOW:
        _progress(f"raw ≤{SUMMARY_SKIP_LLM_BELOW} chars — skip LLM, use raw as summary")
        return _hard_truncate(raw, SUMMARY_MAX_CHARS)

    raw = raw[:10_000]

    q = (user_query or "").strip()
    if q:
        prompt = (
            "คุณคือผู้ช่วยสรุปเนื้อหาเว็บ ตอบเป็นภาษาไทยเท่านั้น\n"
            f"คำถามของผู้ใช้: {q}\n"
            f"URL: {url}\n\n"
            "งาน: สรุปเนื้อหาด้านล่างโดยเน้นข้อมูลที่ตอบคำถามของผู้ใช้\n"
            f"- ความยาวไม่เกิน {SUMMARY_MAX_CHARS} ตัวอักษร\n"
            "- ระบุข้อเท็จจริงและตัวเลขที่เกี่ยวข้อง\n"
            "- ห้ามแต่งข้อมูลที่ไม่อยู่ในเนื้อหา\n"
            "- ตอบเฉพาะเนื้อสรุป ไม่ต้องมีคำนำหรือคำลงท้าย\n\n"
            "เนื้อหา:\n"
            f"{raw}"
        )
    else:
        prompt = (
            "คุณคือผู้ช่วยสรุปเนื้อหาเว็บ ตอบเป็นภาษาไทยเท่านั้น\n"
            f"URL: {url}\n\n"
            f"งาน: สรุปเนื้อหาด้านล่าง ความยาวไม่เกิน {SUMMARY_MAX_CHARS} ตัวอักษร\n"
            "- ระบุประเด็นหลัก ข้อเท็จจริง และตัวเลขสำคัญ\n"
            "- ห้ามแต่งข้อมูลที่ไม่อยู่ในเนื้อหา\n"
            "- ตอบเฉพาะเนื้อสรุป ไม่ต้องมีคำนำหรือคำลงท้าย\n\n"
            "เนื้อหา:\n"
            f"{raw}"
        )

    # Disable Qwen3 thinking mode for summarization — it's a straightforward
    # extraction task, no reasoning needed. With thinking off, latency drops
    # from ~30s → ~3s per call (10× faster, verified 2026-05-29).
    _progress(f"summarizing {len(raw)} chars (≤{SUMMARY_MAX_CHARS})…")
    t0 = time.time()
    last_error = None
    for attempt, temp in enumerate([0.1, 0.5], 1):
        try:
            llm = _get_summarize_llm(temp)
            resp = llm.invoke(prompt, config={"callbacks": []})
            text = (resp.content or "").strip()
            if not text:
                last_error = "empty summary"
                if attempt == 1:
                    _progress(f"attempt 1 empty — retrying with temperature=0.5")
                continue
            _progress(f"summary ready ({len(text)} chars, {time.time()-t0:.1f}s, attempt={attempt})")
            return _hard_truncate(text, SUMMARY_MAX_CHARS)
        except Exception as e:
            last_error = str(e)
            if attempt == 1:
                _progress(f"attempt 1 failed: {e} — retrying")

    log.warning(f"[_summarize] LLM failed for {url[:60]} after 2 attempts: {last_error}")
    _progress(f"summary failed after 2 attempts — using raw prefix fallback")
    return _hard_truncate(raw, SUMMARY_MAX_CHARS)


def summarize_batch(items: list[tuple[str, str]], user_query: str | None = None) -> dict[str, str] | None:
    """Summarize multiple raw bodies in a single LLM call.

    items: [(url, raw), ...] — only items that need LLM summarization
    Returns {url: summary} on success, or None to signal fallback to per-URL summarize().
    """
    items = [(u, (r or "").strip()) for u, r in items if (r or "").strip()]
    if len(items) < 2:
        return None

    q = (user_query or "").strip()
    lines = ["คุณคือผู้ช่วยสรุปเนื้อหาเว็บ ตอบเป็นภาษาไทยเท่านั้น"]
    if q:
        lines.append(f"คำถามของผู้ใช้: {q}")
    lines.append("")
    lines.append(
        "งาน: สรุปเนื้อหาแต่ละแหล่งแยกกัน"
        + (" โดยเน้นข้อมูลที่ตอบคำถามของผู้ใช้" if q else "")
    )
    lines.append(f"- ความยาวไม่เกิน {SUMMARY_MAX_CHARS} ตัวอักษรต่อแหล่ง")
    lines.append("- ระบุข้อเท็จจริงและตัวเลข รวมถึงวันที่ของข้อมูล (ข้อมูล ณ ...) ถ้ามี")
    lines.append("- ห้ามแต่งข้อมูลที่ไม่อยู่ในเนื้อหา")
    lines.append(f"- ตอบทั้ง {len(items)} แหล่งโดยใช้ format นี้เท่านั้น ไม่มีข้อความอื่นนอก format:")
    lines.append("")
    for i in range(1, len(items) + 1):
        lines.append(f"===SOURCE {i}===")
        lines.append(f"<สรุปของแหล่งที่ {i}>")
    lines.append("")
    lines.append("เนื้อหาแต่ละแหล่ง:")
    lines.append("")
    for i, (url, raw) in enumerate(items, 1):
        lines.append(f"[SOURCE {i}: {url}]")
        lines.append(raw[:10_000])
        lines.append("")
    prompt = "\n".join(lines)

    _progress(f"batch summarizing {len(items)} sources ({len(prompt)} chars)…")
    t0 = time.time()
    last_error = None
    for attempt, temp in enumerate([0.1, 0.5], 1):
        try:
            llm = _get_batch_summarize_llm(temp)
            resp = llm.invoke(prompt, config={"callbacks": []})
            text = (resp.content or "").strip()
            parsed = _parse_batch_response(text, len(items))
            if parsed is None:
                last_error = "parse failed or incomplete sections"
                if attempt == 1:
                    _progress("batch attempt 1 unparseable — retrying")
                continue
            _progress(f"batch summary ready ({len(items)} sources, {time.time()-t0:.1f}s, attempt={attempt})")
            return {items[i - 1][0]: _hard_truncate(parsed[i], SUMMARY_MAX_CHARS) for i in range(1, len(items) + 1)}
        except Exception as e:
            last_error = str(e)
            if attempt == 1:
                _progress(f"batch attempt 1 failed: {e} — retrying")

    log.warning(f"[_summarize] batch summarize failed after 2 attempts: {last_error}")
    _progress("batch summary failed — falling back to per-URL summarize")
    return None
