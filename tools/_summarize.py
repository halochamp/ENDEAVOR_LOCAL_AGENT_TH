# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""_summarize.py — query-aware summarization of web content

Reuses the same production model used by the agent (via build_llm).
Output is Thai, ≤ SUMMARY_MAX_CHARS chars (hard truncate if model overshoots).

On any LLM failure → graceful fallback: query-relevant lines (or first
SUMMARY_MAX_CHARS chars) of raw. Short content (≤ SUMMARY_SKIP_LLM_BELOW)
skips the LLM and uses raw as its own summary.
"""
from __future__ import annotations
import datetime
import hashlib
import logging
import re
import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SUMMARY_MAX_CHARS,
    SUMMARY_SKIP_LLM_BELOW,
    SUMMARY_MAX_TOKENS,
    SUMMARY_BATCH_MAX_TOKENS,
)
from tools._progress import progress as _progress
from tools._freshness import staleness_note

log = logging.getLogger(__name__)

_LLM_CACHE: dict = {}
_BATCH_LLM_CACHE: dict = {}
_LLM_CACHE_LOCK = threading.Lock()

# (url, query, day, md5(raw)) → finished summary. Re-browsing the same URL
# within a ReAct session pays the LLM call only once (same idea as read_image's
# re-look cache). FIFO-evicted; process-lifetime only.
_SUMMARY_CACHE: dict[tuple[str, str, str, str], str] = {}
_SUMMARY_CACHE_MAX = 64
_SUMMARY_CONDITION = threading.Condition()


class _SummaryFlight:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: str | None = None


_SUMMARY_INFLIGHT: dict[tuple[str, str, str, str], _SummaryFlight] = {}


def _cache_key(url: str, q: str, raw: str, day: str | None = None) -> tuple[str, str, str, str]:
    # day is part of the key: the cached value embeds a staleness note computed
    # against "today", so an entry must not survive a date rollover on a
    # long-running server (it would understate how old the data is).
    if day is None:
        day = datetime.date.today().isoformat()
    return (url, q, day, hashlib.md5(raw.encode("utf-8", "ignore")).hexdigest())


def _cache_get(key: tuple[str, str, str, str]) -> str | None:
    with _SUMMARY_CONDITION:
        return _SUMMARY_CACHE.get(key)


def _cache_put(key: tuple[str, str, str, str], value: str) -> None:
    with _SUMMARY_CONDITION:
        # Updating an existing entry must not evict an unrelated one when the
        # cache is full (batch summarization can refresh an already-seen key).
        if key not in _SUMMARY_CACHE and len(_SUMMARY_CACHE) >= _SUMMARY_CACHE_MAX:
            _SUMMARY_CACHE.pop(next(iter(_SUMMARY_CACHE)))
        _SUMMARY_CACHE[key] = value


def _cache_claim(
    key: tuple[str, str, str, str],
) -> tuple[str | None, bool, _SummaryFlight | None]:
    """Return ``(cached_value, is_owner, flight)`` for one summary key.

    Identical ToolNode calls can arrive on separate worker threads. Exactly one
    becomes the inference owner; followers wait on its event without holding the
    cache lock, then consume the same result. The per-flight result also shares
    a non-cached fallback with current followers while allowing a later request
    to retry a transient model failure.
    """
    with _SUMMARY_CONDITION:
        cached = _SUMMARY_CACHE.get(key)
        if cached is not None:
            return cached, False, None
        flight = _SUMMARY_INFLIGHT.get(key)
        if flight is not None:
            return None, False, flight
        flight = _SummaryFlight()
        _SUMMARY_INFLIGHT[key] = flight
        return None, True, flight


def _cache_claim_many(
    keys: list[tuple[str, str, str, str]],
) -> tuple[
    dict[tuple[str, str, str, str], str],
    dict[tuple[str, str, str, str], _SummaryFlight],
    dict[tuple[str, str, str, str], _SummaryFlight],
]:
    """Resolve/claim many summary keys in ONE lock acquisition.

    Returns ``(cached, owned, followed)``. Claiming the whole set atomically is
    what keeps overlapping batches deadlock-free: a caller can only ever follow
    a flight that already existed when it claimed, so the wait-for graph has no
    cycles — provided the caller publishes its own flights *before* waiting on
    followed ones (see ``summarize_batch``). Claiming key-by-key would let two
    crossing batches each own what the other waits for.
    """
    cached: dict[tuple[str, str, str, str], str] = {}
    owned: dict[tuple[str, str, str, str], _SummaryFlight] = {}
    followed: dict[tuple[str, str, str, str], _SummaryFlight] = {}
    with _SUMMARY_CONDITION:
        for key in keys:
            if key in cached or key in owned or key in followed:
                continue
            hit = _SUMMARY_CACHE.get(key)
            if hit is not None:
                cached[key] = hit
                continue
            flight = _SUMMARY_INFLIGHT.get(key)
            if flight is not None:
                followed[key] = flight
                continue
            flight = _SummaryFlight()
            _SUMMARY_INFLIGHT[key] = flight
            owned[key] = flight
    return cached, owned, followed


def _cache_finish(
    key: tuple[str, str, str, str],
    flight: _SummaryFlight,
    result: str | None,
) -> None:
    with _SUMMARY_CONDITION:
        flight.result = result
        if _SUMMARY_INFLIGHT.get(key) is flight:
            _SUMMARY_INFLIGHT.pop(key, None)
        flight.event.set()

_SOURCE_MARKER_RE = re.compile(r"===\s*SOURCE\s*(\d+)\s*===")


def _get_summarize_llm(temperature: float):
    """Return a cached ChatOpenAI instance for summarization (no-thinking)."""
    with _LLM_CACHE_LOCK:
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
    with _LLM_CACHE_LOCK:
        if temperature not in _BATCH_LLM_CACHE:
            from llm import build_llm
            _BATCH_LLM_CACHE[temperature] = build_llm(
                temperature=temperature,
                max_tokens=SUMMARY_BATCH_MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        return _BATCH_LLM_CACHE[temperature]


RAW_CAP = 10_000  # max chars of content fed to the summarize LLM per source

_TERM_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*|[฀-๿]{2,}")
_THAI_RE = re.compile(r"[฀-๿]")


def _query_terms(query: str) -> list[tuple[str, tuple[str, ...]]]:
    """Extract scoring terms from a query as (term, thai_4grams) pairs.

    Thai has no word boundaries, so a whole Thai phrase arrives as one long
    token that rarely appears verbatim in content — long Thai tokens also get
    4-gram fragments so partial word overlap still scores.
    """
    terms: list[tuple[str, tuple[str, ...]]] = []
    for t in _TERM_RE.findall(query.lower()):
        if len(t) < 2:
            continue
        grams: tuple[str, ...] = ()
        if len(t) > 4 and _THAI_RE.match(t):
            grams = tuple({t[i : i + 4] for i in range(len(t) - 3)})
        terms.append((t, grams))
    return terms


def _split_chunks(raw: str, size: int = 1000) -> list[str]:
    """Merge non-empty lines into ~size-char chunks, preserving order."""
    chunks: list[str] = []
    cur = ""
    for b in raw.split("\n"):
        b = b.strip()
        if not b:
            continue
        if cur and len(cur) + len(b) + 1 > size:
            chunks.append(cur)
            cur = b
        else:
            cur = f"{cur}\n{b}" if cur else b
    if cur:
        chunks.append(cur)
    return chunks


_WB_PATTERNS: dict[str, re.Pattern] = {}


def _term_count(chunk_lower: str, t: str) -> int:
    """Occurrences of term t. ASCII terms count whole words only — a raw
    substring count lets 2-char terms ride inside unrelated words ("us" in
    "because", "ai" in "said") and outscore genuinely relevant chunks. Thai has
    no word boundaries, so Thai terms keep the substring count."""
    if _THAI_RE.match(t):
        return chunk_lower.count(t)
    pat = _WB_PATTERNS.get(t)
    if pat is None:
        pat = re.compile(rf"\b{re.escape(t)}\b")
        _WB_PATTERNS[t] = pat
    return len(pat.findall(chunk_lower))


def _score_chunk(chunk_lower: str, terms: list[tuple[str, tuple[str, ...]]]) -> int:
    """Full-term hits weigh 3 (capped), Thai 4-gram partial hits weigh 1."""
    score = 0
    for t, grams in terms:
        c = _term_count(chunk_lower, t)
        if c:
            score += 3 * min(c, 3)
        elif grams:
            score += sum(1 for g in grams if g in chunk_lower)
    return score


def _select_relevant(raw: str, user_query: str | None, cap: int = RAW_CAP) -> str:
    """Pick the most query-relevant ≤cap chars instead of blindly taking raw[:cap].

    The answer to the user's question often sits past the cap (e.g. a numbers
    section at the end of a long article) — a blind prefix cut means the LLM
    never sees it. Chunks are scored by query-term overlap; highest scorers are
    kept, remaining budget is filled with leading chunks in document order, and
    the lede chunk is always kept (it anchors title/date context).
    Falls back to raw[:cap] when there is no query or nothing matches.
    """
    if len(raw) <= cap:
        return raw
    terms = _query_terms((user_query or "").strip())
    if not terms:
        return raw[:cap]
    chunks = _split_chunks(raw)
    if len(chunks) <= 1:
        return raw[:cap]
    scores = [_score_chunk(c.lower(), terms) for c in chunks]
    if not any(scores):
        return raw[:cap]

    selected = {0}
    budget = cap - min(len(chunks[0]), cap)
    # relevance first…
    for i in sorted(range(1, len(chunks)), key=lambda i: (-scores[i], i)):
        if scores[i] <= 0 or budget <= 0:
            break
        need = len(chunks[i]) + 8  # join + gap-marker overhead
        if need <= budget:
            selected.add(i)
            budget -= need
    # …then fill what's left with leading chunks so the LLM keeps context
    for i in range(1, len(chunks)):
        if budget <= 0:
            break
        if i in selected:
            continue
        need = len(chunks[i]) + 8
        if need <= budget:
            selected.add(i)
            budget -= need

    order = sorted(selected)
    parts: list[str] = []
    prev = None
    for i in order:
        if prev is not None and i != prev + 1:
            parts.append("[...]")
        parts.append(chunks[i])
        prev = i
    _progress(f"query-aware selection: kept {len(selected)}/{len(chunks)} chunks")
    return "\n".join(parts)[:cap]


_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _unverified_numbers(summary: str, source: str, query: str = "") -> list[str]:
    """Numbers the summary states that appear nowhere in the source content (nor
    in the query) — likely mis-transcribed or invented by the LLM. Comparison is
    numeric after comma-stripping so "1,234" verifies "1234" and "120" verifies
    "120.0". Single-digit integers are skipped (list labels, ordinal noise).
    Returns the offending tokens in the summary's own spelling (≤8)."""

    def _vals(s: str) -> set[float]:
        out: set[float] = set()
        for tok in _NUM_RE.findall(s):
            try:
                out.add(float(tok.replace(",", "")))
            except ValueError:
                pass
        return out

    today = datetime.date.today()
    # The prompt feeds the LLM today's date, so the summary legitimately knows
    # the current year (CE and Thai BE) without it appearing in the source.
    known = (_vals(source) | _vals(query) | _vals(today.isoformat())
             | {float(today.year + 543)})
    flagged: list[str] = []
    seen: set[float] = set()
    for tok in _NUM_RE.findall(summary):
        try:
            v = float(tok.replace(",", ""))
        except ValueError:
            continue
        if v < 10 and v == int(v):
            continue
        if v in known or v in seen:
            continue
        seen.add(v)
        flagged.append(tok)
    return flagged[:8]


def _number_flag(summary: str, source: str, query: str = "") -> str | None:
    """⚠ caution line for _with_note, or None when every number checks out."""
    odd = _unverified_numbers(summary, source, query)
    if not odd:
        return None
    return "⚠ ตัวเลขในสรุปที่ไม่พบในต้นฉบับ (อาจคลาดเคลื่อน): " + ", ".join(odd)


_BOUNDARIES = ("\n", "。", ". ", "! ", "? ", " ")


def _hard_truncate(s: str, n: int) -> str:
    """Truncate to ≤n chars, preferring a sentence/line boundary over a mid-sentence cut."""
    if len(s) <= n:
        return s
    if n <= 3:  # the n-3 slice below would go negative and keep almost everything
        return s[:n]
    cut = s[: n - 3]
    best = max(cut.rfind(b) for b in _BOUNDARIES)
    if best > int(n * 0.6):  # boundary close enough to the limit — don't lose too much
        cut = cut[:best]
    return cut.rstrip() + "..."


def _fallback_prefix(raw: str, n: int, user_query: str | None = None) -> str:
    """Raw fallback when the LLM fails: drop short nav/boilerplate lines, then —
    when a query is given — keep the query-relevant lines (document order)
    instead of a blind prefix, so the answer isn't lost past the n-char cut."""
    lines = [ln for ln in raw.splitlines() if len(ln.strip()) >= 30]
    if sum(len(ln) for ln in lines) < 100:  # filter removed too much — use plain raw
        return _hard_truncate(raw, n)
    terms = _query_terms((user_query or "").strip())
    if terms:
        relevant = [ln for ln in lines if _score_chunk(ln.lower(), terms) > 0]
        if relevant:
            picked: list[str] = []
            used = 0
            for ln in relevant:
                picked.append(ln)
                used += len(ln) + 1
                if used >= n:
                    break
            lines = picked
    return _hard_truncate("\n".join(lines), n)


def _with_note(body: str, note: str | None, max_chars: int, flag: str | None = None) -> str:
    """Prepend a deterministic staleness note and append an optional ⚠ flag,
    reserving budget for both so the combined result still respects max_chars."""
    budget = max_chars
    if note:
        budget -= len(note) + 2
    if flag:
        budget -= len(flag) + 1
    out = _hard_truncate(body, max(0, budget))
    if note:
        out = note + "\n\n" + out
    if flag:
        out = out + "\n" + flag
    return out


def _summarize_impl(raw: str, url: str, user_query: str | None = None) -> str:
    """Implementation for one summary owner; public concurrency gate is below."""
    raw = (raw or "").strip()
    if not raw:
        return "(เนื้อหาว่าง)"

    today_d = datetime.date.today()
    note = staleness_note(raw, url, today_d)

    # Short content → use raw as summary, save an LLM call.
    # SUMMARY_SKIP_LLM_BELOW (default 1500) covers most trafilatura outputs
    if len(raw) <= SUMMARY_SKIP_LLM_BELOW:
        _progress(f"raw ≤{SUMMARY_SKIP_LLM_BELOW} chars — skip LLM, use raw as summary")
        return _with_note(raw, note, SUMMARY_MAX_CHARS)

    q = (user_query or "").strip()
    key = _cache_key(url, q, raw)
    cached = _cache_get(key)
    if cached is not None:
        _progress("summary cache hit — reuse, skip LLM")
        return cached

    full_raw = raw  # keep pre-selection content — number fidelity is checked
    raw = _select_relevant(raw, q)  # against the whole source, not the excerpt
    today = today_d.strftime("%Y-%m-%d")
    if q:
        prompt = (
            "You summarize web content. Reply in Thai ONLY.\n"
            f"Today's date: {today}\n"
            f"User question: {q}\n"
            f"URL: {url}\n\n"
            "Task: summarize the content below, prioritizing information that answers the user question.\n"
            f"- Max {SUMMARY_MAX_CHARS} characters.\n"
            "- Keep concrete facts and numbers; include the data's date (ข้อมูล ณ ...) whenever the content states one.\n"
            "- If the content shows a market/trading status label (e.g. 'Closed', 'Open', 'ปิดตลาด') paired with a date or time,"
            " keep that label verbatim — never drop it as boilerplate. Do NOT judge or flag it as stale yourself"
            " (that comparison is done separately, in code) — just report the label and date exactly as written.\n"
            "- Keep entity names, technical terms, and units exactly as written.\n"
            "- NEVER invent information not present in the content — this includes dates: never write a date"
            " that is not literally present in the content below.\n"
            "- Output the summary only — no preamble, no closing remarks.\n\n"
            "Content:\n"
            f"{raw}"
        )
    else:
        prompt = (
            "You summarize web content. Reply in Thai ONLY.\n"
            f"Today's date: {today}\n"
            f"URL: {url}\n\n"
            f"Task: summarize the content below. Max {SUMMARY_MAX_CHARS} characters.\n"
            "- Cover the main points, concrete facts, and key numbers; include the data's date (ข้อมูล ณ ...) whenever the content states one.\n"
            "- If the content shows a market/trading status label (e.g. 'Closed', 'Open', 'ปิดตลาด') paired with a date or time,"
            " keep that label verbatim — never drop it as boilerplate. Do NOT judge or flag it as stale yourself"
            " (that comparison is done separately, in code) — just report the label and date exactly as written.\n"
            "- Keep entity names, technical terms, and units exactly as written.\n"
            "- NEVER invent information not present in the content — this includes dates: never write a date"
            " that is not literally present in the content below.\n"
            "- Output the summary only — no preamble, no closing remarks.\n\n"
            "Content:\n"
            f"{raw}"
        )

    # Disable Qwen3 thinking mode for summarization — it's a straightforward
    # extraction task, no reasoning needed. With thinking off, latency drops
    # from ~30s → ~3s per call (10× faster).
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
            result = _with_note(text, note, SUMMARY_MAX_CHARS, _number_flag(text, full_raw, q))
            _cache_put(key, result)
            return result
        except Exception as e:
            last_error = str(e)
            if attempt == 1:
                _progress(f"attempt 1 failed: {e} — retrying")

    log.warning(f"[_summarize] LLM failed for {url[:60]} after 2 attempts: {last_error}")
    _progress(f"summary failed after 2 attempts — using raw prefix fallback")
    return _with_note(_fallback_prefix(raw, SUMMARY_MAX_CHARS, q), note, SUMMARY_MAX_CHARS)


def summarize(raw: str, url: str, user_query: str | None = None) -> str:
    """Summarize web content, coalescing identical concurrent LLM misses."""
    normalized_raw = (raw or "").strip()
    if not normalized_raw or len(normalized_raw) <= SUMMARY_SKIP_LLM_BELOW:
        return _summarize_impl(normalized_raw, url, user_query)

    q = (user_query or "").strip()
    key = _cache_key(url, q, normalized_raw)
    cached, is_owner, flight = _cache_claim(key)
    if not is_owner:
        if cached is None:
            assert flight is not None
            flight.event.wait()
            cached = flight.result
            if cached is None:
                # The owner raised unexpectedly before producing even a
                # fallback. Retry as a new flight instead of returning empty.
                return summarize(normalized_raw, url, user_query)
            _progress("summary single-flight hit — reuse owner result")
        else:
            _progress("summary cache hit — reuse, skip LLM")
        return cached

    assert flight is not None
    try:
        # Recheck inside the implementation so a concurrent batch summarizer
        # that populated this key between claim and execution is also reused.
        result = _summarize_impl(normalized_raw, url, user_query)
    except BaseException:
        _cache_finish(key, flight, None)
        raise
    _cache_finish(key, flight, result)
    return result


def _parse_batch_response(text: str, n: int) -> dict[int, str] | None:
    """Split a "===SOURCE i===\\n<summary>" formatted response into {i: summary}.

    Returns None if any of the n expected sections is missing or empty —
    the caller falls back to per-URL summarize() in that case.
    """
    pieces = _SOURCE_MARKER_RE.split(text)
    # split() on a 1-group pattern → [pre, idx1, body1, idx2, body2, ...]
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


def _batch_infer(
    items: list[tuple[str, str]], q: str, today_d: datetime.date
) -> dict[str, str] | None:
    """One batched LLM call over `items` ([(url, raw)]). Performs no cache
    writes — ``summarize_batch`` owns publishing, so the cache and the
    single-flight registry are always updated together."""
    notes = {url: staleness_note(raw, url, today_d) for url, raw in items}

    today = today_d.strftime("%Y-%m-%d")
    lines = ["You summarize multiple web sources at once. Reply in Thai ONLY."]
    lines.append(f"Today's date: {today}")
    if q:
        lines.append(f"User question: {q}")
    lines.append("")
    lines.append(
        "Task: summarize each source below SEPARATELY"
        + (", prioritizing information that answers the user question." if q else ".")
    )
    lines.append(f"- Max {SUMMARY_MAX_CHARS} characters per source.")
    lines.append("- Keep concrete facts and numbers; include the data's date (ข้อมูล ณ ...) whenever a source states one.")
    lines.append(
        "- If a source shows a market/trading status label (e.g. 'Closed', 'Open', 'ปิดตลาด') paired with a date or time,"
        " keep that label verbatim — never drop it as boilerplate. Do NOT judge or flag it as stale yourself"
        " (that comparison is done separately, in code) — just report the label and date exactly as written."
    )
    lines.append("- Keep entity names, technical terms, and units exactly as written.")
    lines.append("- NEVER invent information not present in the content — this includes dates: never write a date"
                  " that is not literally present in that source's content.")
    lines.append(f"- Answer for ALL {len(items)} sources using EXACTLY this format, no other text outside it:")
    lines.append("")
    for i in range(1, len(items) + 1):
        lines.append(f"===SOURCE {i}===")
        lines.append(f"<สรุปของแหล่งที่ {i}>")
    lines.append("")
    lines.append("เนื้อหาแต่ละแหล่ง:")
    lines.append("")
    # Raw bodies can be 20k each (web_search caches full pages) — split the
    # single-call RAW_CAP budget across sources so the batch prompt stays flat.
    per_cap = max(2500, RAW_CAP // len(items))
    for i, (url, raw) in enumerate(items, 1):
        lines.append(f"[SOURCE {i}: {url}]")
        lines.append(_select_relevant(raw, q, cap=per_cap))
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
            out: dict[str, str] = {}
            for i, (url, full) in enumerate(items, 1):
                out[url] = _with_note(
                    parsed[i], notes.get(url), SUMMARY_MAX_CHARS, _number_flag(parsed[i], full, q)
                )
            return out
        except Exception as e:
            last_error = str(e)
            if attempt == 1:
                _progress(f"batch attempt 1 failed: {e} — retrying")

    log.warning(f"[_summarize] batch summarize failed after 2 attempts: {last_error}")
    _progress("batch summary failed — falling back to per-URL summarize")
    return None


def summarize_batch(items: list[tuple[str, str]], user_query: str | None = None) -> dict[str, str] | None:
    """Summarize multiple raw bodies in a single LLM call.

    items: [(url, raw_with_title), ...] — only items that need LLM summarization
    (i.e. would not hit the SUMMARY_SKIP_LLM_BELOW short-circuit in summarize()).
    Returns {url: summary} on success, or None to signal the caller should fall
    back to per-URL summarize() (e.g. <2 items, or the model didn't return a
    parseable per-source response).

    Batch and single share ONE per-item cache and ONE single-flight registry:
    already-summarized sources are reused, concurrent duplicates coalesce onto
    whichever caller claimed the key first (batch or single), and only genuinely
    unclaimed misses reach the model.
    """
    # Dedupe by URL first: callers key their results off the returned dict, and
    # one URL must resolve to exactly one cache key. Two entries for the same
    # URL with different raw bodies would otherwise publish a summary computed
    # from one body under the other body's key.
    seen_urls: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for url, raw in items:
        raw = (raw or "").strip()
        if not raw or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append((url, raw))
    items = deduped
    if len(items) < 2:
        return None

    q = (user_query or "").strip()
    today_d = datetime.date.today()
    keys = {url: _cache_key(url, q, raw) for url, raw in items}
    cached, owned, followed = _cache_claim_many([keys[url] for url, _ in items])

    out: dict[str, str] = {}
    miss_items: list[tuple[str, str]] = []
    claimed: set[tuple[str, str, str, str]] = set()
    for url, raw in items:
        key = keys[url]
        if key in cached:
            out[url] = cached[key]
        elif key in owned and key not in claimed:
            claimed.add(key)
            miss_items.append((url, raw))
    owned_url = {keys[url]: url for url, _ in miss_items}
    if cached:
        _progress(f"batch summary cache hit for {len(cached)}/{len(items)} sources")

    computed: dict[str, str] = {}
    try:
        if miss_items:
            inferred = _batch_infer(miss_items, q, today_d)
            if inferred is None:
                return None  # caller falls back per URL; finally releases flights
            computed = inferred
            for url, summary in computed.items():
                _cache_put(keys[url], summary)  # later single reads reuse it
                out[url] = summary
    finally:
        # Success, parse failure, Exception and BaseException all land here.
        # Every owned flight must be published or released, or a follower waits
        # forever and _SUMMARY_INFLIGHT leaks.
        for key, flight in owned.items():
            _cache_finish(key, flight, computed.get(owned_url.get(key, "")))

    # Only now wait on flights owned by someone else. Publishing first is what
    # makes crossing batches (A=[u1,u2], B=[u2,u3]) impossible to deadlock.
    for url, _raw in items:
        flight = followed.get(keys[url])
        if flight is None:
            continue
        flight.event.wait()
        if flight.result is None:
            # Owner died without producing even a fallback — return None rather
            # than a partial dict; the caller's per-URL path reuses whatever we
            # already published.
            return None
        out[url] = flight.result
    return out
