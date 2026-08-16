# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""rag_tool.py — Knowledge base search pipe via a separate local ENDEAVOR_RAG engine
(MiniLM + ChromaDB + BM25 + RRF), forked from ENDEAVOR_LOCAL_AGENT_MAX's tools/rag_tool.py.

This tool file ships WITHOUT the RAG engine and WITHOUT any index/knowledge base —
those are a separate project you set up yourself (see _RAG_DIR below and the
README's RAG section). If the engine isn't present, rag_search returns an
actionable [error] telling you what to clone/build instead of crashing or
silently doing nothing.

Flow: rag_search(query) → raw parent chunks + SOURCES (absolute paths)
      → agent reads full file via existing read_file(path) if needed
"""
from __future__ import annotations
import sys
import os
import re
import datetime
from langchain_core.tools import tool
from tools._progress import progress as _progress

# TH's layout is flatter than MAX's (this repo root IS the "project" level,
# there's no nested ENDEAVOR_LOCAL_AGENT_TH/ subfolder) — one level up from
# tools/ lands on this repo's own parent directory, where a sibling
# ENDEAVOR_RAG_TH engine checkout is expected, mirroring the
# ENDEAVOR_RAG_MAX / ENDEAVOR_RAG_API naming convention used by the other
# forks (see ENDEAVOR_LOCAL_AGENT_MAX/CLAUDE.md's sync table — this path
# constant is intentionally fork-specific, never copy it from another fork).
_RAG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ENDEAVOR_RAG_TH")
)
_RAG_ENTRYPOINT = os.path.join(_RAG_DIR, "rag_retrieve.py")

# Conservative KB staleness threshold. `created` is the DOC AUTHORING date,
# NOT the data-as-of date, so a blanket short window (e.g. web's 1 day) would
# false-alarm every reference doc. Surface age neutrally; flag ⚠️ only past this.
_KB_STALE_DAYS = 365


def _rag_engine_available() -> bool:
    return os.path.isfile(_RAG_ENTRYPOINT)


def _missing_engine_message() -> str:
    return (
        "[error] rag_search: no RAG engine found at "
        f"{_RAG_DIR} — this tool only ships the search wrapper, not the "
        "engine or any knowledge base. Clone/build your own retrieval "
        f"engine into a sibling folder named 'ENDEAVOR_RAG_TH' (i.e. next "
        "to this repo, not inside it) with a top-level rag_retrieve.py "
        "exposing a `rag_retrieve` LangChain tool, then re-run this call."
    )


def _ensure_rag_path() -> None:
    if _RAG_DIR not in sys.path:
        sys.path.insert(0, _RAG_DIR)


def _read_created(path: str) -> tuple[datetime.date | None, str]:
    """Resolve a doc's creation date. Returns (date, provenance).

    provenance: "" = YAML frontmatter `created:` (authoritative, portable);
                "fs" = filesystem birth time (weak fallback — reflects when the
                       file landed on THIS disk, collapses to copy-date after a
                       clone/sync/restore; only used when frontmatter is absent);
                date is None when neither is available (→ caller leaves line bare).
    """
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            head = f.read(2000)
        # Restrict to the YAML frontmatter block (---…---). Measured max offset of
        # `created:` in this corpus is 822B (22 docs exceed 600B, 3 exceed 800B), so
        # a 600/800 window spuriously dropped to the `fs` fallback; 2000B + the block
        # bound covers all and avoids matching a stray `created:` in the body.
        fm = re.match(r"^---\n(.*?)\n---", head, re.S)
        scope = fm.group(1) if fm else head
        m = re.search(r"^created:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", scope, re.M)
        if m:
            return datetime.date.fromisoformat(m.group(1)), ""
    except Exception:
        pass
    # Fallback: filesystem birth time. macOS-only (st_birthtime); on Linux it is
    # absent → AttributeError → caught → (None, "") so the line stays bare.
    try:
        return datetime.date.fromtimestamp(os.stat(path).st_birthtime), "fs"
    except Exception:
        return None, ""


def _annotate_freshness(out: str, today: datetime.date | None = None) -> str:
    """Append `created`/age to each chunk's source line. ⚠️ only past _KB_STALE_DAYS.

    Reads `created` from the source file (authoritative) — not from chunk text or
    chroma metadata — so it stays correct even if frontmatter is later stripped.
    Falls back to filesystem birth time (tagged `fs`) when frontmatter is absent.
    """
    today = today or datetime.date.today()

    def _repl(m: re.Match) -> str:
        path = m.group(1)
        created, prov = _read_created(path)
        if created is None:
            return m.group(0)
        age = (today - created).days
        src = " fs" if prov == "fs" else ""
        note = f" | created: {created.isoformat()}{src} (age {age}d"
        note += " ⚠️)" if age > _KB_STALE_DAYS else ")"
        return f" | file: {path}{note})"

    # Anchor the closing `)` to end-of-line (re.M): source filenames/paths may
    # contain literal `)` (e.g. Thai "(ราคาเป้าหมาย)") — a bare `\)` would stop at
    # the first inner paren and truncate the path. The structural `)` is at EOL.
    return re.sub(r" \| file: (.+?)\)$", _repl, out, flags=re.M)


_TERM_RE = re.compile(r"[A-Za-z0-9%+.-]+|[฀-๿]{2,}")
_STOP_TERMS = {
    "what", "when", "where", "which", "how", "why", "the", "and", "or", "for",
    "is", "are", "a", "an", "to", "of",
    "คือ", "อะไร", "ยังไง", "ไหม", "หรือ", "และ", "ของ", "ที่", "ใน", "ให้",
}


def _query_terms(*parts: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for part in parts:
        for raw in _TERM_RE.findall((part or "").lower()):
            term = raw.strip(".-+")
            if len(term) < 2 or term in _STOP_TERMS or term in seen:
                continue
            seen.add(term)
            terms.append(term)
    return terms[:16]


def _retriever_coverage(out: str) -> tuple[int, int]:
    dense = sum(int(x) for x in re.findall(r"dense_hits:\s*(\d+)", out))
    bm25 = sum(int(x) for x in re.findall(r"bm25_hits:\s*(\d+)", out))
    dense += sum(int(x) for x in re.findall(r"retriever_coverage:\s*dense=(\d+)", out))
    bm25 += sum(int(x) for x in re.findall(r"retriever_coverage:\s*dense=\d+,\s*bm25=(\d+)", out))
    return dense, bm25


def _needs_full_source(text: str) -> bool:
    return bool(re.search(r"ต้นฉบับ|ครบทุก|ทุกประเด็น|ละเอียด|full source|source document|entire", text or "", re.I))


def _diagnostics(out: str, mode: str, terms: list[str], user_need: str) -> str:
    low = out.lower()
    matched = [t for t in terms if t in low]
    missing = [t for t in terms if t not in low]
    dense, bm25 = _retriever_coverage(out)
    chunk_count = out.count("\n[") + (1 if out.startswith("[") else 0)
    if mode in {"files", "source_first"}:
        chunk_count = len(re.findall(r"(?m)^- file:\s+", out))
    possible_mismatch = bool(terms and len(matched) < max(1, min(2, len(terms) // 3 + 1)))
    if mode == "source_first" or _needs_full_source(user_need):
        answerability = "partial_need_full_file"
        suggested = "open the best read_file_hint before final synthesis"
    elif possible_mismatch:
        answerability = "likely_mismatch"
        suggested = "try a narrower rag_search; do not cite weak chunks as [KB]"
    elif dense == 0 or bm25 == 0:
        answerability = "partial"
        suggested = "use top chunks if relevant; consider source_first/read_file if exact source detail is needed"
    else:
        answerability = "direct"
        suggested = "synthesize from top chunks; use RELATED_DOCS only if insufficient"
    return "\n".join([
        "RAG_DIAGNOSTICS:",
        f"  answerability: {answerability}",
        f"  possible_mismatch: {str(possible_mismatch).lower()}",
        f"  matched_query_terms: {', '.join(matched) if matched else '[none]'}",
        f"  missing_query_terms: {', '.join(missing[:8]) if missing else '[none]'}",
        f"  retriever_coverage: dense_hits={dense}, bm25_hits={bm25}, chunks={chunk_count}",
        f"  suggested_next: {suggested}",
        "",
    ])


def _rag_search_impl(sentence_th: str, sentence_en: str, keywords_th: str, keywords_en: str,
                     mode: str = "chunks", tags: str = "", filename_contains: str = "",
                     created_after: str = "", created_before: str = "",
                     source_type: str = "") -> str:
    if not _rag_engine_available():
        return _missing_engine_message()
    try:
        _ensure_rag_path()
        mode = (mode or "chunks").strip().lower()
        variants = [sentence_th, sentence_en, keywords_th, keywords_en]
        seen_q: set[str] = set()
        queries: list[str] = []
        for q in variants:
            q = (q or "").strip()
            if q and q.lower() not in seen_q:
                seen_q.add(q.lower())
                queries.append(q)
        if not queries:
            return "[error] rag_search: all 4 query fields are empty — fill them per the schema"
        _progress(f"searching knowledge base… ({len(queries)} query variants)")
        from rag_retrieve import rag_retrieve as _fn
        result = _fn.invoke({
            "query": queries,
            "mode": mode,
            "tags": tags or "",
            "filename_contains": filename_contains or "",
            "created_after": created_after or "",
            "created_before": created_before or "",
            "source_type": source_type or "",
        })
        if result is None:
            return "[error] rag_search: rag_retrieve returned no result"
        out = result if isinstance(result, str) else str(result)
        if out.startswith("[error]"):
            _progress("RAG error — no chunks returned")
            return out
        out = _annotate_freshness(out)
        chunk_count = out.count("\n[") + (1 if out.startswith("[") else 0)
        if mode in {"files", "source_first"}:
            chunk_count = len(re.findall(r"(?m)^- file:\s+", out))
            _progress(f"found {chunk_count} source file(s)")
        else:
            _progress(f"found {chunk_count} chunk(s)")
        terms = _query_terms(sentence_th, sentence_en, keywords_th, keywords_en)
        user_need = " ".join([sentence_th or "", sentence_en or ""])
        return _diagnostics(out, mode, terms, user_need) + out
    except ModuleNotFoundError:
        # _RAG_ENTRYPOINT existed (checked above) but its own imports failed —
        # an incomplete/half-set-up engine checkout, not "never installed".
        return (
            "[error] rag_search: RAG engine at "
            f"{_RAG_DIR} is present but failed to import — its own "
            "dependencies may not be installed, or its index hasn't been "
            "built yet. Check that project's own setup instructions."
        )
    except Exception as e:
        return f"[error] rag_search: {e}"


@tool
def rag_search(sentence_th: str, sentence_en: str, keywords_th: str, keywords_en: str,
               mode: str = "chunks", tags: str = "", filename_contains: str = "",
               created_after: str = "", created_before: str = "",
               source_type: str = "") -> str:
    """Search the user's own local knowledge base via BM25 + vector search.

    Requires a separate ENDEAVOR_RAG engine set up alongside this repo (see
    README) — if it isn't present, this call returns an actionable [error]
    instead of a KB result; do not retry the same call, just report that to
    the user and fall back to web_search/other tools for the current need.

    The KB may hold BOTH Thai and English documents. Provide ALL 4 variants of
    the SAME question — each variant is searched separately (BM25 + dense) and
    results are RRF-fused, so cross-language variants reach documents the
    original language misses.

    HOW TO FILL EACH FIELD (query quality drives recall):
      sentence_th: the user's core question as ONE natural Thai sentence.
                   Strip greetings/filler (ช่วย/หน่อย/ครับ/ค่ะ) — keep only the info need.
      sentence_en: natural English translation of sentence_th (meaning, not word-by-word).
      keywords_th: 3-6 Thai CONTENT words — domain nouns + entities only;
                   no question words (อะไร/ยังไง/ไหม), no stopwords.
      keywords_en: 3-6 English terms — technical terms + synonyms + related jargon.
    RULES:
      - Keyword fields must NOT be a copy of the sentence — select terms, add synonyms.
      - ONE topic per call. Multi-topic question → one rag_search call per topic.
      - Optional mode:
          mode="chunks" (default): return top chunks + diagnostics.
          mode="files": return matched source files only.
          mode="source_first": return best source paths/read_file_hints first; use when user asks
            "จากเอกสารต้นฉบับ", "ครบทุกประเด็น", exact source detail, or chunks are insufficient.
      - Optional filters narrow the KB before returning: tags (all required), filename_contains,
        created_after/created_before (YYYY-MM-DD), source_type (md/pdf/docx/xlsx/csv).
      - tags uses AND semantics: every listed tag must be present in the document.
      - Prefer source_first when you expect a follow-up read_file call.

    Decision table:
      - Need a direct KB answer -> mode="chunks"
      - Need which source files are relevant -> mode="files"
      - Need source document / complete coverage / likely read_file next -> mode="source_first"
      - Multi-topic comparison -> one rag_search call per topic, never one combined query
      - User already constrained KB slice -> pass filters directly (tags / filename_contains / created_* / source_type)

    Chunks may not be an exact match — caller must verify relevance before synthesizing.

    Output format per chunk:
      RAG_DIAGNOSTICS: answerability / matched terms / retriever coverage / suggested next action
      [N] (source: filename | rrf: 0.xxxx | dense_hits: n | bm25_hits: n | raw_score: n | file: /absolute/path | created: ...)
      meta / heading / read_file_hint
      <chunk text>

    How to use the output:
      - answerability=direct: usually safe to synthesize from top chunks.
      - answerability=partial_need_full_file: open the best read_file_hint before final synthesis.
      - possible_mismatch=true: do not cite weak chunks as [KB]; narrow the query or inspect files.
      - dense_hits + bm25_hits together are stronger than only one retriever hitting.
      - heading shows the matched section path; use it to decide whether the snippet is actually on-topic.

    `created` is the doc's AUTHORING date (age = days since), not its data-as-of
    date — ⚠️ marks docs older than one year; weigh against evergreen vs time-sensitive topics.
    A `fs` tag (`created: … fs`) means the date came from filesystem birth time, not
    frontmatter — a weaker signal (may reflect a copy/sync, not authoring); trust it less.
    To read the full file: use read_file with the absolute path from 'file:' field.
    Prefer the exact read_file_hint when present.

    RELATED_DOCS may follow as structured YAML-like lines; use only when top chunks
    are insufficient or a second source/different angle is needed.
    Follow-up patterns:
      - Need exact source wording / whole section / complete coverage -> mode="source_first", then read_file.
      - Need only likely sources -> mode="files".
      - Empty/thin/mismatched result -> try one better single-topic query.
      - Error says candidates existed but filters removed all results -> relax filters before rewriting the whole query.
    [error] prefix = failure, no results, or the engine isn't set up.
    Do NOT use for general knowledge, math, small talk, coding, or current/web
    information — this only searches whatever local KB the user has built into
    their own ENDEAVOR_RAG engine, and returns nothing for anything outside it.
    """
    return _rag_search_impl(
        sentence_th, sentence_en, keywords_th, keywords_en,
        mode, tags, filename_contains, created_after, created_before, source_type,
    )
