"""graph.py — single-node orchestration

START → react (main agent คุมเอง: ถ้าซับซ้อน → เรียก create_plan tool → ทำ steps → รวมคำตอบ) → END

ไม่มี planner_node / execute_node / synthesize_node แยกอีกต่อไป — main agent
เห็น full history + ทุก tool result ใน loop เดียว แล้วตัดสินใจเอง
"""
from __future__ import annotations
import logging
import re
from typing import Annotated, TypedDict

import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, RemoveMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from react import build_react_agent, ctx_stats, get_system_prompt
from llm import build_llm
from config import RECURSION_LIMIT, CONTEXT_MAX_CHARS
from planner import plan as _plan
from tools.create_plan import format_plan as _format_plan
from tools._progress import phase as _phase, emit_plan as _emit_plan

log = logging.getLogger(__name__)

# {query} = current turn's user message — anchors the retry so the model answers
# THIS question, not whatever topic dominates earlier turns' tool/search results
# (a turn with no tool calls has nothing to "cite", so without this anchor the model
# drifts to summarizing a previous turn's research instead).
_SYNTH_RETRY_PROMPT = (
    "กรุณาเขียนคำตอบสุดท้ายภาษาไทยให้ผู้ใช้โดยตรง สำหรับคำถามล่าสุดนี้เท่านั้น:\n"
    "\"{query}\"\n"
    "หากมีผลค้นหาที่เกี่ยวข้องกับคำถามนี้ ให้ระบุข้อเท็จจริงและตัวเลขจากผลนั้น "
    "อย่าพูดว่า 'ดูข้างบน' และอย่าตอบคำถามอื่นที่ไม่ใช่คำถามนี้"
)

# ── Deterministic search/research intercept (mirrors ENDEAVOR_VISION _inject_intent_hint) ──
# Explicit research/search VERBS only — NOT bare "หา" (avoids matching "หาค่าเฉลี่ย" = compute).
# Negative lookahead excludes retrospective phrasing ("...ค้นหาที่ผ่านมา" = "past
# searches", a meta-question about history, not a new search request) so it doesn't
# wrongly inject _SEARCH_DIRECTIVE for questions like "scratch_write มีประโยชน์ไหม
# ในการค้นหาที่ผ่านมา".
_SEARCH_VERB_RE = re.compile(
    r"(?:ทำวิจัย|วิจัย|ค้นหา|ค้นคว้า|สืบค้น|หาข้อมูล|หาข่าว|เช็คข้อมูล|ค้นจาก|หาจาก|search)"
    r"(?!\s*(?:ที่ผ่านมา|ก่อนหน้า))"
)

_SEARCH_DIRECTIVE = (
    "[SEARCH DIRECTIVE — deterministic intercept]\n"
    "ผู้ใช้สั่งให้ค้นหา/วิจัยอย่างชัดเจนใน turn นี้ → turn นี้ต้องดึงข้อมูลจากภายนอก "
    "ห้ามตอบจาก training เพียงอย่างเดียว แม้จะคิดว่ารู้คำตอบ หรือเพิ่งคุยหัวข้อนี้ไปแล้ว "
    "(ถ้าประโยคนี้ไม่ระบุหัวข้อ → ใช้หัวข้อล่าสุดในบทสนทนา; ไม่มีหัวข้อเลย → ถาม 1 คำถามว่าจะค้นเรื่องอะไร)\n"
    "เรื่องเดียว → web_search ตรงๆ\n"
    "ข้อยกเว้น (escape): directive นี้มาจาก keyword-match อัตโนมัติ อาจ inject ผิดได้ "
    "ถ้าข้อความผู้ใช้จริงๆ ไม่ใช่คำสั่งให้ค้นหา/วิจัยใหม่ (เช่น เป็นคำถาม meta เกี่ยวกับ tool "
    "หรือการสนทนา/การค้นหาที่ผ่านมา) → ถือว่า directive นี้ใช้ไม่ได้กับ turn นี้ "
    "ตอบคำถามจริงของผู้ใช้ตามปกติ ไม่ต้อง search"
)


def _force_plan_or_directive(msgs: list) -> tuple[list, list]:
    """Deterministic research intercept — single path, no nudge-vs-nudge fallback.

    When the latest user turn is an explicit research/search command, the planner
    (the authority on simple/complex) decides:
      • complex → SEED create_plan in code: append AIMessage(tool_call) + ToolMessage(plan)
        so create_plan firing is code-GUARANTEED, not left to the non-deterministic model
        (a prompt nudge can be ignored on a bad roll; a seeded tool result cannot).
      • simple  → soft directive nudge (single-topic search → web_search direct).

    Returns (msgs_for_agent, seeded). `seeded` is the [AIMessage, ToolMessage] pair that
    MUST be persisted by the caller (react_node returns it) so the plan call appears in
    cross-turn history and the slice `out[len(trimmed):]` does not drop it.
    `seeded` is [] for the simple/no-match cases (directive patch is ephemeral, not persisted).
    """
    if not msgs:
        return msgs, []
    idx = next((i for i in range(len(msgs) - 1, -1, -1)
                if isinstance(msgs[i], HumanMessage)), None)
    if idx is None:
        return msgs, []
    content = str(msgs[idx].content)
    if content.startswith("[SEARCH DIRECTIVE") or not _SEARCH_VERB_RE.search(content):
        return msgs, []

    # research-verb detected → planner classifies simple vs complex
    try:
        result = _plan(content)
    except Exception as e:
        log.warning("[force-plan] planner failed (%s) → soft directive", e)
        result = {"mode": "simple"}

    if result.get("mode") == "complex" and result.get("plan"):
        plan_text = _format_plan(result["plan"])
        tcid = "forced_" + uuid.uuid4().hex[:8]
        ai = AIMessage(content="[planning]",
                       tool_calls=[{"name": "create_plan", "args": {"query": content}, "id": tcid}])
        tm = ToolMessage(content=plan_text, tool_call_id=tcid, name="create_plan")
        log.info("[force-plan] create_plan seeded (%d steps)", len(result["plan"]))
        _emit_plan(plan_text)  # fire before _REACT.invoke so plan appears before web tools in UI
        return list(msgs) + [ai, tm], [ai, tm]

    # simple research query → soft directive nudge (ephemeral patch, not persisted)
    patched = list(msgs)
    patched[idx] = HumanMessage(content=_SEARCH_DIRECTIVE + "\n\n" + content)
    log.info("[search-intercept] soft directive injected (simple, chars=%d)", len(content))
    return patched, []


class V2State(TypedDict):
    messages: Annotated[list, add_messages]  # cross-turn (outer MemorySaver)


_REACT = None  # set by build_graph
_SYNTH_LLM = None   # cached synthesis LLM (no-thinking, max_tokens=2048)
_COMPACT_LLM = None  # cached compact-summarizer LLM (no-thinking, max_tokens=250)

_COMPACT_TRIGGER = 1.00   # compact เมื่อ context เต็ม ≥ 100%
_COMPACT_RESET   = 0.70   # cooldown หายหลัง context ลงต่ำกว่า 70%
_COMPACT_MIN_MSGS = 4     # ต้องมีอย่างน้อย 4 messages ถึงจะ compact
_COMPACT_STRIP_MAX_CHARS = 5000  # stripped H+A pairs ≤ this → skip LLM summarization


def _find_turn_cut(msgs: list) -> int:
    """Return index to cut msgs[:index] for summarization, ~50% of total chars.

    Primary: cut just before a HumanMessage (turn boundary).
    Fallback: cut at message boundary if no turn cut is possible (e.g. single huge turn).
    Returns 0 if no suitable cut found.
    """
    if len(msgs) < 2:
        return 0

    total = sum(len(str(m.content)) for m in msgs)
    target = total // 2

    # Collect (index, cumulative_chars_before) for each valid turn-boundary cut
    cuts = []
    cumulative = 0
    for i, m in enumerate(msgs):
        if isinstance(m, HumanMessage) and i > 0:
            cuts.append((i, cumulative))
        cumulative += len(str(m.content))

    if cuts:
        # Pick the turn boundary whose chars_before is closest to target
        best_idx, _ = min(cuts, key=lambda c: abs(c[1] - target))
        if best_idx > 0:
            return best_idx

    # Fallback: cut at message boundary closest to 50% (single huge turn)
    cumulative = 0
    for i, m in enumerate(msgs[:-1]):   # keep at least 1 message
        cumulative += len(str(m.content))
        if cumulative >= target:
            return i + 1

    return 0


def _get_compact_llm():
    global _COMPACT_LLM
    if _COMPACT_LLM is None:
        _COMPACT_LLM = build_llm(
            temperature=0.1,
            max_tokens=250,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    return _COMPACT_LLM


def _get_synth_llm():
    global _SYNTH_LLM
    if _SYNTH_LLM is None:
        _SYNTH_LLM = build_llm(
            temperature=0.1,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    return _SYNTH_LLM


def _strip_tool_messages(msgs: list) -> list:
    """Keep only HumanMessage + AIMessage-without-tool_calls with non-empty content.

    Drops ToolMessages and AIMessages that only carry tool_calls (no final text).
    Used by _compact_core to shrink msgs_to_sum before deciding whether to LLM-summarize.
    """
    return [
        m for m in msgs
        if isinstance(m, HumanMessage)
        or (
            isinstance(m, AIMessage)
            and not (getattr(m, "tool_calls", None) or [])
            and (m.content or "").strip()
        )
    ]


def _compact_core(msgs_to_sum: list) -> str:
    """Hybrid compaction: strip ToolMessages first, LLM-summarize only when stripped pairs still large.

    Fast path (stripped chars ≤ _COMPACT_STRIP_MAX_CHARS):
      Format H+A pairs directly — no LLM call.
    Heavy path (stripped chars > threshold):
      LLM summarize the stripped pairs (not the originals — avoids feeding tool JSON noise).
    """
    stripped = _strip_tool_messages(msgs_to_sum)
    stripped_chars = sum(len(str(m.content)) for m in stripped)

    if stripped_chars <= _COMPACT_STRIP_MAX_CHARS:
        if not stripped:
            return f"({len(msgs_to_sum)} earlier messages — tool interactions only)"
        lines = []
        for m in stripped:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            content = (m.content or "").strip()[:500]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # Heavy path: LLM summarize stripped H+A pairs
    return _summarize_for_compact(stripped)


def _summarize_for_compact(msgs: list) -> str:
    """LLM call to summarize old messages into a compact English digest (≤1000 chars)."""
    lines = []
    for m in msgs:
        role = type(m).__name__.replace("Message", "")
        content = (m.content or "").strip()
        if content:
            lines.append(f"{role}: {content[:300]}")

    if not lines:
        return f"({len(msgs)} earlier messages — no content)"

    try:
        llm = _get_compact_llm()
        # Reuse the main agent's system prompt (not a dedicated "summarizer" persona) so this
        # call shares the cached [base_prompt] prefix instead of evicting it from the shared
        # mlx_lm.server LRU pool — keeps the segment warm right before the post-compact turn
        # needs it (V2-PF03: a distinct system message here caused full ~19.6k cache-miss reprocess).
        resp = llm.invoke([
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=(
                "TASK (one-off — ignore your usual role for this message): "
                "summarize the following conversation segment in English. "
                "Be concise. Preserve key facts, decisions, and context. "
                "Max 1000 characters total.\n\n" + "\n".join(lines)
            )),
        ])
        text = (resp.content or "").strip()
        return text[:1000] if text else f"({len(msgs)} earlier messages)"
    except Exception as e:
        log.warning(f"[compact] summarize failed: {e}")
        return f"({len(msgs)} earlier messages — summary unavailable)"


def summarize_history(app, config: dict) -> dict:
    """Return stats + LLM-generated topic list for the loaded history thread.

    Returns:
        {
          "total": int,       # total messages in state
          "turns": int,       # human turns (= conversation rounds)
          "topics": str,      # short Thai bullet list of topics, or fallback text
        }
    """
    msgs = list((app.get_state(config).values or {}).get("messages", []))
    total = len(msgs)
    turns = sum(1 for m in msgs if isinstance(m, HumanMessage))

    if total == 0:
        return {"total": 0, "turns": 0, "topics": "(ยังไม่มี history)"}

    # Build condensed text from human messages only (≤60 chars each) for topic extraction
    human_lines = []
    for m in msgs:
        if isinstance(m, HumanMessage):
            text = (m.content or "").strip()[:60]
            if text:
                human_lines.append(f"- {text}")

    if not human_lines:
        return {"total": total, "turns": turns, "topics": "(ไม่สามารถดึงหัวข้อได้)"}

    prompt = (
        "จากรายการคำถาม/คำสั่งของ user ด้านล่าง "
        "สรุปเป็นหัวข้อหลัก 3-5 ข้อ (bullet สั้น ไม่เกิน 10 คำต่อข้อ) ใน**ภาษาไทย**:\n\n"
        + "\n".join(human_lines[:40])  # cap at 40 turns
        + "\n\nตอบเฉพาะ bullet list เท่านั้น ไม่ต้องอธิบายเพิ่ม"
    )
    try:
        llm = _get_compact_llm()
        resp = llm.invoke([HumanMessage(content=prompt)])
        topics = (resp.content or "").strip()[:500]
    except Exception as e:
        log.warning(f"[history] topic summary failed: {e}")
        topics = "(สรุปหัวข้อไม่ได้)"

    return {"total": total, "turns": turns, "topics": topics}


def force_compact(app, config: dict) -> dict:
    """Manual compaction — same logic as auto compact in react_node but bypasses pct gate.

    Returns {"cut": N, "before": chars_before, "after": chars_after}
    or {"error": reason} if compaction was not possible.
    """
    state = app.get_state(config)
    msgs = list(state.values.get("messages", []))

    if len(msgs) < _COMPACT_MIN_MSGS:
        return {"error": f"ต้องมีอย่างน้อย {_COMPACT_MIN_MSGS} messages (ปัจจุบัน {len(msgs)})"}

    cut_idx = _find_turn_cut(msgs)
    if cut_idx <= 0:
        return {"error": "ไม่พบ turn boundary ที่เหมาะสมสำหรับตัด"}

    chars_before = sum(len(str(m.content)) for m in msgs)
    msgs_to_sum  = msgs[:cut_idx]
    msgs_keep    = msgs[cut_idx:]

    summary_text = _compact_core(msgs_to_sum)
    summary_msg  = AIMessage(content=f"[Context summary — {cut_idx} earlier messages]\n{summary_text}")

    # Remove ALL messages, then re-add in correct order: summary → remaining
    # (add_messages always appends — can't insert at position 0 without full replace)
    all_removes  = [RemoveMessage(id=m.id) for m in msgs if getattr(m, "id", None)]
    rebuilt      = [summary_msg] + [m.model_copy(update={"id": str(uuid.uuid4())}) for m in msgs_keep]
    app.update_state(config, {"messages": all_removes + rebuilt})

    state_after  = app.get_state(config)
    msgs_after   = list(state_after.values.get("messages", []))
    chars_after  = sum(len(str(m.content)) for m in msgs_after)

    ctx_stats["cooldown"] = True
    ctx_stats["compact_msg"] = cut_idx

    return {"cut": cut_idx, "before": chars_before, "after": chars_after}


_REWARM_THREAD_ID = "__cache_rewarm__"


def rewarm_after_compact(app, db_conn, seed_msgs: list) -> None:
    """Re-warm the LLM prefix cache after manual /compact (CLI or AGENT_UI).

    The compaction summarizer's call sits between the live conversation's last
    cached entry and the user's next turn, but its prompt diverges (different
    suffix), so by compact-time the [base_prompt + history] entry is already
    evicted from the shared LRU pool (V2-PF03: measured post-compact prefill
    stuck at ~19.7k tokens — full reprocess). Fire a disposable call sharing the
    NEW [base_prompt + summary + kept_msgs] prefix synchronously here — caller
    runs this in a background thread so it doesn't block the user's response.
    """
    def _purge():
        db_conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (_REWARM_THREAD_ID,))
        db_conn.execute("DELETE FROM writes WHERE thread_id = ?", (_REWARM_THREAD_ID,))
        db_conn.commit()
    try:
        _purge()
        rcfg = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": _REWARM_THREAD_ID}}
        seed = [m.model_copy(update={"id": str(uuid.uuid4())}) for m in seed_msgs]
        app.update_state(rcfg, {"messages": seed})
        for _ in app.stream({"messages": [{"role": "user", "content": "hi"}]},
                            config=rcfg, stream_mode="updates"):
            pass
    except Exception:
        pass
    finally:
        try:
            _purge()
        except Exception:
            pass


def _last_ai_content(messages: list) -> str:
    """คืน content ของ AIMessage สุดท้ายที่ไม่มี tool_calls"""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not (getattr(m, "tool_calls", None) or []):
            return (m.content or "").strip()
    return ""


def _last_human_content(messages: list) -> str:
    """คืน content ของ HumanMessage สุดท้าย (current turn's user query)"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""


def react_node(state: V2State, config: RunnableConfig) -> dict:
    """main agent — เห็น full cross-turn history, ตัดสินใจเองว่าจะเรียก tool ไหน

    Context management: trim messages เกิน CONTEXT_MAX_CHARS ก่อนส่งให้ agent
    (MemorySaver state ไม่ถูกตัด — เฉพาะ window ที่ invoke เท่านั้น)

    Retry logic: ถ้า agent ทำ tool ครบแต่ไม่ synthesize (final='') →
    inject synthesis prompt แล้ว call LLM 1 ครั้ง (ไม่ใช่ full react loop ใหม่)
    """
    trimmed = trim_messages(
        state["messages"],
        max_tokens=CONTEXT_MAX_CHARS,
        token_counter=lambda msgs: sum(len(str(m.content)) for m in msgs),
        strategy="last",
        allow_partial=False,
        include_system=True,
    )
    if len(trimmed) < len(state["messages"]):
        log.info(f"[react_node] context trimmed: {len(state['messages'])} → {len(trimmed)} messages")

    current_chars = sum(len(str(m.content)) for m in trimmed)
    sp_chars = sum(len(str(m.content)) for m in trimmed if isinstance(m, SystemMessage))
    _effective_max = max(CONTEXT_MAX_CHARS - sp_chars, 1)
    pct = (current_chars - sp_chars) / _effective_max
    ctx_stats["chars"] = current_chars
    ctx_stats["max_chars"] = CONTEXT_MAX_CHARS
    ctx_stats["compact_msg"] = None

    # Cooldown reset: context ลดต่ำกว่า 70% → อนุญาต compact ครั้งถัดไป
    if ctx_stats["cooldown"] and pct < _COMPACT_RESET:
        ctx_stats["cooldown"] = False

    # Context compaction: ≥90% + ไม่อยู่ใน cooldown + มี messages พอ
    extra_updates: list = []
    if pct >= _COMPACT_TRIGGER and not ctx_stats["cooldown"] and len(state["messages"]) >= _COMPACT_MIN_MSGS:
        ctx_stats["compact_before"] = current_chars
        _phase("⚙️ กำลังบีบอัด context…")
        state_msgs = state["messages"]
        cut_idx = _find_turn_cut(state_msgs)
        if cut_idx > 0:
            msgs_to_sum = state_msgs[:cut_idx]
            log.info(f"[compact] {pct*100:.0f}% — summarizing {cut_idx} messages")

            summary_text = _compact_core(msgs_to_sum)
            summary_msg = AIMessage(content=f"[Context summary — {cut_idx} earlier messages]\n{summary_text}")

            # RemoveMessage only for messages with valid IDs (guard against edge case)
            removes = [RemoveMessage(id=m.id) for m in msgs_to_sum if getattr(m, "id", None)]
            if len(removes) < len(msgs_to_sum):
                log.warning(f"[compact] {len(msgs_to_sum) - len(removes)} messages had no ID — skipped from removal")

            extra_updates = removes + [summary_msg]

            # Rebuild trimmed from compacted state
            compacted = [summary_msg] + list(state_msgs[cut_idx:])
            trimmed = trim_messages(
                compacted,
                max_tokens=CONTEXT_MAX_CHARS,
                token_counter=lambda msgs: sum(len(str(m.content)) for m in msgs),
                strategy="last",
                allow_partial=False,
                include_system=True,
            )
            ctx_stats["chars"] = sum(len(str(m.content)) for m in trimmed)
            ctx_stats["cooldown"] = True
            ctx_stats["compact_msg"] = cut_idx
            log.info(f"[compact] after: {ctx_stats['chars']:,} chars ({ctx_stats['chars']/CONTEXT_MAX_CHARS*100:.0f}%)")

    # Deterministic intercept: when user explicitly commands research, seed create_plan
    # in code (complex) or nudge (simple). `seeded` must be persisted so the plan call
    # survives in cross-turn history and is not dropped by the slice below.
    trimmed, seeded = _force_plan_or_directive(trimmed)

    out = _REACT.invoke(
        {"messages": trimmed},
        config={**config, "recursion_limit": RECURSION_LIMIT},
    )
    new_msgs = out["messages"][len(trimmed):]  # slice by trimmed length (correct after compaction); excludes seeded

    # Update char count after full turn (include response in this turn's tally)
    ctx_stats["chars"] = sum(len(str(m.content)) for m in list(trimmed) + list(new_msgs))

    # ตรวจว่ามี final synthesis หรือไม่
    if not _last_ai_content(new_msgs):
        log.warning("[react_node] final empty — retry synthesis once")
        current_query = _last_human_content(state["messages"])
        retry_msgs = out["messages"] + [HumanMessage(content=_SYNTH_RETRY_PROMPT.format(query=current_query))]
        resp = _get_synth_llm().invoke(retry_msgs)
        synth = (resp.content or "").strip()
        if synth:
            new_msgs = new_msgs + [AIMessage(content=synth)]
            log.info(f"[react_node] retry synthesis OK ({len(synth)} chars)")
        else:
            log.warning("[react_node] retry synthesis also empty")

    # Order: removes+summary (old) → seeded plan call (this turn) → agent output (this turn)
    return {"messages": extra_updates + seeded + new_msgs}


def build_graph(checkpointer=None, memory: str = "", tools=None):
    global _REACT
    _REACT = build_react_agent(checkpointer=None, memory=memory, tools=tools)
    g = StateGraph(V2State)
    g.add_node("react", react_node)
    g.add_edge(START, "react")
    g.add_edge("react", END)
    return g.compile(checkpointer=checkpointer)
