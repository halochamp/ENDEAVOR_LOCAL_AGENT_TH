"""react.py — ReAct agent (create_react_agent) — main agent ที่คุมทุกอย่าง

ใน V2 ใหม่: main agent ตัวเดียวคุมทั้งระบบ
- เห็น query ง่าย → ตอบเลย หรือเรียก tool เดี่ยว
- เห็น query ซับซ้อน → เรียก create_plan ก่อน → ทำ steps ด้วย tool อื่น → รวมคำตอบ
"""
from __future__ import annotations
import datetime
from langgraph.prebuilt import create_react_agent
from config import CONTEXT_MAX_CHARS

from llm import build_llm
from tools import ALL_TOOLS
from system_prompt import SYSTEM


# Main agent system prompt — Thai output, tool guidance, complex-routing, synthesize rules
# System prompt lives in the compiled system_prompt extension.

_BUILT_PROMPT: str = ""

# Shared per-turn context stats — updated by graph.py before each react_node invocation
ctx_stats: dict = {
    "chars": 0,
    "max_chars": CONTEXT_MAX_CHARS,
    "cooldown": False,      # True after compact — wait for pct < 70% before next compact
    "compact_msg": None,    # set to int (n_msgs) when compact fires; endeavor_agent.py reads + clears
    "compact_before": 0,    # chars before compact (for UI display)
}


def get_system_prompt() -> str:
    """Return the system prompt injected in the last build_react_agent call — for logging."""
    return _BUILT_PROMPT


def _make_ctx_note() -> str:
    chars = ctx_stats["chars"]
    max_c = ctx_stats["max_chars"]
    pct = chars / max_c * 100 if max_c > 0 else 0
    if pct >= 90:
        bucket = "NEAR LIMIT: be concise, avoid long code blocks"
    elif pct >= 70:
        bucket = "high"
    elif pct >= 50:
        bucket = "moderate"
    else:
        bucket = "ok"
    today = datetime.date.today().strftime("%Y-%m-%d")
    return f"\n\n[Today: {today}] [Context window: {bucket}]"


def build_react_agent(checkpointer=None, memory: str = "", tools=None, **llm_overrides):
    """สร้าง ReAct agent หลัก. memory = เนื้อหาจาก memory.md (โหลดตอน startup)
    tools: override ALL_TOOLS เช่น กรณี offline mode"""
    from langchain_core.messages import SystemMessage as _SM

    llm = build_llm(**llm_overrides)
    active_tools = tools if tools is not None else ALL_TOOLS
    active_names = {t.name for t in active_tools}

    # If web tools are disabled → notify agent clearly that there is no internet
    offline_note = ""
    if "web_search" not in active_names:
        offline_note = (
            "\n\n!! OFFLINE MODE: no internet access"
            "\n- Never use bash to fetch URLs or curl any external endpoint"
            "\n- If user asks for real-time data (prices, news, web content) → reply directly: 'ไม่มีอินเทอร์เน็ตตอนนี้ ไม่สามารถดึงข้อมูลได้'"
            "\n- Never fabricate or guess data that requires internet access"
        )

    global _BUILT_PROMPT
    system = SYSTEM
    base_prompt = (f"## Your memory about this user:\n{memory}\n\n---\n\n" + system + offline_note) if memory else (system + offline_note)
    _BUILT_PROMPT = base_prompt

    # Dynamic prompt callable — injects ctx_note into last HumanMessage (not system)
    # so system message stays byte-identical every turn → mlx_lm.server system segment cache hit
    def dynamic_prompt(state: dict) -> list:
        from langchain_core.messages import HumanMessage
        msgs = list(state.get("messages", []))
        ctx_note = _make_ctx_note()
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                m = msgs[i]
                if isinstance(m.content, list):
                    new_content = m.content + [{"type": "text", "text": ctx_note}]
                else:
                    new_content = m.content + ctx_note
                msgs[i] = HumanMessage(content=new_content, id=getattr(m, "id", None))
                break
        return [_SM(content=base_prompt)] + msgs

    return create_react_agent(llm, active_tools, prompt=dynamic_prompt, checkpointer=checkpointer)
