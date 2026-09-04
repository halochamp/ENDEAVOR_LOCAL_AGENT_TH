"""react.py — ReAct agent (create_react_agent) — main agent ที่คุมทุกอย่าง

ใน V2 ใหม่: main agent ตัวเดียวคุมทั้งระบบ
- เห็น query ง่าย → ตอบเลย หรือเรียก tool เดี่ยว
- เห็น query ซับซ้อน → เรียก create_plan ก่อน → ทำ steps ด้วย tool อื่น → รวมคำตอบ
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import datetime
import logging
from langgraph.prebuilt import create_react_agent
from config import CONTEXT_MAX_CHARS

from llm import build_llm
from tools import ALL_TOOLS
from system_prompt import SYSTEM

log = logging.getLogger(__name__)


# Main agent system prompt — Thai output, tool guidance, complex-routing, synthesize rules
# System prompt lives in the compiled system_prompt extension.

_BUILT_PROMPT: str = ""
_VISION_PUBLICATION_SUSPENDED = ContextVar(
    "endeavor_vision_publication_suspended", default=False
)


@contextmanager
def suspend_vision_publication():
    """Keep a utility/cache-warm invocation from consuming live tool pixels."""
    token = _VISION_PUBLICATION_SUSPENDED.set(True)
    try:
        yield
    finally:
        _VISION_PUBLICATION_SUSPENDED.reset(token)

# The public build still loads the tracked compiled prompt extension.  Keep this
# small, deterministic overlay in ordinary source so the old OCR/text-only
# contract is explicitly superseded without modifying that opaque artifact.
_DIRECT_VISION_OVERLAY = (
    "\n\n[PUBLIC DIRECT-VISION CONTRACT]\n"
    "- Route explicit image-understanding requests to read_image and explicit screen/application actions to computer.\n"
    "- read_image is progressive direct vision: a source-only call exposes the whole original image to the main VLM first, without an automatic OCR/classifier/table/QR pass.  Only after that same source is visible may you request detail=text, detail=chart, detail=slide, find, or region/zoom assistance.  Do not pass a semantic question or prompt argument; the conversation supplies semantics.\n"
    "- computer is an independent direct-vision/action tool.  It owns its current screenshot and observation lifecycle, while read_image owns its own image lifecycle.  Their queues, guards, and snapshots are never shared; the main model receives whichever pixels each tool publishes.\n"
)

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


def prepare_turn_vision_messages(messages: list) -> list:
    """Attach currently published tool pixels to the last human message.

    Both tools publish only transient data URLs; promoting their pending queues
    here keeps the pixels visible to every remaining ReAct step and to a
    same-turn synthesis retry without persisting image blocks in graph state.
    """
    if _VISION_PUBLICATION_SUSPENDED.get():
        return list(messages)

    read_images: list[str] = []
    computer_images: list[str] = []
    try:
        from tools.read_image import active_turn_images
        read_images = list(active_turn_images())
    except Exception:
        log.debug("read_image image publication unavailable", exc_info=True)
    try:
        from tools.computer_use import active_computer_turn_images
        computer_images = list(active_computer_turn_images())
    except Exception:
        log.debug("computer image publication unavailable", exc_info=True)
    published = [url for url in read_images + computer_images if url]
    if not published:
        return list(messages)

    msgs = list(messages)
    human_index = next(
        (i for i in range(len(msgs) - 1, -1, -1) if _is_human_message(msgs[i])),
        None,
    )
    if human_index is None:
        log.warning("published image data had no HumanMessage recipient")
        return msgs

    from langchain_core.messages import HumanMessage
    message = msgs[human_index]
    existing_urls: set[str | None] = set()
    if isinstance(message.content, list):
        for block in message.content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            image_url = block.get("image_url")
            if isinstance(image_url, dict):
                existing_urls.add(image_url.get("url"))
    image_blocks = [
        {"type": "image_url", "image_url": {"url": url}}
        for url in published
        if url not in existing_urls
    ]
    if not image_blocks:
        return msgs
    if isinstance(message.content, list):
        content = list(message.content) + image_blocks
    else:
        content = [{"type": "text", "text": message.content}] + image_blocks
    msgs[human_index] = HumanMessage(content=content, id=getattr(message, "id", None))
    return msgs


def _is_human_message(message) -> bool:
    from langchain_core.messages import HumanMessage
    return isinstance(message, HumanMessage)


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
    system = SYSTEM + _DIRECT_VISION_OVERLAY
    base_prompt = (f"## Your memory about this user:\n{memory}\n\n---\n\n" + system + offline_note) if memory else (system + offline_note)
    _BUILT_PROMPT = base_prompt

    # Dynamic prompt callable — injects published pixels and ctx_note into the
    # last HumanMessage (not system), so image tool output is visible in the
    # same outer turn while the system message stays byte-identical.
    def dynamic_prompt(state: dict) -> list:
        from langchain_core.messages import HumanMessage
        msgs = prepare_turn_vision_messages(list(state.get("messages", [])))
        ctx_note = _make_ctx_note()
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                m = msgs[i]
                if isinstance(m.content, list):
                    new_content = list(m.content) + [{"type": "text", "text": ctx_note}]
                else:
                    new_content = m.content + ctx_note
                msgs[i] = HumanMessage(content=new_content, id=getattr(m, "id", None))
                break
        return [_SM(content=base_prompt)] + msgs

    return create_react_agent(llm, active_tools, prompt=dynamic_prompt, checkpointer=checkpointer)
