# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""llm.py — ChatOpenAI factory ชี้ไปที่ mlx_lm.server

qwen3 reasoning ถูก mlx_lm.server ส่งใน field แยก แล้ว ChatOpenAI drop ทิ้ง
→ content สะอาด ไม่มี <think> (verified 2026-05-27 บน 1.7B) → ไม่ต้อง strip
"""
from __future__ import annotations
from langchain_openai import ChatOpenAI

from config import MLX_BASE_URL, MODEL, API_KEY, TEMPERATURE, MAX_TOKENS, THINKING_BUDGET, REPETITION_PENALTY


_DEFAULT_EXTRA_BODY = {
    "chat_template_kwargs": {
        "enable_thinking": True,
        "thinking_budget": THINKING_BUDGET,
    },
    "repetition_penalty": REPETITION_PENALTY,  # top-level body field, server.py:1180
}


def build_llm(**overrides) -> ChatOpenAI:
    """สร้าง ChatOpenAI client สำหรับ mlx_lm.server (OpenAI-compatible).

    overrides: override param ใดก็ได้ (เช่น temperature, extra_body สำหรับ thinking)
    extra_body ถูก deep-merge กับ _DEFAULT_EXTRA_BODY (รวม chat_template_kwargs)
    เพื่อไม่ให้ caller ที่ override แค่ enable_thinking ทำ repetition_penalty หายไป
    """
    params = dict(
        base_url=MLX_BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        streaming=True,   # enable token-by-token streaming สำหรับ LangGraph "messages" mode
    )
    override_extra_body = overrides.pop("extra_body", {})
    params.update(overrides)

    extra_body = {**_DEFAULT_EXTRA_BODY, **override_extra_body}
    if "chat_template_kwargs" in override_extra_body:
        extra_body["chat_template_kwargs"] = {
            **_DEFAULT_EXTRA_BODY["chat_template_kwargs"],
            **override_extra_body["chat_template_kwargs"],
        }
    params["extra_body"] = extra_body
    return ChatOpenAI(**params)
