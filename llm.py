# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""llm.py — ChatOpenAI factory ชี้ไปที่ mlx_vlm.server

mlx-vlm exposes Qwen3 thinking controls as top-level OpenAI-compatible request
fields; its reasoning stream is kept separate from the user-facing content.
"""
from __future__ import annotations
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_openai import ChatOpenAI

from config import MLX_BASE_URL, MODEL, API_KEY, TEMPERATURE, MAX_TOKENS, THINKING_BUDGET, REPETITION_PENALTY


_DEFAULT_EXTRA_BODY = {
    "enable_thinking": True,
    "thinking_budget": THINKING_BUDGET,
    "repetition_penalty": REPETITION_PENALTY,  # top-level body field in mlx_vlm
}


def _has_image_blocks(messages: list) -> bool:
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        if any(
            isinstance(block, dict) and block.get("type") in {"image_url", "image"}
            for block in content
        ):
            return True
    return False


def _replace_image_blocks(messages: list, fallback_text: str) -> list:
    """Remove image blocks while preserving the original human question."""
    cleaned: list[Any] = []
    image_indexes: list[int] = []
    human_image_indexes: list[int] = []
    for index, message in enumerate(messages):
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            cleaned.append(message)
            continue
        blocks = [
            block for block in content
            if not (isinstance(block, dict) and block.get("type") in {"image_url", "image"})
        ]
        if len(blocks) != len(content):
            image_indexes.append(index)
            if getattr(message, "type", "") == "human":
                human_image_indexes.append(index)
        cleaned.append((message, blocks))

    if not image_indexes:
        return list(messages)
    target = (human_image_indexes or image_indexes)[-1]
    rebuilt: list[Any] = []
    for index, item in enumerate(cleaned):
        if not isinstance(item, tuple):
            rebuilt.append(item)
            continue
        message, blocks = item
        if index == target:
            blocks = list(blocks) + [{"type": "text", "text": fallback_text}]
        try:
            rebuilt.append(message.model_copy(update={"content": blocks}))
        except AttributeError:
            rebuilt.append(type(message)(content=blocks))
    return rebuilt


def _text_only_fallback_text() -> str:
    """Collect OCR for all active read-image originals in stable source order."""
    sections: list[str] = [
        "[TEXT-ONLY IMAGE FALLBACK]",
        "The configured model rejected image input. Use the full OCR below; "
        "do not invent visual facts that OCR does not contain.",
    ]
    try:
        from tools.read_image import active_turn_image_sources, full_ocr_for_text_only

        sources = active_turn_image_sources()
    except Exception:
        sources = []
        full_ocr_for_text_only = None
    if not sources or full_ocr_for_text_only is None:
        sections.append(
            "[no retained image source was available for OCR — this text-only model "
            "cannot understand the rejected image pixels]"
        )
        return "\n".join(sections)

    for index, (source, path) in enumerate(sources, 1):
        try:
            ocr = full_ocr_for_text_only(path, source)
        except Exception as error:
            ocr = (
                f"[OCR unavailable for image {index}: {type(error).__name__}; "
                "do not infer its visual contents]"
            )
        sections.append(f"[IMAGE {index} — source: {str(source)[:160]}]\n{ocr}")
    return "\n\n".join(sections)


def _invalidate_transient_images() -> None:
    try:
        from tools.read_image import invalidate_published_images

        invalidate_published_images()
    except Exception:
        pass
    try:
        from tools.computer_use import invalidate_computer_images

        invalidate_computer_images()
    except Exception:
        pass


def _recover_without_images(messages: list) -> list:
    fallback_text = _text_only_fallback_text()
    retry_messages = _replace_image_blocks(messages, fallback_text)
    _invalidate_transient_images()
    return retry_messages


class VisionFallbackChatOpenAI(ChatOpenAI):
    """ChatOpenAI with one same-call OCR recovery for rejected image input."""

    def _capability_args(self) -> tuple[str | None, str | None]:
        return (
            getattr(self, "openai_api_base", None),
            getattr(self, "model_name", None),
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        has_images = _has_image_blocks(messages)
        try:
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as error:
            from tools._vision_capability import is_unsupported_image_error, mark_text_only

            if not has_images or not is_unsupported_image_error(error):
                raise
            mark_text_only(*self._capability_args())
            return super()._generate(
                _recover_without_images(list(messages)),
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        if has_images:
            from tools._vision_capability import mark_vision

            mark_vision(*self._capability_args())
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        has_images = _has_image_blocks(messages)
        try:
            result = await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except Exception as error:
            from tools._vision_capability import is_unsupported_image_error, mark_text_only

            if not has_images or not is_unsupported_image_error(error):
                raise
            mark_text_only(*self._capability_args())
            return await super()._agenerate(
                _recover_without_images(list(messages)),
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        if has_images:
            from tools._vision_capability import mark_vision

            mark_vision(*self._capability_args())
        return result

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        messages = args[0] if args else kwargs.get("messages", [])
        has_images = _has_image_blocks(messages)
        yielded = False
        try:
            for chunk in super()._stream(*args, **kwargs):
                yielded = True
                yield chunk
        except Exception as error:
            from tools._vision_capability import is_unsupported_image_error, mark_text_only

            if yielded or not has_images or not is_unsupported_image_error(error):
                raise
            mark_text_only(*self._capability_args())
            retry_args = (_recover_without_images(list(messages)), *args[1:])
            for chunk in super()._stream(*retry_args, **kwargs):
                yield chunk
            return
        if has_images:
            from tools._vision_capability import mark_vision

            mark_vision(*self._capability_args())

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        messages = args[0] if args else kwargs.get("messages", [])
        has_images = _has_image_blocks(messages)
        yielded = False
        try:
            async for chunk in super()._astream(*args, **kwargs):
                yielded = True
                yield chunk
        except Exception as error:
            from tools._vision_capability import is_unsupported_image_error, mark_text_only

            if yielded or not has_images or not is_unsupported_image_error(error):
                raise
            mark_text_only(*self._capability_args())
            retry_args = (_recover_without_images(list(messages)), *args[1:])
            async for chunk in super()._astream(*retry_args, **kwargs):
                yield chunk
            return
        if has_images:
            from tools._vision_capability import mark_vision

            mark_vision(*self._capability_args())


def build_llm(**overrides) -> ChatOpenAI:
    """สร้าง ChatOpenAI client สำหรับ mlx_vlm.server (OpenAI-compatible).

    overrides: override param ใดก็ได้ (เช่น temperature, extra_body สำหรับ thinking)
    extra_body ถูก merge กับ _DEFAULT_EXTRA_BODY เพื่อไม่ให้ caller ที่ override
    แค่ enable_thinking ทำ repetition_penalty หายไป
    """
    vision_fallback = overrides.pop("vision_fallback", True)
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
    if extra_body.get("enable_thinking") is False:
        extra_body.pop("thinking_budget", None)
    params["extra_body"] = extra_body
    client_type = VisionFallbackChatOpenAI if vision_fallback else ChatOpenAI
    return client_type(**params)
