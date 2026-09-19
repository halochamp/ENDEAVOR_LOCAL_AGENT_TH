"""Fail-closed Qwen3.6 tool-call template stabilization for TH-owned APC."""
from __future__ import annotations

import inspect
from typing import Any


_UPSTREAM_CONDITION = (
    "(preserve_thinking is defined and preserve_thinking is true) "
    "or (loop.index0 > ns.last_query_index)"
)
_PATCHED_CONDITION = (
    _UPSTREAM_CONDITION
    + " or (message.tool_calls and message.tool_calls is iterable "
    "and message.tool_calls is not mapping)"
)
_PATCH_MARKER = "_endeavor_th_qwen36_tool_call_apc_patch"


class TemplatePatchError(RuntimeError):
    """The loaded Qwen3.6 template or provider seam is not recognized."""


def patch_template_text(template: str) -> str:
    """Patch exactly one known Qwen3.6 condition, or preserve an idempotent result."""
    if not isinstance(template, str) or not template:
        raise TemplatePatchError("chat template is missing or not a string")
    if _PATCHED_CONDITION in template:
        return template
    count = template.count(_UPSTREAM_CONDITION)
    if count != 1:
        raise TemplatePatchError(
            "expected exactly one Qwen3.6 preserve-thinking condition; "
            f"found {count}"
        )
    return template.replace(_UPSTREAM_CONDITION, _PATCHED_CONDITION, 1)


def _processor_template(processor: Any) -> str | None:
    direct = getattr(processor, "chat_template", None)
    if isinstance(direct, str) and direct:
        return direct
    tokenizer = getattr(processor, "tokenizer", None)
    nested = getattr(tokenizer, "chat_template", None)
    if isinstance(nested, str) and nested:
        return nested
    return None


def _get_chat_template_seam(prompt_utils: Any):
    current = getattr(prompt_utils, "get_chat_template", None)
    if not callable(current):
        return None
    try:
        signature = inspect.signature(current)
    except (TypeError, ValueError):
        return None
    if "processor" not in signature.parameters:
        return None
    if not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or parameter.name == "chat_template"
        for parameter in signature.parameters.values()
    ):
        return None
    return current


def apply_for_model(model: str) -> bool:
    """Install the Qwen3.6-only patch; return false for an unsafe target seam.

    Non-target models are a successful no-op. Qwen3.6 template content is
    validated lazily when the real processor reaches this seam, so tests do
    not need to download or initialize a model. Ambiguous content raises
    ``TemplatePatchError`` instead of silently changing history rendering.
    """
    from config import HIGH_QUALITY_MODEL

    if str(model or "").strip() != HIGH_QUALITY_MODEL:
        return True

    try:
        from mlx_vlm import prompt_utils
    except Exception:
        return False
    current = _get_chat_template_seam(prompt_utils)
    if current is None or getattr(current, _PATCH_MARKER, False):
        return current is not None

    original = current

    def patched_get_chat_template(
        processor,
        messages,
        add_generation_prompt,
        tokenize=False,
        **kwargs,
    ):
        template = kwargs.get("chat_template")
        if template is None:
            template = _processor_template(processor)
        # Validate the actual processor/template at first use. Never guess
        # when the pinned Qwen3.6 condition has changed or is ambiguous.
        kwargs["chat_template"] = patch_template_text(template)
        return original(
            processor,
            messages,
            add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    setattr(patched_get_chat_template, _PATCH_MARKER, True)
    setattr(patched_get_chat_template, "_endeavor_original", original)
    prompt_utils.get_chat_template = patched_get_chat_template
    return True


__all__ = ["TemplatePatchError", "apply_for_model", "patch_template_text"]
