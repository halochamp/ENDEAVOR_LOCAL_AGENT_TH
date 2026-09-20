"""Fail-closed APC tool-call template stabilization for Agent TH.

The owner launcher selects one declarative policy for each public model. The
rewrite is installed in memory and validates the actual processor/tokenizer
template lazily on first use; no model snapshot or private revision is baked
into the public project.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

from model_registry import REGISTRY, ModelRegistryError, get_native_apc_contract


_UPSTREAM_CONDITION = (
    "(preserve_thinking is defined and preserve_thinking is true) "
    "or (loop.index0 > ns.last_query_index)"
)
_PATCHED_CONDITION = (
    _UPSTREAM_CONDITION
    + " or (message.tool_calls and message.tool_calls is iterable "
    "and message.tool_calls is not mapping)"
)
_QWEN35_POSITION_UPSTREAM_TAG = "{%- if loop.index0 > ns.last_query_index %}"
_QWEN35_POSITION_PATCHED_TAG = (
    "{%- if loop.index0 > ns.last_query_index "
    "or (message.tool_calls is defined and message.tool_calls) %}"
)
_QWEN35_TOOL_PREDICATE = (
    "message.tool_calls and message.tool_calls is iterable "
    "and message.tool_calls is not mapping"
)
_QWEN3_NESTED_UPSTREAM_TAG = (
    "{%- if loop.last or (not loop.last and reasoning_content) %}"
)
_QWEN3_NESTED_PATCHED_TAG = (
    "{%- if loop.last or (not loop.last and reasoning_content) "
    "or (message.tool_calls is defined and message.tool_calls) %}"
)
_QWEN3_TOOL_TAG = "{%- if message.tool_calls %}"
_NATIVE_STABLE_CONDITION = (
    "preserve_thinking is undefined or preserve_thinking is true "
    "or loop.index0 > ns.last_query_index"
)
_NATIVE_PATCH_MARKER = "_endeavor_th_native_preserve_thinking_apc_patch"

_PATCH_MARKERS = {
    "qwen36_tool_call": "_endeavor_th_qwen36_tool_call_apc_patch",
    "qwen35_tool_call": "_endeavor_th_qwen35_tool_call_apc_patch",
    "qwen3_tool_call": "_endeavor_th_qwen3_tool_call_apc_patch",
}
SUPPORTED_TEMPLATE_POLICIES = frozenset({*_PATCH_MARKERS, "native_preserve"})

# Compatibility/introspection view only. Runtime selection is contract-driven
# through model_registry.json; this module never branches on repo IDs.
MODEL_TEMPLATE_POLICIES = {
    spec.repo_id: spec.native_apc.template_policy
    for spec in REGISTRY.selectable_models
    if spec.native_apc.enabled
}


class TemplatePatchError(RuntimeError):
    """The actual loaded template no longer matches a verified policy seam."""


def patch_template_text(template: str) -> str:
    """Patch exactly one verified Qwen3.6 tool-call condition."""
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


def _patch_qwen35_position(template: str) -> str:
    if _QWEN35_POSITION_PATCHED_TAG in template:
        return template
    count = template.count(_QWEN35_POSITION_UPSTREAM_TAG)
    if count != 1:
        raise TemplatePatchError(
            "expected exactly one Qwen3.5 position-dependent thinking condition; "
            f"found {count}"
        )
    return template.replace(_QWEN35_POSITION_UPSTREAM_TAG, _QWEN35_POSITION_PATCHED_TAG, 1)


def patch_qwen35_template_text(template: str) -> str:
    """Patch the verified Qwen3.5 outer position guard only."""
    if not isinstance(template, str) or not template:
        raise TemplatePatchError("chat template is missing or not a string")
    if template.count(_QWEN35_TOOL_PREDICATE) != 1:
        raise TemplatePatchError("Qwen3.5 tool-call rendering predicate is missing")
    if _QWEN3_NESTED_UPSTREAM_TAG in template or _QWEN3_NESTED_PATCHED_TAG in template:
        raise TemplatePatchError("Qwen3 nested thinking guard is not a Qwen3.5 template")
    return _patch_qwen35_position(template)


def _patch_qwen3_outer(template: str) -> str:
    if _QWEN35_POSITION_PATCHED_TAG in template:
        return template
    count = template.count(_QWEN35_POSITION_UPSTREAM_TAG)
    if count != 1:
        raise TemplatePatchError(
            "expected exactly one Qwen3 position-dependent thinking condition; "
            f"found {count}"
        )
    return template.replace(_QWEN35_POSITION_UPSTREAM_TAG, _QWEN35_POSITION_PATCHED_TAG, 1)


def patch_qwen3_template_text(template: str) -> str:
    """Patch Qwen3's outer and nested reasoning guards for tool-call history."""
    if not isinstance(template, str) or not template:
        raise TemplatePatchError("chat template is missing or not a string")
    if template.count(_QWEN3_TOOL_TAG) != 1:
        raise TemplatePatchError("Qwen3 tool-call rendering predicate is missing")
    patched = _patch_qwen3_outer(template)
    if _QWEN3_NESTED_PATCHED_TAG in patched:
        if patched.count(_QWEN3_NESTED_PATCHED_TAG) != 1:
            raise TemplatePatchError("unexpected number of patched Qwen3 nested guards")
        return patched
    count = patched.count(_QWEN3_NESTED_UPSTREAM_TAG)
    if count != 1:
        raise TemplatePatchError(
            "expected exactly one Qwen3 nested thinking condition; "
            f"found {count}"
        )
    return patched.replace(_QWEN3_NESTED_UPSTREAM_TAG, _QWEN3_NESTED_PATCHED_TAG, 1)


_POLICY_PATCHERS: dict[str, Callable[[str], str]] = {
    "qwen36_tool_call": patch_template_text,
    "qwen35_tool_call": patch_qwen35_template_text,
    "qwen3_tool_call": patch_qwen3_template_text,
}


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


def _validate_native_template_text(template: str) -> None:
    if not isinstance(template, str) or not template:
        raise TemplatePatchError("chat template is missing or not a string")
    if template.count(_NATIVE_STABLE_CONDITION) != 1:
        raise TemplatePatchError(
            "expected exactly one native preserve-thinking condition"
        )
    if _QWEN35_TOOL_PREDICATE not in template:
        raise TemplatePatchError("native tool-call rendering predicate is missing")


def _apply_native_preserve() -> bool:
    """Lock an already-stable native template to preserve thinking history."""
    try:
        from mlx_vlm import prompt_utils
    except Exception:
        return False
    current = _get_chat_template_seam(prompt_utils)
    if current is None:
        return False
    if getattr(current, _NATIVE_PATCH_MARKER, False):
        return True
    original = current

    def patched_get_chat_template(
        processor,
        messages,
        add_generation_prompt,
        tokenize=False,
        **kwargs,
    ):
        if kwargs.get("chat_template") is None:
            template = _processor_template(processor)
            if template is None:
                raise TemplatePatchError("owner processor exposes no string chat template")
            _validate_native_template_text(template)
            kwargs["preserve_thinking"] = True
        return original(
            processor,
            messages,
            add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    setattr(patched_get_chat_template, _NATIVE_PATCH_MARKER, True)
    setattr(patched_get_chat_template, "_endeavor_th_apc_template_policy", "native_preserve")
    setattr(patched_get_chat_template, "_endeavor_original", original)
    prompt_utils.get_chat_template = patched_get_chat_template
    return True


def apply_policy(policy: str) -> bool:
    """Install one declarative native-AR APC template policy."""
    policy = str(policy or "").strip()
    if policy not in SUPPORTED_TEMPLATE_POLICIES:
        return False
    if policy == "native_preserve":
        return _apply_native_preserve()
    patcher = _POLICY_PATCHERS[policy]
    marker = _PATCH_MARKERS[policy]

    try:
        from mlx_vlm import prompt_utils
    except Exception:
        return False
    current = _get_chat_template_seam(prompt_utils)
    if current is None:
        return False
    installed_policy = getattr(current, "_endeavor_th_apc_template_policy", None)
    if installed_policy is not None:
        return installed_policy == policy
    if getattr(current, marker, False):
        return True

    original = current

    def patched_get_chat_template(
        processor,
        messages,
        add_generation_prompt,
        tokenize=False,
        **kwargs,
    ):
        # Explicit custom templates belong to the caller and must not be
        # rewritten by a model policy intended for the owner template.
        if kwargs.get("chat_template") is None:
            template = _processor_template(processor)
            if template is None:
                raise TemplatePatchError("owner processor exposes no string chat template")
            kwargs["chat_template"] = patcher(template)
        return original(
            processor,
            messages,
            add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    setattr(patched_get_chat_template, marker, True)
    setattr(patched_get_chat_template, "_endeavor_th_apc_template_policy", policy)
    setattr(patched_get_chat_template, "_endeavor_original", original)
    prompt_utils.get_chat_template = patched_get_chat_template
    return True


def apply_for_model(model: str) -> bool:
    """Compatibility wrapper resolving policy from the declarative registry."""
    try:
        contract = get_native_apc_contract(str(model or "").strip())
    except ModelRegistryError:
        return True
    if not contract.enabled:
        return True
    return apply_policy(contract.template_policy)


def apply(model: str) -> bool:
    """Compatibility alias for callers that use the launcher-facing name."""
    return apply_for_model(model)


__all__ = [
    "MODEL_TEMPLATE_POLICIES",
    "SUPPORTED_TEMPLATE_POLICIES",
    "TemplatePatchError",
    "apply",
    "apply_for_model",
    "apply_policy",
    "patch_qwen3_template_text",
    "patch_qwen35_template_text",
    "patch_template_text",
]
