"""Project-local APC self-check compatibility for cache-less language models.

Some public mlx-vlm language models omit ``make_cache`` even though runtime
generation supports them through ``generate.ar._make_cache``. This bridge only
corrects that diagnostic mismatch; models with ``make_cache`` continue through
the upstream self-check unchanged.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable


_PATCH_MARKER = "_endeavor_th_no_make_cache_apc_self_check"


def _runtime_make_cache() -> Callable[..., Any] | None:
    try:
        from mlx_vlm.generate.ar import _make_cache

        parameters = inspect.signature(_make_cache).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    if "model" not in parameters or "left_padding" not in parameters:
        return None
    return _make_cache


def apply() -> bool:
    """Install the fallback self-check seam, returning False if unsafe."""
    try:
        from mlx_vlm import apc
    except Exception:
        return False

    original = getattr(apc, "self_check_model_apc", None)
    if not callable(original):
        return False
    if getattr(original, _PATCH_MARKER, False):
        return True

    make_cache = _runtime_make_cache()
    validate_layout = getattr(apc, "validate_prompt_cache_layout", None)
    model_apc_mode = getattr(apc, "model_apc_mode", None)
    if not callable(make_cache) or not callable(validate_layout) or not callable(model_apc_mode):
        return False

    def patched(model: Any, *, kv_bits: Any = None, log: bool = True) -> Any:
        language_model = getattr(model, "language_model", model)
        if hasattr(language_model, "make_cache"):
            return original(model, kv_bits=kv_bits, log=log)

        layers = getattr(language_model, "layers", None)
        try:
            if layers is None or len(layers) == 0:
                return original(model, kv_bits=kv_bits, log=log)
        except (TypeError, AttributeError):
            return original(model, kv_bits=kv_bits, log=log)

        try:
            parameters = inspect.signature(make_cache).parameters
            kwargs: dict[str, Any] = {}
            if "kv_bits" in parameters:
                kwargs["kv_bits"] = kv_bits
            if "prefill_length" in parameters:
                kwargs["prefill_length"] = 0
            caches = make_cache(language_model, [0], **kwargs)
            if not caches:
                return original(model, kv_bits=kv_bits, log=log)
            result = validate_layout(caches, apc_mode=model_apc_mode(language_model))
            try:
                result.notes = list(getattr(result, "notes", ()) or ())
                result.notes.append("no-make_cache fallback via mlx_vlm.generate.ar._make_cache")
            except Exception:
                pass
            return result
        except Exception:
            return original(model, kv_bits=kv_bits, log=log)

    setattr(patched, _PATCH_MARKER, True)
    setattr(patched, "_endeavor_original", original)
    apc.self_check_model_apc = patched
    return True


__all__ = ["apply"]
