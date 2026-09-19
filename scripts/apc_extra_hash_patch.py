"""Small, fail-closed APC correctness patch for the TH-owned mlx-vlm server.

This is intentionally limited to the generic exact-prefix hash seam.  It does
not implement a cache, touch site-packages on disk, or know about MAX/DFlash.
Background/helper tenants bypass lookup/store so they cannot evict the main
interactive tenant's exact-prefix entries.
"""
from __future__ import annotations

from typing import Any


_MARKER = "_endeavor_th_apc_extra_hash_patch"
_BACKGROUND_PREFIXES = ("endeavor-th-background", "endeavor-apc-bypass:")


def is_background_tenant(value: Any) -> bool:
    return str(value or "").startswith(_BACKGROUND_PREFIXES)


def _apc_hash_kwargs(prompt_kwargs: dict | None) -> dict:
    kwargs = dict(prompt_kwargs or {})
    # mlx-vlm hashes image content separately through image_hash.  Keep the
    # image payload as the semantic salt, but drop inputs_embeds whose shape is
    # tied to the full prompt. Audio/video embeddings can carry modality state
    # not represented by the text key, so preserve them conservatively.
    has_audio_or_video = any(
        kwargs.get(key) is not None
        for key in ("input_features", "pixel_values_videos")
    )
    if not has_audio_or_video:
        kwargs.pop("inputs_embeds", None)
    # The all-ones mask is redundant regardless of image/audio/video. Keep
    # unusual masks in the hash so the patch fails closed for real padding or
    # other non-standard attention semantics.
    mask = kwargs.get("attention_mask")
    try:
        values = mask.tolist() if mask is not None else None
        flat = []
        if values is not None:
            stack = [values]
            while stack:
                item = stack.pop()
                if isinstance(item, (list, tuple)):
                    stack.extend(item)
                else:
                    flat.append(item)
        if flat and all(value == 1 for value in flat):
            kwargs.pop("attention_mask", None)
    except Exception:
        pass
    return kwargs


def _mask_background_meta(meta: list | None, prompt_kwargs_list: list | None) -> list | None:
    """Remove APC harvest metadata only for helper rows in a mixed batch."""
    if meta is None:
        return None
    for index, kwargs in enumerate(prompt_kwargs_list or []):
        if index < len(meta) and is_background_tenant((kwargs or {}).get("_apc_tenant")):
            meta[index] = None
    return meta


def apply() -> bool:
    """Patch the installed in-memory class when the expected seam exists.

    Returning ``False`` is safe: native mlx-vlm APC remains in control instead
    of a partial/incompatible monkey patch being installed.
    """
    try:
        from mlx_vlm.generate.ar import BatchGenerator
    except Exception:
        return False
    if getattr(BatchGenerator, _MARKER, False):
        return True
    required = (
        "_apc_extra_hash",
        "_apc_pick_for",
        "_build_apc_meta_for_cold",
        "_build_mixed_prompt_batch",
    )
    if any(not callable(getattr(BatchGenerator, name, None)) for name in required):
        return False

    original_hash = BatchGenerator._apc_extra_hash
    original_pick = BatchGenerator._apc_pick_for
    original_cold = BatchGenerator._build_apc_meta_for_cold
    original_mixed = getattr(BatchGenerator, "_build_mixed_prompt_batch", None)

    def patched_hash(self, prompt_kwargs):
        return original_hash(self, _apc_hash_kwargs(prompt_kwargs))

    def patched_pick(self, sequence):
        try:
            tenant = sequence[3].get("_apc_tenant") if sequence[3] else None
        except (IndexError, AttributeError, TypeError):
            tenant = None
        if is_background_tenant(tenant):
            return None
        return original_pick(self, sequence)

    def patched_cold(self, input_ids_list, prompt_kwargs_list):
        meta = original_cold(self, input_ids_list, prompt_kwargs_list)
        return _mask_background_meta(meta, prompt_kwargs_list)

    def patched_mixed(self, sequences):
        batch = original_mixed(self, sequences)
        if batch is None:
            return None
        prompt_kwargs_list = [sequence[3] if len(sequence) > 3 else None for sequence in sequences or []]
        _mask_background_meta(getattr(batch, "_apc_meta", None), prompt_kwargs_list)
        return batch

    # Install as one transaction.  If a provider class unexpectedly makes one
    # seam read-only, restore every earlier assignment so a partial policy
    # patch cannot leave APC in a mixed state.
    replacements = {
        "_apc_extra_hash": patched_hash,
        "_apc_pick_for": patched_pick,
        "_build_apc_meta_for_cold": patched_cold,
        "_build_mixed_prompt_batch": patched_mixed,
    }
    try:
        for name, replacement in replacements.items():
            setattr(BatchGenerator, name, replacement)
    except Exception:
        for name, original in (
            ("_apc_extra_hash", original_hash),
            ("_apc_pick_for", original_pick),
            ("_build_apc_meta_for_cold", original_cold),
            ("_build_mixed_prompt_batch", original_mixed),
        ):
            try:
                setattr(BatchGenerator, name, original)
            except Exception:
                pass
        return False

    setattr(BatchGenerator, _MARKER, True)
    return True
