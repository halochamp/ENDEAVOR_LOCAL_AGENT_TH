#!/usr/bin/env python3
"""Agent TH-owned mlx_vlm launcher.

The wrapper gives the standalone server a deterministic process identity and
pins every request to the model selected by Agent TH. Other local clients may
share that already-loaded model, but they cannot use request.model to evict or
replace it.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


def _option_value(argv: Sequence[str], option: str) -> str:
    for index, arg in enumerate(argv):
        if arg == option and index + 1 < len(argv):
            return str(argv[index + 1])
        if arg.startswith(option + "="):
            return arg.split("=", 1)[1]
    return ""


def _install_owner_model_gate(owner_model: str) -> None:
    owner_model = str(owner_model or "").strip()
    if not owner_model:
        raise RuntimeError("owner model is required")

    import mlx_vlm.server as server_package

    current = server_package.get_cached_model
    if getattr(current, "_endeavor_th_owner_model", None) == owner_model:
        return

    def guarded_get_cached_model(model_path, *args, **kwargs):
        if str(model_path) != owner_model:
            raise ValueError(
                "model switch denied: Agent TH owns this model server with model "
                f"{owner_model!r}; requested {model_path!r}"
            )
        return current(model_path, *args, **kwargs)

    guarded_get_cached_model._endeavor_th_owner_model = owner_model
    server_package.get_cached_model = guarded_get_cached_model


def main() -> None:
    owner_model = _option_value(sys.argv, "--model")
    _install_owner_model_gate(owner_model)
    # Apply only the generic, in-memory APC seam. Incompatible mlx-vlm builds
    # keep their native APC behavior; the patch module is fail-closed.
    try:
        from scripts.apc_extra_hash_patch import apply as _apply_apc_patch
        _apply_apc_patch()
    except Exception:
        pass
    from scripts.apc_tool_call_template_patch import apply_for_model
    if not apply_for_model(owner_model):
        raise RuntimeError(
            "APC tool-call template stabilization seam is unavailable for the owner model"
        )
    try:
        from scripts.apc_self_check_patch import apply as _apply_apc_self_check
    except Exception as exc:
        raise RuntimeError("APC self-check compatibility seam is unavailable") from exc
    if not _apply_apc_self_check():
        raise RuntimeError("APC self-check compatibility seam could not be installed")
    from mlx_vlm.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
