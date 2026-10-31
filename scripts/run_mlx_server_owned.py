#!/usr/bin/env python3
"""Agent TH-owned mlx_vlm launcher.

The wrapper gives the standalone server a deterministic process identity and
pins every request to the model selected by Agent TH. Other local clients may
share that already-loaded model, but they cannot use request.model to evict or
replace it.
"""
from __future__ import annotations

import importlib.metadata
import re
import sys
from collections.abc import Sequence
from pathlib import Path


_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from model_registry import ModelRegistryError, get_native_apc_contract

# Native APC memory ceiling, sorted tool-schema keys and the speculative/APC fixes
# this project depends on first shipped in mlx-vlm 0.7.10; older servers are refused.
MIN_MLX_VLM = "0.7.10"


def _release_tuple(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", str(version or ""))
    return tuple(int(part) for part in match.groups()) if match else None


def _require_supported_mlx_vlm(installed: str | None = None) -> str:
    """Return the installed mlx-vlm version, or raise when it is too old to serve."""
    if installed is None:
        try:
            installed = importlib.metadata.version("mlx-vlm")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"mlx-vlm is not installed; Agent TH requires mlx-vlm >= {MIN_MLX_VLM}"
            ) from exc
    found = _release_tuple(installed)
    if found is None:
        raise RuntimeError(f"cannot read the mlx-vlm version {installed!r}")
    if found < _release_tuple(MIN_MLX_VLM):
        raise RuntimeError(
            f"Agent TH requires mlx-vlm >= {MIN_MLX_VLM} but this Python has {installed}; "
            "install install_library/requirements.txt or point V2_MODEL_SERVER_PYTHON "
            "at an environment with a supported mlx-vlm"
        )
    return installed


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
    _require_supported_mlx_vlm()
    # Conversation-derived APC snapshots stay in memory: switch the SSD tier off and
    # delete what earlier processes left, before mlx_vlm reads its APC settings.
    from scripts import apc_disk_policy
    apc_disk_policy.apply()
    owner_model = _option_value(sys.argv, "--model")
    _install_owner_model_gate(owner_model)
    try:
        apc_contract = get_native_apc_contract(owner_model)
    except ModelRegistryError as exc:
        raise RuntimeError(
            "owner model has no declarative runtime contract in model_registry.json"
        ) from exc

    if apc_contract.enabled:
        # Native APC is capability-driven rather than repo-ID-driven. Future
        # models inherit this path by declaring a validated registry contract.
        try:
            from scripts.apc_extra_hash_patch import apply as _apply_apc_patch
        except Exception as exc:
            raise RuntimeError("APC request-policy/salt seam is unavailable") from exc
        if not _apply_apc_patch():
            raise RuntimeError("APC request-policy/salt seam could not be installed")

        from scripts.apc_tool_call_template_patch import apply_policy
        if not apply_policy(apc_contract.template_policy):
            raise RuntimeError(
                "APC template policy could not be installed: "
                f"{apc_contract.template_policy!r}"
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
