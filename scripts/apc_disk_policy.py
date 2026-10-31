"""Keep the native APC off the SSD.

mlx-vlm 0.7 spills every exact APC store to ``~/.cache/mlx-vlm/apc`` (about 1 GB
per long turn, up to 20 GB per model). Those snapshots are derived from private
conversation content and would persist across restarts without the user ever
choosing it, so Agent TH keeps conversation-derived state in memory only: the disk
tier is switched off and snapshots left by earlier processes are deleted.
Memory APC is unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil

# Upstream namespace directory: _safe_namespace("<model>#s<schema>-<8 hex>").
_NAMESPACE_DIR = re.compile(r".+_s\d+-[0-9a-f]{8}")


def disk_root() -> Path:
    if os.environ.get("APC_DISK_PATH"):
        return Path(os.environ["APC_DISK_PATH"])
    home = os.environ.get("MLX_VLM_CACHE_HOME") or Path.home() / ".cache" / "mlx-vlm"
    return Path(home) / "apc"


def apply(root: Path | None = None) -> int:
    """Disable the APC disk tier and delete snapshots earlier processes left."""
    os.environ["APC_DISK_ENABLED"] = "0"
    root = root or disk_root()
    removed = 0
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and not child.is_symlink() and _NAMESPACE_DIR.fullmatch(child.name):
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
    print(f"agent_th_apc_disk: disabled removed_namespaces={removed}", flush=True)
    return removed


__all__ = ["apply", "disk_root"]
