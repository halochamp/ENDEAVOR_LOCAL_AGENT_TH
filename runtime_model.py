"""Shared local MLX listener/model discovery for Agent TH Python front ends.

The Electron host has an equivalent JS helper because it owns native process
lifecycle. Python callers (CLI + agent_server) share this module so active-model
identity stays process-first and fail-closed everywhere.
"""
from __future__ import annotations

import re
import subprocess
import urllib.parse


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def model_from_process_command(command: str) -> str:
    match = re.search(
        r"(?:^|\s)--model(?:=|\s+)(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))",
        str(command or ""),
    )
    return (match.group(1) or match.group(2) or match.group(3)) if match else ""


def active_local_mlx_model(base_url: str) -> str:
    """Return the local listener's ``--model`` value, else ``""``.

    ``/v1/models`` is intentionally not used as active-model truth. A launcher
    may advertise multiple supported models while one process owns exactly one
    active model. Remote/custom backends are not process-inspected here.
    """
    try:
        parsed = urllib.parse.urlparse(base_url)
        host = (parsed.hostname or "").lower()
        if host not in _LOCAL_HOSTS:
            return ""
        port = int(parsed.port or 80)
        found = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        pid = 0
        for line in found.stdout.splitlines():
            if line.startswith("p") and line[1:].isdigit():
                pid = int(line[1:])
                break
        if not pid:
            return ""
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return model_from_process_command(proc.stdout.strip())
    except Exception:
        return ""
