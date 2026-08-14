# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

from __future__ import annotations

import os


def title_from_browse_summary(url: str, summary: str) -> str:
    lines = [l.strip() for l in summary.split("\n") if l.strip()]
    if lines:
        return lines[0][:80]
    return url.split("/")[-1][:60]


def title_from_path(path: str) -> str:
    return os.path.basename(path)


def title_from_command(command: str) -> str:
    return command[:60]
