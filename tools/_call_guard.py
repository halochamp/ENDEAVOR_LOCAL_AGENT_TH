# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""_call_guard.py — per-turn "is this the first call to tool X this turn?" tracker.

Shared by tools whose docstring is split into a short, always-visible
routing-critical schema half and a deferred syntax/retry manual injected
into the tool's own return string the first time it's called each turn.
Injecting into the tool's own return value — not a separate message type —
needs no ToolNode override and never touches the bound tool schema, so it
can't break prefix/KV caching.

reset() must be called exactly once per turn, from graph.py's react_node.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_counts: dict[str, int] = {}


def reset() -> None:
    with _lock:
        _counts.clear()


def first_call_this_turn(name: str) -> bool:
    """Atomically increment `name`'s per-turn call count and return True iff
    this is the first call. Locked: ToolNode (langgraph.prebuilt) runs
    several tool_calls from one AIMessage concurrently via a real
    ThreadPoolExecutor, so an unlocked increment lets two concurrent calls
    both read 0."""
    with _lock:
        n = _counts.get(name, 0) + 1
        _counts[name] = n
        return n == 1
