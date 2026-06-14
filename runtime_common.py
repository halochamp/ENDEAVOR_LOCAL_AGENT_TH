"""runtime_common.py — shared infra for both ENDEAVOR_AGENT_V2 entry points

CLI (endeavor_agent.py) and AGENT_UI backend (agent_server.py) bootstrap the same memory
store, poll the same MLX server / internet liveness, and detect skills the same
way. This module is the single source of truth for that overlap so the two
entry points cannot silently drift apart (Dual-Path Prohibition, CLAUDE.md §5).

Each entry point still owns its own turn-execution loop, callback/event
transport, and presentation (tool-detail truncation, printing vs. websocket
push) — those differ by design for CLI vs. web UI and are NOT shared here.
"""
from __future__ import annotations
import json
import os
import re
import sqlite3
import threading
import urllib.request
import uuid
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from config import MLX_BASE_URL, RECURSION_LIMIT as _RECURSION_LIMIT

# ── Shared paths & constants ───────────────────────────────────────────────────

_MEMORY_DB = os.path.join(os.path.dirname(__file__), "logs", "history.db")
_MEMORY_MD = os.path.join(os.path.dirname(__file__), "logs", "memory.md")
_MEMORY_THREAD = "main"
_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")

_MAX_MEMORY_PAIRS = 100  # 100 Q&A pairs — trim oldest when exceeded
_MAX_DB_ROWS = 1000      # จำกัด checkpoint rows ใน history.db

# Bump this when LangGraph schema changes in a breaking way (new columns, blob format change).
# On mismatch: old DB is renamed to history.db.bak.<ts> and a fresh DB is created.
_DB_SCHEMA_VERSION = 1

_THINKING_ROTATE_S = 30  # rotate thinking message every 30s
_THINKING_MSGS = [
    "thinking…",
    "analyzing…",
    "processing…",
    "reasoning…",
    "reviewing…",
    "still working…",
]


# ── Thinking-message rotator ───────────────────────────────────────────────────

class ThinkingTimer:
    """Background-thread message rotator shared by CLI and AGENT_UI UI callbacks.

    _UICallback and _WSCallback each drove an identical rotate-_THINKING_MSGS
    state machine via self-rescheduling threading.Timer — guarded by two subtly
    different sentinels (_llm_start: float|None vs _active: bool, same meaning).
    This is the single source so the two can't drift apart (CLAUDE.md §5).
    The caller supplies emit(label) — terminal relabel vs. websocket queue push —
    everything about *how* a label is shown stays in the caller, only the *when*
    and *what* (rotation timing, message sequence) lives here.
    """

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._timer: threading.Timer | None = None
        self._msg_idx: int = 0
        self._active: bool = False
        self._had_tool: bool = False

    def _tick(self) -> None:
        if not self._active:
            return
        self._msg_idx = (self._msg_idx + 1) % len(_THINKING_MSGS)
        self._emit(_THINKING_MSGS[self._msg_idx])
        self._timer = threading.Timer(_THINKING_ROTATE_S, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def start(self) -> None:
        self.cancel()
        self._active = True
        self._msg_idx = 0
        self._emit(_THINKING_MSGS[0])
        self._timer = threading.Timer(_THINKING_ROTATE_S, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def mark_tool(self) -> None:
        """Signal that a tool ran this turn (does not affect the timer)."""
        self._had_tool = True

    @property
    def had_tool(self) -> bool:
        return self._had_tool

    def cancel(self) -> None:
        self._active = False
        if self._timer:
            self._timer.cancel()
            self._timer = None


# ── Liveness checks ────────────────────────────────────────────────────────────

def mlx_up() -> bool:
    try:
        urllib.request.urlopen(MLX_BASE_URL.rstrip("/") + "/models", timeout=3)
        return True
    except Exception:
        return False


def internet_up(retries: int = 3, timeout: int = 1) -> bool:
    import socket
    for _ in range(retries):
        try:
            socket.setdefaulttimeout(timeout)
            socket.create_connection(("1.1.1.1", 53))
            return True
        except Exception:
            pass
    return False


# ── Memory store bootstrap ─────────────────────────────────────────────────────

def guard_db_schema(path: str, verbose: bool = True) -> None:
    """Rename history.db on schema version mismatch to prevent silent corruption."""
    if not os.path.exists(path):
        return
    try:
        conn = sqlite3.connect(path)
        (ver,) = conn.execute("PRAGMA user_version").fetchone()
        if ver == 0:
            # Pre-versioning DB or fresh — stamp as current version and continue.
            conn.execute(f"PRAGMA user_version = {_DB_SCHEMA_VERSION}")
            conn.commit()
        elif ver != _DB_SCHEMA_VERSION:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = f"{path}.bak.{ts}"
            conn.close()
            os.rename(path, bak)
            if verbose:
                print(f" ⚠️  history.db schema เวอร์ชันไม่ตรง (found={ver}, expected={_DB_SCHEMA_VERSION})")
                print(f"    ย้ายไปที่ {os.path.basename(bak)} — เริ่ม session ใหม่โดยไม่มี memory เก่า")
            return
        conn.close()
    except Exception as e:
        if verbose:
            print(f" ⚠️  ตรวจสอบ history.db schema ไม่ได้: {e}")


def open_memory_store(path: str = _MEMORY_DB, verbose: bool = True) -> tuple[sqlite3.Connection, SqliteSaver]:
    """Guard schema, open the sqlite connection, wrap it in a SqliteSaver — the
    exact bootstrap sequence both entry points need before build_graph()."""
    guard_db_schema(path, verbose=verbose)
    conn = sqlite3.connect(path, check_same_thread=False)
    return conn, SqliteSaver(conn)


def load_memory_md(path: str = _MEMORY_MD) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def purge_thread(conn: sqlite3.Connection, thread_id: str) -> None:
    """ลบ checkpoint rows ของ thread ที่ใช้แล้วทิ้ง (UUID session)"""
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    conn.commit()


def enforce_db_limit(conn: sqlite3.Connection, max_rows: int = _MAX_DB_ROWS, keep_thread: str = "") -> None:
    """ถ้า checkpoints เกิน max_rows → ลบแถวเก่าสุดออก (ยกเว้น main thread และ current session)
    หลังลบ: ลบ writes rows ที่ parent checkpoint หายไปแล้ว (orphan cleanup)"""
    (count,) = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()
    if count <= max_rows:
        return
    delete_n = count - max_rows
    keep = [t for t in [_MEMORY_THREAD, keep_thread] if t]
    placeholders = ",".join("?" * len(keep))
    conn.execute(f"""
        DELETE FROM checkpoints WHERE (thread_id, checkpoint_ns, checkpoint_id) IN (
            SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints
            WHERE thread_id NOT IN ({placeholders})
            ORDER BY checkpoint_id ASC
            LIMIT ?
        )
    """, (*keep, delete_n))
    conn.execute("""
        DELETE FROM writes WHERE (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
            SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints
        )
    """)
    conn.commit()


def vacuum_db(conn: sqlite3.Connection) -> None:
    """VACUUM the SQLite DB to reclaim disk space from deleted rows.
    Blocks — only call on process exit, never mid-session."""
    try:
        conn.execute("VACUUM")
        conn.commit()
    except Exception:
        pass


def trim_memory(saver: SqliteSaver, max_pairs: int) -> None:
    """Trim main thread to last max_pairs Q&A pairs."""
    cfg = {"configurable": {"thread_id": _MEMORY_THREAD}}
    tup = saver.get_tuple(cfg)
    if not tup:
        return
    msgs = tup.checkpoint.get("channel_values", {}).get("messages", [])
    if len(msgs) <= max_pairs * 2:
        return
    trimmed = msgs[-(max_pairs * 2):]
    new_cp = {
        **tup.checkpoint,
        "id": tup.checkpoint["id"],  # reuse existing id — INSERT OR REPLACE updates in place, preserves monotonic ordering
        "channel_values": {**tup.checkpoint.get("channel_values", {}), "messages": trimmed},
    }
    new_versions = {k: v for k, v in tup.checkpoint.get("channel_versions", {}).items()}
    saver.put(
        {"configurable": {**tup.config["configurable"], "checkpoint_id": new_cp["id"]}},
        new_cp, tup.metadata or {}, new_versions,
    )


_HISTORY_LOAD_MAX_CHARS = 30_000  # cap history injected into context on /history


def load_history_pairs(saver: SqliteSaver, max_chars: int = _HISTORY_LOAD_MAX_CHARS):
    """Read Q&A pairs from _MEMORY_THREAD, newest-first until max_chars.

    Returns (msgs, loaded_chars, total_pairs):
      msgs         — list[HumanMessage|AIMessage] in chronological order
      loaded_chars — total chars of loaded msgs
      total_pairs  — total Q&A pairs stored in _MEMORY_THREAD
    """
    cfg = {"configurable": {"thread_id": _MEMORY_THREAD}}
    tup = saver.get_tuple(cfg)
    if not tup:
        return [], 0, 0
    all_msgs = tup.checkpoint.get("channel_values", {}).get("messages", [])
    qa_msgs = [m for m in all_msgs if isinstance(m, (HumanMessage, AIMessage))]
    total_pairs = sum(1 for m in qa_msgs if isinstance(m, HumanMessage))

    total_chars = 0
    selected: list = []
    for m in reversed(qa_msgs):
        chars = len(str(m.content or ""))
        if total_chars + chars > max_chars:
            break
        total_chars += chars
        selected.insert(0, m)

    return selected, total_chars, total_pairs


def append_pair_to_history(app, saver: SqliteSaver, query: str, response: str) -> None:
    """Append one Q&A pair to _MEMORY_THREAD after each ephemeral-session turn.

    Called only when thread_id != _MEMORY_THREAD so we don't double-write.
    Uses app.update_state() which triggers the add_messages reducer (append),
    creating the checkpoint if _MEMORY_THREAD doesn't exist yet.
    Trims to _MAX_MEMORY_PAIRS after appending.
    """
    if not response.strip():
        return
    cfg = {"recursion_limit": _RECURSION_LIMIT, "configurable": {"thread_id": _MEMORY_THREAD, "checkpoint_ns": ""}}
    try:
        app.update_state(cfg, {"messages": [HumanMessage(content=query), AIMessage(content=response)]})
        trim_memory(saver, _MAX_MEMORY_PAIRS)
    except Exception:
        pass


# ── Plan-step parsing ──────────────────────────────────────────────────────────

def parse_plan_steps(text: str) -> list[str]:
    """ดึง numbered steps จาก create_plan ToolMessage"""
    steps = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(\d+)\.\s+(.+)", line)
        if m:
            steps.append(m.group(2).strip())
    return steps


def extract_tool_content(output) -> str:
    """Extract text content from a tool output (ToolMessage or plain value).

    LangChain wraps tool results in ToolMessage; str() gives repr, not content.
    Single source so CLI and AGENT_UI cannot silently diverge.
    """
    if output is not None and hasattr(output, "content"):
        return output.content or ""
    return str(output) if output is not None else ""


def parse_tool_args(input_str, kwargs: dict) -> dict:
    """Parse tool input args from LangChain callback parameters.

    LangChain sometimes passes already-parsed inputs in kwargs["inputs"];
    fall back to json.loads(input_str) when not present. Single source so
    both adapters see the same args regardless of which path LangChain takes.
    """
    args = kwargs.get("inputs") or {}
    if not args:
        try:
            args = json.loads(input_str) if isinstance(input_str, str) else (input_str or {})
        except Exception:
            args = {}
    return args


# ── Skill detection ────────────────────────────────────────────────────────────

def first_role_line(content: str) -> str:
    """First non-heading line under the '## Role' section, truncated to 60 chars
    (or "" if the section is absent/empty). Single canonical scan — CLI registry,
    AGENT_UI skill list, and skill activation all detected this independently
    before, with one subtly different stop condition."""
    in_role = False
    for line in content.splitlines():
        line = line.strip()
        if line == "## Role":
            in_role = True
            continue
        if in_role and line and not line.startswith("#"):
            return line[:60]
    return ""


def scan_skill_roles(skills_dir: str = _SKILLS_DIR) -> list[tuple[str, str]]:
    """[(skill_name, role_line)] for every *.md in skills_dir.

    Shared skill-detection primitive for the CLI registry and the AGENT_UI skill
    list — each caller wraps the neutral (name, role) tuple in its own dict shape
    (different key names / fallback text tuned for their own UI surface)."""
    result = []
    if not os.path.isdir(skills_dir):
        return result
    for fname in sorted(os.listdir(skills_dir)):
        if not fname.endswith(".md"):
            continue
        name = fname[:-3]
        role = ""
        try:
            with open(os.path.join(skills_dir, fname), encoding="utf-8") as f:
                role = first_role_line(f.read())
        except Exception:
            pass
        result.append((name, role))
    return result


# ── Turn execution ─────────────────────────────────────────────────────────────

def run_turn_core(
    app,
    query: str,
    stream_cfg: dict,
    *,
    thread_id: str,
    saver: SqliteSaver,
    db_conn: sqlite3.Connection,
    clear_scratch: Callable[[], None],
    reset_web_counter: Callable[[], None],
    on_final: Callable[[str], None] | None = None,
) -> str:
    """Shared turn skeleton for both entry points.

    _run_turn (CLI) and _run_agent_sync (AGENT_UI) each drove an identical
    clear-scratch → reset-web-counter → app.stream(stream_mode="updates") →
    AIMessage final-content loop → trim_memory/enforce_db_limit sequence —
    only the callback object inside stream_cfg and the UI reaction to the
    final chunk differed. Single source so the two can't drift apart
    (Dual-Path Prohibition, CLAUDE.md §5).

    clear_scratch/reset_web_counter are injected rather than imported here:
    both resolve to `tools.scratchpad`/`tools.web_cache` symbols, and importing
    either pulls in all of `tools/__init__.py` (27 tools incl. browser/RAG
    deps, ~1s) — infra (this module) staying free of the tool layer keeps
    `import runtime_common` cheap for callers like _test_thinking_timer.py.

    Housekeeping runs in `finally`: the old CLI _run_turn swallowed stream
    exceptions internally and returned normally, so the REPL loop's
    trim/enforce always ran after every turn — including failed ones.
    AGENT_UI's old _run_agent_sync only ran them on the success path inside
    its try. Standardizing on "always run" preserves the CLI (primary
    interface, CLAUDE.md risk rule) unchanged and only hardens the server
    side against unbounded DB growth from a string of failing turns.

    on_final(content) fires once per qualifying AIMessage chunk — adapters
    use it to react to the synthesis phase starting (set a UI label, stash
    into their own turn-state object, etc). The return value is the same
    final text, for callers that don't keep their own running copy.
    """
    clear_scratch()
    reset_web_counter()
    final = ""
    try:
        for chunk in app.stream(
            {"messages": [{"role": "user", "content": query}]},
            config=stream_cfg, stream_mode="updates",
        ):
            for _node, update in chunk.items():
                msgs = update.get("messages", []) if isinstance(update, dict) else []
                for m in msgs:
                    if isinstance(m, AIMessage):
                        if not (getattr(m, "tool_calls", None) or []) and (m.content or "").strip():
                            final = m.content
                            if on_final:
                                on_final(final)
    finally:
        if thread_id == _MEMORY_THREAD:
            try:
                trim_memory(saver, _MAX_MEMORY_PAIRS)
            except Exception:
                pass
        else:
            # auto-save every ephemeral turn to _MEMORY_THREAD
            try:
                append_pair_to_history(app, saver, query, final)
            except Exception:
                pass
        try:
            enforce_db_limit(db_conn, keep_thread=thread_id if thread_id != _MEMORY_THREAD else "")
        except Exception:
            pass
    return final
