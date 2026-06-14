"""streamlit_app.py — Streamlit UI for ENDEAVOR_LOCAL_AGENT_TH

Dark VS Code-style interface, mirrors AGENT_UI (Electron) design.
WebSocket connection to ws://localhost:8765/ws with token auth.

Launch:
    cd ENDEAVOR_LOCAL_AGENT_TH
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import html
import json
import os
import queue as _queue
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import CONTEXT_MAX_CHARS, SERVER_PORT, AUTH_DISABLED

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
_TH_ROOT       = Path(__file__).parent
_TOKEN_FILE    = _TH_ROOT / ".agent_token"
_WS_URI        = f"ws://localhost:{SERVER_PORT}/ws"
_AUTH_DISABLED = AUTH_DISABLED
_SPINNER       = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_MAX_ACTIVITY  = 500


def _load_token() -> str:
    env = os.getenv("AGENT_SERVER_TOKEN", "")
    if env:
        return env
    try:
        return _TOKEN_FILE.read_text().strip()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be the first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ENDEAVOR",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark VS Code theme overlay
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""<style>
/* ── Base ──────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background:#1e1e1e !important; }
[data-testid="stSidebar"]          { background:#252526 !important; }
[data-testid="stHeader"]           { background:#252526 !important; box-shadow:none !important; }
.stMainBlockContainer              { padding-top:0.5rem; }

/* ── Hide Streamlit chrome ─────────────────────────────────────── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stStatusWidget"]     { display:none !important; }

/* ── Chat ──────────────────────────────────────────────────────── */
[data-testid="stChatMessage"]                { background:transparent !important; }
[data-testid="stChatMessage"] > div:first-child { align-items:flex-start; }

/* ── Chat input ────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background:#252526 !important;
    border-top:1px solid #3c3c3c !important;
}
[data-testid="stChatInput"] textarea {
    background:#3c3c3c !important;
    color:#d4d4d4 !important;
    border-color:#3c3c3c !important;
    caret-color:#569cd6;
}

/* ── Buttons ───────────────────────────────────────────────────── */
.stButton > button {
    background:#2d2d2d !important; color:#d4d4d4 !important;
    border:1px solid #3c3c3c !important; font-size:12px !important;
    padding:3px 8px !important;
}
.stButton > button:hover { background:#3c3c3c !important; }

/* ── Tabs ──────────────────────────────────────────────────────── */
[data-baseweb="tab-list"]  { background:#252526 !important; border-bottom:1px solid #3c3c3c; }
[data-baseweb="tab"]       { background:transparent !important; color:#808080 !important; font-size:12px !important; }
[aria-selected="true"][data-baseweb="tab"] {
    color:#d4d4d4 !important;
    border-bottom:2px solid #569cd6 !important;
}

/* ── Divider ───────────────────────────────────────────────────── */
hr { border-color:#3c3c3c !important; margin:4px 0 !important; }

/* ── Code ──────────────────────────────────────────────────────── */
code { background:#2d2d2d !important; color:#ce9178 !important; border-radius:3px; }
pre  { background:#2d2d2d !important; border:1px solid #3c3c3c; }

/* ── Metrics ───────────────────────────────────────────────────── */
[data-testid="stMetricLabel"] { color:#808080 !important; font-size:11px !important; }
[data-testid="stMetricValue"] { color:#d4d4d4 !important; }

/* ── Caption / markdown ────────────────────────────────────────── */
.stCaption, .stMarkdown p { color:#d4d4d4; }
small, [data-testid="stCaptionContainer"] { color:#808080 !important; font-size:11px !important; }

/* ── Sidebar padding ───────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child { padding:0.5rem 0.5rem 1rem; }
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    defs: dict = {
        # WS connection
        "_ws_conn":    None,
        "_ws_q":       _queue.Queue(),
        "_ws_thread":  None,
        # Chat
        "messages":    [],       # list of {role, content}
        "activity":    [],       # list of {kind, name, detail, ts}
        # Context
        "ctx_chars":   0,
        "ctx_max":     CONTEXT_MAX_CHARS,
        # Server state
        "model":       "",
        "server_up":   False,
        "online":      False,
        "skill":       "",
        "skills":      [],
        # Files
        "files":       [],
        "current_path": "",
        "_root_path":  "",
        "open_file":   None,     # {path, content, is_image}
        # Run state
        "is_busy":     False,
        "run_id":      "",
        # History
        "history_info": None,
        # Connection
        "conn_error":  "",
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init()

# ─────────────────────────────────────────────────────────────────────────────
# WebSocket helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ws_reader_thread(conn, q: _queue.Queue):
    try:
        while True:
            raw = conn.recv()
            try:
                q.put(json.loads(raw))
            except Exception:
                pass
    except Exception as e:
        q.put({"type": "_disconnect", "msg": str(e)})


def _connect():
    import websockets.sync.client as _ws_mod
    token = _load_token()
    try:
        if not _AUTH_DISABLED and token:
            conn = _ws_mod.connect(_WS_URI, subprotocols=[token])
        else:
            conn = _ws_mod.connect(_WS_URI)
        # Flush stale events from prior connection before new reader thread starts
        q = st.session_state._ws_q
        while True:
            try:
                q.get_nowait()
            except _queue.Empty:
                break
        st.session_state._ws_conn  = conn
        st.session_state.conn_error = ""
        t = threading.Thread(
            target=_ws_reader_thread,
            args=(conn, st.session_state._ws_q),
            daemon=True,
        )
        t.start()
        st.session_state._ws_thread = t
    except Exception as e:
        st.session_state._ws_conn  = None
        st.session_state.conn_error = str(e)


def _ensure_ws():
    conn = st.session_state._ws_conn
    t    = st.session_state._ws_thread
    if conn is None or (t is not None and not t.is_alive()):
        _connect()
        if st.session_state._ws_conn is not None:
            _drain_until({"files"}, timeout=3.0)


def _send(msg: dict):
    conn = st.session_state._ws_conn
    if conn is None:
        return
    try:
        conn.send(json.dumps(msg))
    except Exception:
        st.session_state._ws_conn = None


# ─────────────────────────────────────────────────────────────────────────────
# Event application
# ─────────────────────────────────────────────────────────────────────────────
def _apply(ev: dict):
    typ = ev.get("type")
    if typ == "status":
        st.session_state.server_up = ev.get("server_up", False)
        st.session_state.online    = ev.get("online",    False)
        st.session_state.skill     = ev.get("skill",     "")
        st.session_state.skills    = ev.get("skills",    [])
        st.session_state.model     = ev.get("model",     "")
    elif typ == "files":
        st.session_state.files        = ev.get("files", [])
        st.session_state.current_path = ev.get("path",  "")
        if not st.session_state._root_path:
            st.session_state._root_path = ev.get("root", ev.get("path", ""))
    elif typ == "file_content":
        st.session_state.open_file = {
            "path": ev.get("path", ""), "content": ev.get("content", ""), "is_image": False,
        }
    elif typ == "file_image":
        st.session_state.open_file = {"path": ev.get("path", ""), "content": "", "is_image": True}
    elif typ == "clear_ok":
        st.session_state.ctx_chars = 0
        st.session_state.messages.append({"role": "system", "content": "✓ Conversation cleared"})
    elif typ == "ctx_update":
        st.session_state.ctx_chars = ev.get("chars",     0)
        st.session_state.ctx_max   = ev.get("max_chars", CONTEXT_MAX_CHARS)
    elif typ == "compact_result":
        if "error" in ev:
            st.session_state.messages.append({"role": "system", "content": f"⚠ /compact error: {ev['error']}"})
        else:
            cut, before, after = ev.get("cut", 0), ev.get("before", 0), ev.get("after", 0)
            st.session_state.messages.append({
                "role": "system",
                "content": f"✓ Compacted — removed {cut} messages ({before:,} → {after:,} chars)",
            })
    elif typ == "memory_ok":
        st.session_state.history_info = ev
        topics = ", ".join(ev.get("topics", [])[:5]) or "–"
        st.session_state.messages.append({
            "role": "system",
            "content": (
                f"📚 History: {ev.get('turns', 0)} turns / {ev.get('total', 0):,} chars  |  "
                f"Topics: {topics}"
            ),
        })
    elif typ == "skill_change":
        st.session_state.skill = ev.get("skill", "")
        if msg_text := ev.get("msg", ""):
            st.session_state.messages.append({"role": "system", "content": f"⚡ {msg_text}"})
    elif typ == "_disconnect":
        st.session_state._ws_conn  = None
        st.session_state.conn_error = ev.get("msg", "disconnected")


def _drain() -> bool:
    """Drain queue without blocking. Returns True if anything was processed."""
    q = st.session_state._ws_q
    changed = False
    while True:
        try:
            _apply(q.get_nowait())
            changed = True
        except _queue.Empty:
            return changed


def _drain_until(event_types: set, timeout: float = 15.0) -> dict | None:
    """Block-drain until one of event_types arrives or timeout. Returns the matching event."""
    q = st.session_state._ws_q
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ev = q.get(timeout=0.1)
            _apply(ev)
            if ev.get("type") in event_types:
                return ev
        except _queue.Empty:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def _file_icon(name: str) -> str:
    ext = Path(name).suffix.lower().lstrip(".")
    return {
        "py": "🐍", "js": "📜", "ts": "📜", "json": "📋", "md": "📝",
        "csv": "📊", "txt": "📄", "pdf": "📕", "sh": "⚙", "yml": "⚙",
        "yaml": "⚙", "toml": "⚙", "html": "🌐", "css": "🎨",
        "png": "🖼", "jpg": "🖼", "jpeg": "🖼", "gif": "🖼", "svg": "🖼", "webp": "🖼",
    }.get(ext, "📄")


def _render_files():
    files = st.session_state.files
    cur   = st.session_state.current_path
    root  = st.session_state._root_path

    if cur:
        rel = os.path.relpath(cur, root) if root else cur
        st.caption(f"📂 {rel}")

    dirs  = [f for f in files if f["type"] == "dir"]
    flist = [f for f in files if f["type"] == "file"]

    for f in dirs:
        if st.button(f"📁 {f['name']}", key=f"dir_{f['path']}", use_container_width=True):
            _send({"type": "get_files", "path": f["path"]})
            _drain_until({"files"}, timeout=2.0)
            st.rerun()

    for f in flist:
        icon = _file_icon(f["name"])
        size_str = f"  {f['size'] // 1024}KB" if f.get("size", 0) > 1024 else ""
        lbl = f"{icon} {f['name']}{size_str}"
        if st.button(lbl, key=f"file_{f['path']}", use_container_width=True):
            _send({"type": "open_file", "path": f["path"]})
            ev = _drain_until({"files", "file_content", "file_image", "error"}, timeout=2.0)
            if ev and ev.get("type") == "error":
                st.session_state.messages.append({"role": "system", "content": f"⚠ {ev.get('msg', 'error')}"})
            st.rerun()

    if cur and cur != root:
        if st.button("⬆ ..", key="go_up", use_container_width=True):
            _send({"type": "get_files", "path": str(Path(cur).parent)})
            _drain_until({"files"}, timeout=2.0)
            st.rerun()

    if not files:
        st.caption("_(empty workspace)_")

    # File viewer
    of = st.session_state.open_file
    if of:
        st.divider()
        fname = Path(of["path"]).name
        col_a, col_b = st.columns([5, 1])
        col_a.markdown(f"**{fname}**")
        if col_b.button("✕", key="close_file"):
            st.session_state.open_file = None
            st.rerun()
        if of.get("is_image"):
            try:
                st.image(of["path"])
            except Exception:
                st.caption("_(cannot display image)_")
        else:
            content = of.get("content", "")
            if len(content) > 6000:
                content = content[:6000] + "\n\n... (truncated)"
            ext = Path(of["path"]).suffix.lower()
            lang = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".json": "json", ".md": "markdown", ".sh": "bash",
                ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
                ".html": "html", ".css": "css", ".txt": "text",
            }.get(ext, "text")
            st.code(content, language=lang, wrap_lines=True)


def _render_activity():
    acts = st.session_state.activity
    if not acts:
        st.caption("_(no activity yet)_")
        return
    # Newest first, max 100 shown
    for a in reversed(acts[-100:]):
        kind   = a.get("kind", "")
        name   = a.get("name", "")
        detail = a.get("detail", "")
        ts     = a.get("ts", "")
        if kind == "tool":
            detail_clip = detail[:80] + "…" if len(detail) > 80 else detail
            e_name   = html.escape(name)
            e_detail = html.escape(detail_clip)
            e_ts     = html.escape(ts)
            st.markdown(
                f'<div style="font-size:12px;border-left:2px solid #3c3c3c;padding:2px 6px;margin:2px 0;">'
                f'🔧 <span style="color:#dcdcaa;">{e_name}</span>'
                + (f'<br><span style="color:#808080;font-size:11px;">{e_detail}</span>' if e_detail else "")
                + (f'<br><span style="color:#555;font-size:10px;">{e_ts}</span>' if e_ts else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        elif kind == "sep":
            st.markdown(
                '<hr style="border-color:#3c3c3c;margin:4px 0;">',
                unsafe_allow_html=True,
            )
        elif kind == "phase":
            st.markdown(
                f'<div style="font-size:11px;color:#808080;padding:1px 6px;">⚡ {html.escape(name)}</div>',
                unsafe_allow_html=True,
            )


def _render_history():
    info = st.session_state.history_info
    if not info:
        st.caption("พิมพ์ `/history` เพื่อโหลดประวัติจาก archive")
        return
    c1, c2 = st.columns(2)
    c1.metric("Turns",   info.get("turns", 0))
    c2.metric("Chars",   f'{info.get("total", 0):,}')
    topics = info.get("topics", [])
    if topics:
        st.markdown("**Topics:**")
        for t in topics[:10]:
            st.markdown(f"- {t}")
    loaded = info.get("loaded_pairs", 0)
    total  = info.get("total_pairs",  0)
    st.caption(f"Loaded {loaded} / {total} conversation pairs")


def _render_sidebar():
    with st.sidebar:
        tab_files, tab_act, tab_hist = st.tabs(["📁 Files", "⚡ Activity", "🕐 History"])
        with tab_files:
            _render_files()
        with tab_act:
            _render_activity()
        with tab_hist:
            _render_history()


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
def _render_header():
    model  = st.session_state.model or "–"
    up     = st.session_state.server_up
    online = st.session_state.online
    skill  = st.session_state.skill

    short = model.split("/")[-1] if "/" in model else model
    if len(short) > 38:
        short = short[:35] + "…"

    srv_dot = "🟢" if up     else "🔴"
    net_dot = "🌐" if online else "📴"

    skill_span = (
        f'<span style="background:#4ec9b0;color:#1e1e1e;font-size:11px;'
        f'padding:2px 8px;border-radius:10px;font-weight:600;margin-left:6px;">{skill}</span>'
        if skill else ""
    )

    c1, c2 = st.columns([2, 8])
    with c1:
        st.markdown("**ENDEAVOR**")
    with c2:
        st.markdown(
            f'<div style="display:flex;align-items:center;padding:4px 0;gap:6px;">'
            f'<span style="background:#2d2d2d;border:1px solid #569cd6;color:#569cd6;'
            f'font-size:11px;padding:2px 10px;border-radius:10px;">{short}</span>'
            f'{srv_dot} {net_dot}{skill_span}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Context bar
# ─────────────────────────────────────────────────────────────────────────────
def _render_ctx():
    chars = st.session_state.ctx_chars
    max_c = st.session_state.ctx_max or CONTEXT_MAX_CHARS
    pct   = min(chars / max_c, 1.0) if max_c else 0.0
    pct_i = int(pct * 100)
    col   = "#f48771" if pct > 0.9 else ("#dcb67a" if pct > 0.7 else "#4caf50")
    bar   = "▓" * int(pct * 20) + "░" * (20 - int(pct * 20))
    st.markdown(
        f'<div style="font-size:11px;color:#808080;padding:2px 4px;margin-bottom:4px;">'
        f'<span style="color:{col};">{bar} {pct_i}%</span>'
        f'  <span>{chars:,} / {max_c:,} chars</span></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chat messages
# ─────────────────────────────────────────────────────────────────────────────
def _render_messages():
    for msg in st.session_state.messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            st.markdown(
                f'<div style="text-align:center;color:#808080;font-size:12px;padding:6px 0;">{html.escape(content)}</div>',
                unsafe_allow_html=True,
            )
        elif role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Streaming turn (blocking — updates st.empty() in-place)
# ─────────────────────────────────────────────────────────────────────────────
def _stream_turn(query: str):
    """Send query to WS and stream response. Blocks until done/error/cancelled."""
    q = st.session_state._ws_q

    _send({"type": "query", "content": query})
    st.session_state.is_busy = True
    st.session_state.run_id  = ""

    # Placeholders — created before the while loop so they hold their position
    phase_ph = st.empty()
    with st.chat_message("assistant"):
        resp_ph = st.empty()

    tokens: list[str] = []
    final   = ""
    phase   = "thinking…"
    tick    = 0
    deadline = time.time() + 600  # 10-min safety cutoff

    while time.time() < deadline:
        try:
            ev = q.get(timeout=0.05)
        except _queue.Empty:
            tick += 1
            sp = _SPINNER[tick % len(_SPINNER)]
            phase_ph.markdown(
                f'<div style="font-size:12px;color:#4ec9b0;padding:2px 0;">{sp} {phase}</div>',
                unsafe_allow_html=True,
            )
            if tokens:
                resp_ph.markdown("".join(tokens) + "▋")
            continue

        typ = ev.get("type")

        if typ == "start":
            st.session_state.run_id = ev.get("run_id", "")

        elif typ == "phase":
            phase = ev.get("label", "") or phase
            tick += 1
            sp = _SPINNER[tick % len(_SPINNER)]
            phase_ph.markdown(
                f'<div style="font-size:12px;color:#4ec9b0;padding:2px 0;">{sp} {phase}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.activity.append(
                {"kind": "phase", "name": phase, "detail": "", "ts": _ts()}
            )

        elif typ == "progress":
            sub  = ev.get("msg", "")
            tick += 1
            sp   = _SPINNER[tick % len(_SPINNER)]
            phase_ph.markdown(
                f'<div style="font-size:12px;color:#4ec9b0;padding:2px 0;">{sp} {phase}</div>'
                f'<div style="font-size:11px;color:#808080;padding-left:10px;">→ {sub}</div>',
                unsafe_allow_html=True,
            )

        elif typ == "tool":
            name   = ev.get("name",   "")
            detail = ev.get("detail", "")
            st.session_state.activity.append(
                {"kind": "tool", "name": name, "detail": detail, "ts": _ts()}
            )
            if len(st.session_state.activity) > _MAX_ACTIVITY:
                st.session_state.activity = st.session_state.activity[-_MAX_ACTIVITY:]

        elif typ == "plan":
            steps = ev.get("steps", [])
            if steps:
                plan_md = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
                resp_ph.markdown(f"**📋 Plan:**\n\n{plan_md}\n\n---")

        elif typ == "token":
            tokens.append(ev.get("text", ""))
            resp_ph.markdown("".join(tokens) + "▋")

        elif typ == "discard_stream":
            tokens.clear()
            resp_ph.empty()

        elif typ == "response":
            final = ev.get("content", "")
            resp_ph.markdown(final)

        elif typ == "done":
            phase_ph.empty()
            st.session_state.messages.append(
                {"role": "assistant", "content": final or "_(ไม่มีคำตอบ)_"}
            )
            st.session_state.activity.append({"kind": "sep", "name": "", "detail": "", "ts": ""})
            break

        elif typ == "error":
            err = ev.get("msg", "unknown error")
            resp_ph.error(f"❌ {err}")
            st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {err}"})
            phase_ph.empty()
            break

        elif typ == "cancelled":
            resp_ph.warning("✕ ยกเลิกแล้ว")
            st.session_state.messages.append({"role": "system", "content": "✕ ยกเลิก"})
            phase_ph.empty()
            break

        elif typ == "_disconnect":
            _apply(ev)
            resp_ph.error("❌ WebSocket disconnected")
            phase_ph.empty()
            break

        else:
            # files, ctx_update, compact_result, skill_change, memory_ok, etc.
            _apply(ev)

    st.session_state.is_busy = False
    st.session_state.run_id  = ""


# ─────────────────────────────────────────────────────────────────────────────
# Command handler
# ─────────────────────────────────────────────────────────────────────────────
_CMD_EXPECT = {
    "/clear":   {"clear_ok", "error"},
    "/compact": {"compact_result", "error"},
    "/history": {"memory_ok", "error"},
}


def _run_command(cmd: str):
    _send({"type": "command", "cmd": cmd})
    expect = _CMD_EXPECT.get(cmd)
    if expect:
        _drain_until(expect, timeout=30.0)
    else:
        # Skill toggle: wait briefly
        time.sleep(0.4)
        _drain()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _ensure_ws()

    # Connection error banner
    if st.session_state.conn_error and st.session_state._ws_conn is None:
        st.error(
            f"❌ ไม่สามารถเชื่อมต่อ server ที่ `{_WS_URI}`\n\n"
            f"**Error:** `{st.session_state.conn_error}`\n\n"
            "เปิด server ก่อน:\n```bash\npython agent_server.py\n```"
        )
        if st.button("🔄 Reconnect"):
            st.session_state.conn_error = ""
            _connect()
            st.rerun()
        return

    # Drain queue (picks up initial status + files events from fresh connection)
    changed = _drain()

    # Layout
    _render_sidebar()
    _render_header()
    st.markdown('<hr style="border-color:#3c3c3c;margin:4px 0;">', unsafe_allow_html=True)
    _render_messages()
    _render_ctx()

    # Chat input
    user_in = st.chat_input(
        "พิมพ์ข้อความ… (/ สำหรับคำสั่ง: /clear /compact /history /research …)",
        disabled=st.session_state.is_busy,
    )

    if user_in:
        text  = user_in.strip()
        lower = text.lower()

        if lower.startswith("/"):
            # Command (built-in or skill toggle)
            st.session_state.messages.append({"role": "user", "content": text})
            _run_command(lower)
            st.rerun()
        else:
            # Conversational query → streaming
            with st.chat_message("user"):
                st.markdown(text)
            st.session_state.messages.append({"role": "user", "content": text})
            _stream_turn(text)
            st.rerun()

    elif changed:
        # Fresh connection: re-render to show status/files in sidebar
        time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()
