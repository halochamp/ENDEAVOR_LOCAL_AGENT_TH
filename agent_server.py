# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""agent_server.py — WebSocket + REST backend for ENDEAVOR Agent V2 UI

WebSocket  ws://localhost:8765/ws       — real-time chat + event stream (Electron)
POST       http://localhost:8765/chat   — sync request/response (Telegram, future)
GET        http://localhost:8765/status — health check
GET        http://localhost:8765/files  — list workspace files
GET        http://localhost:8765/file   — read file (?path=...)
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import stat
import sys
import threading
import uuid

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.callbacks import BaseCallbackHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_log import AgentLogger
from config import MLX_BASE_URL, MODEL, RECURSION_LIMIT, WORKSPACE, CONTEXT_MAX_CHARS, SERVER_PORT, AUTH_DISABLED
from graph import build_graph, force_compact, rewarm_after_compact, summarize_history
from react import get_system_prompt, ctx_stats as _ctx_stats
from runtime_common import (
    mlx_up as _mlx_up, internet_up as _internet_up,
    parse_plan_steps as _parse_plan_steps,
    extract_tool_content as _extract_tool_content,
    parse_tool_args as _parse_tool_args,
    guard_db_schema as _guard_db_schema, open_memory_store as _open_memory_store,
    load_memory_md as _load_memory_md, purge_thread as _purge_thread,
    scan_skill_roles as _scan_skill_roles, first_role_line as _first_role_line,
    ThinkingTimer, run_turn_core,
    _MEMORY_DB, _MEMORY_MD, _MEMORY_THREAD, _SKILLS_DIR, _DB_SCHEMA_VERSION,
    _MAX_DB_ROWS, load_history_pairs as _load_history_pairs,
)
from tools import ALL_TOOLS, SKILL_TOOLS
from tools._progress import (
    set_callback as set_progress_callback,
    set_phase_callback,
    set_plan_callback,
    set_run_callbacks,
    ToolCancelled,
)
from tools.scratchpad import _PAD as _scratch_pad
from tools.web_cache import web_count_reset as _reset_web_counter

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_WEB_TOOLS = {
    "web_search", "browse_url", "browser_use", "recall_web",
    "fetch_sitemap", "batch_browse", "scrape_table",
    "research_orchestrator",
}

PORT = SERVER_PORT

# Leading-JSON-echo filter (see _WSCallback.on_llm_new_token): if brace depth hasn't
# balanced within this many chars, it wasn't JSON — flush the held-back text instead
# of suppressing the whole response. Large enough that a multi-step plan/tool-result
# JSON echo still balances within the window; runaway prose that merely starts with
# "{" would not balance even at this size, so it still gets caught.
_JSON_GUARD_MAX_CHARS = 4000

# ── Auth (static token on every request) ──────────────────────────────────────
# A custom Electron/web UI generates a token and passes it via env AGENT_SERVER_TOKEN.
# Standalone runs (Telegram / headless) generate + persist a token file instead.
# AGENT_AUTH_DISABLED=1 turns auth off for browser dev (opening a UI directly).
_AUTH_DISABLED = AUTH_DISABLED
_TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agent_token")


def _load_or_create_token() -> str:
    env_tok = os.getenv("AGENT_SERVER_TOKEN")
    if env_tok and env_tok.strip():
        return env_tok.strip()
    try:
        if os.path.exists(_TOKEN_FILE):
            with open(_TOKEN_FILE, encoding="utf-8") as f:
                t = f.read().strip()
            if t:
                return t
        t = secrets.token_urlsafe(32)
        with open(_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(t)
        try:
            os.chmod(_TOKEN_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except Exception:
            pass
        return t
    except Exception:
        return secrets.token_urlsafe(32)  # ephemeral fallback — still better than no auth


_AUTH_TOKEN = _load_or_create_token()


_ALLOWED_ORIGINS = {f"http://127.0.0.1:{SERVER_PORT}", f"http://localhost:{SERVER_PORT}"}


def _origin_ok(origin: str | None) -> bool:
    """Reject cross-origin browser requests (DNS rebinding to 127.0.0.1).

    Browsers always send Origin on fetch()/WS; non-browser clients (curl, bots)
    typically omit it, so a missing header is allowed.
    """
    return origin is None or origin in _ALLOWED_ORIGINS


def _require_token(x_auth_token: str | None = Header(default=None)):
    """FastAPI dependency — reject REST requests without a valid X-Auth-Token header."""
    if _AUTH_DISABLED:
        return
    if not (x_auth_token and secrets.compare_digest(x_auth_token, _AUTH_TOKEN)):
        raise HTTPException(status_code=401, detail="unauthorized")

# ── Session state (single-session; Telegram can extend with session_id) ───────

def _load_skill_content(sname: str) -> str | None:
    if not sname or any(c in sname for c in ("/", "\\")) or ".." in sname:
        return None
    path = os.path.join(_SKILLS_DIR, f"{sname}.md")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _load_builtin_cmds() -> list[dict]:
    try:
        import json as _json
        with open(os.path.join(_SKILLS_DIR, "skill.json"), encoding="utf-8") as f:
            return _json.load(f).get("builtin_cmds", [])
    except Exception:
        return []


def _list_skills() -> list[dict]:
    return [{"name": n, "desc": role or "skill mode"} for n, role in _scan_skill_roles(_SKILLS_DIR)]


class _State:
    def __init__(self):
        self.thread_id = str(uuid.uuid4())
        self._db_conn, self._saver = _open_memory_store(_MEMORY_DB, verbose=False)
        self.online = _internet_up(retries=1)
        self._rebuild_app()
        self.skill = ""
        self.skill_content = ""
        self.logger = AgentLogger()

    def _get_skill_tools(self, skill: str) -> list:
        tools = SKILL_TOOLS.get(skill, [])
        if not self.online:
            return [t for t in tools if t.name not in _WEB_TOOLS]
        return tools

    def _rebuild_app(self, extra_tools: list | None = None):
        active_tools = [t for t in ALL_TOOLS if self.online or t.name not in _WEB_TOOLS]
        all_tools = active_tools + (extra_tools or [])
        self.app = build_graph(
            checkpointer=self._saver,
            memory=_load_memory_md(),
            tools=all_tools,
        )
        self.cfg = {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {"thread_id": self.thread_id},
        }

    def clear(self):
        if self.thread_id != _MEMORY_THREAD:
            _purge_thread(self._db_conn, self.thread_id)
        self.thread_id = str(uuid.uuid4())
        self.cfg = {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {"thread_id": self.thread_id},
        }

    def toggle_skill(self, sname: str) -> dict:
        sname = sname.lstrip("/").strip().lower()
        old_extra = self._get_skill_tools(self.skill)
        if sname == self.skill or sname in ("exit", "quit"):
            old = self.skill
            self.skill = ""
            self.skill_content = ""
            new_extra = self._get_skill_tools("")
            if old_extra != new_extra:
                self._rebuild_app(new_extra)
            return {"skill": "", "msg": f"ปิด {old} mode" if old else ""}
        content = _load_skill_content(sname)
        if content is None:
            return {"skill": self.skill, "msg": f"ไม่พบ skill '{sname}'", "error": True}
        self.skill = sname
        self.skill_content = content
        new_extra = self._get_skill_tools(sname)
        if old_extra != new_extra:
            self._rebuild_app(new_extra)
        return {"skill": sname, "msg": f"เปิด {sname} mode"}


_state = _State()
_busy = asyncio.Lock()


async def _ws_busy_guard(websocket: WebSocket, msg: str = "agent กำลังทำงานอยู่") -> bool:
    """Return True (and send an error frame) if _busy is held; caller should `continue`."""
    if _busy.locked():
        await websocket.send_json({"type": "error", "msg": msg})
        return True
    return False


# ── Tool detail helper ─────────────────────────────────────────────────────────

def _tool_detail(name: str, args: dict) -> str:
    def _strip(url: str, limit: int = 120) -> str:
        return url.replace("https://", "").replace("http://", "")[:limit]

    if name == "web_search":
        q = args.get("query", "")
        return f'"{q}"' if q else ""
    if name in ("browse_url", "recall_web", "browser_use"):
        url = args.get("url", "")
        uq = args.get("user_query", "") or args.get("task", "")
        base = _strip(url, 120)
        return f"{base}  [{uq[:60]}]" if uq else base
    if name == "fetch_sitemap":
        return _strip(args.get("url", ""), 120)
    if name == "batch_browse":
        urls = args.get("urls", [])
        if not isinstance(urls, list):
            urls = []
        n = len(urls)
        preview = " | ".join(_strip(u, 60) for u in urls[:3])
        suffix = f" +{n-3}" if n > 3 else ""
        return f"{n} URLs — {preview}{suffix}" if preview else f"{n} URLs"
    if name in ("read_file", "write_file", "edit"):
        p = args.get("file_path", "")
        parts = p.replace("\\", "/").split("/")
        return "/".join(parts[-3:]) if len(parts) >= 3 else p
    if name == "bash":
        cmd = args.get("command", "").strip()
        return cmd[:150]
    if name == "python_exec":
        code = args.get("code", "")
        lines = [l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
        return lines[0][:120] if lines else code[:120]
    if name == "grep":
        pattern = args.get("pattern", "")
        path = args.get("path", "")
        fname = path.split("/")[-1] if path else "."
        return f'"{pattern}" in {fname}'
    if name == "remember":
        return str(args.get("fact", ""))[:100]
    if name == "research_orchestrator":
        topic = args.get("topic", "")
        n = args.get("n", "")
        kw = args.get("keywords", "")
        kw_str = f"  [{kw[:60]}]" if kw else ""
        return f"{topic} ({n} sources){kw_str}" if n else topic
    if name == "tool_loop":
        action = args.get("action", "")
        n = len(args.get("items", []))
        ctx = args.get("context", "")
        return f"{action}  {n} items" + (f"  [{ctx[:40]}]" if ctx else "")
    if name == "scratch_write":
        key = args.get("key", "")
        val = str(args.get("value", ""))[:60]
        return f"[{key}] {val}" if key else val
    if name == "create_plan":
        return args.get("query", "")[:100]
    if name == "plot":
        return args.get("description", "")[:100]
    if name == "workspace_ls":
        return args.get("path", "") or "."
    if name == "scrape_table":
        return _strip(args.get("url", ""), 120)
    return ""


# ── WebSocket callback ─────────────────────────────────────────────────────────

def _response_has_tool_calls(response) -> bool:
    """Return True if an LLMResult's first generation contains tool calls."""
    try:
        msg = response.generations[0][0].message
        return bool(getattr(msg, "tool_calls", None) or msg.additional_kwargs.get("tool_calls"))
    except Exception:
        return False


class _CancelledError(BaseException):
    """Raised inside LangGraph callbacks when the user cancels a turn.
    Must extend BaseException (not Exception) so LangChain's handle_event
    cannot swallow it via `except Exception` even when raise_error=False."""


class _WSCallback(BaseCallbackHandler):
    raise_error = False

    def __init__(
        self,
        q: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        logger,
        turn_id: str,
        cancel_event: threading.Event | None = None,
    ):
        super().__init__()
        self._q = q
        self._loop = loop
        self._run_to_name: dict = {}
        self._timer = ThinkingTimer(emit=lambda label: self._put({"type": "phase", "label": label}))
        self._log = logger
        self._tid = turn_id
        self._cancel = cancel_event
        # Buffer tokens per LLM run_id. Only flushed to client after on_llm_end
        # confirms no tool calls — prevents pre-tool deliberation tokens from leaking.
        self._run_buffers: dict = {}     # str(run_id) -> int (token count)
        self._run_json_depth: dict = {}  # str(run_id) -> int (brace depth; >0 = inside JSON)
        self._run_json_done: dict = {}   # str(run_id) -> bool (True once leading JSON stripped)
        self._run_json_buf: dict = {}    # str(run_id) -> str (suppressed text, in case it's not JSON)

    def _check_cancel(self) -> None:
        if self._cancel and self._cancel.is_set():
            raise _CancelledError("turn cancelled by user")

    def _put(self, event: dict):
        self._loop.call_soon_threadsafe(self._q.put_nowait, event)

    def _cancel_timer(self) -> None:
        self._timer.cancel()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._timer.start()
        if self._timer.had_tool:
            self._put({"type": "synthesis_start"})

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._check_cancel()
        if not token:
            return
        run_key = str(kwargs.get("run_id", "default"))
        if run_key not in self._run_buffers:
            self._run_buffers[run_key] = 0
            self._run_json_depth[run_key] = 0
            self._run_json_buf[run_key] = ""
            # Detect leading JSON echo (e.g. create_plan result re-echoed by Qwen3)
            self._run_json_done[run_key] = token.lstrip()[:1] != "{"
            self._timer.cancel()  # LLM is generating — stop thinking timer

        self._run_buffers[run_key] += 1

        # JSON filter: suppress leading JSON block so prose reasoning shows but JSON doesn't
        if not self._run_json_done[run_key]:
            depth = self._run_json_depth[run_key]
            for ch in token:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
            self._run_json_depth[run_key] = depth
            if depth <= 0:
                self._run_json_done[run_key] = True  # JSON block closed; future tokens shown
                self._run_json_buf.pop(run_key, None)
                return
            buf = self._run_json_buf[run_key] + token
            if len(buf) <= _JSON_GUARD_MAX_CHARS:
                self._run_json_buf[run_key] = buf
                return  # suppress token while inside JSON
            # Brace never balanced within the guard window — not JSON after all,
            # flush what was held back and stop suppressing.
            self._run_json_done[run_key] = True
            self._run_json_buf.pop(run_key, None)
            self._put({"type": "token", "text": buf})
            return

        self._put({"type": "token", "text": token})

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._timer.cancel()
        run_key = str(run_id) if run_id else "default"
        token_count = self._run_buffers.pop(run_key, 0)
        self._run_json_depth.pop(run_key, None)
        self._run_json_done.pop(run_key, None)
        self._run_json_buf.pop(run_key, None)

        if _response_has_tool_calls(response):
            # Pre-tool deliberation — tokens already streamed but should not be shown
            if token_count > 0:
                self._put({"type": "discard_stream"})
            if self._timer.had_tool:
                self._put({"type": "phase", "label": "executing..."})
        # else: direct response / synthesis — tokens already streamed, nothing to do

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._timer.cancel()
        run_key = str(run_id) if run_id else "default"
        token_count = self._run_buffers.pop(run_key, 0)
        self._run_json_depth.pop(run_key, None)
        self._run_json_done.pop(run_key, None)
        self._run_json_buf.pop(run_key, None)
        if token_count > 0:
            self._put({"type": "discard_stream"})

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._check_cancel()
        self._timer.cancel()
        self._timer.mark_tool()
        name = serialized.get("name", "")
        args = _parse_tool_args(input_str, kwargs)
        self._run_to_name[str(run_id)] = name
        self._put({"type": "tool", "name": name, "detail": _tool_detail(name, args)})
        self._log.tool_call(name, args, self._tid)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._check_cancel()  # fire immediately after tool completes, before LLM processes result
        name = self._run_to_name.get(str(run_id), "")
        result_text = _extract_tool_content(output)
        self._log.tool_result(name, result_text, self._tid)
        if name == "create_plan":
            steps = _parse_plan_steps(result_text)
            if steps:
                self._loop.call_soon_threadsafe(self._q.put_nowait, {"type": "plan", "steps": steps})
        if self._timer.had_tool:
            self._put({"type": "phase", "label": "executing..."})

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._timer.cancel()
        if self._timer.had_tool:
            self._put({"type": "phase", "label": "executing..."})


# ── Agent runner (sync, runs in thread) ───────────────────────────────────────

def _run_agent_sync(
    query: str,
    q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    sub_cb=None,
    phase_cb=None,
    plan_cb=None,
    cancel_event: threading.Event | None = None,
) -> None:
    # Bind per-run callbacks as ContextVars so parallel tool threads (LangGraph
    # ToolNode uses ThreadPoolExecutor.submit which copies context) inherit them
    # without touching the module-level globals used by the REST /chat path.
    set_run_callbacks(
        sub_cb, phase_cb, plan_cb,
        cancel_check=(cancel_event.is_set if cancel_event else None),
    )
    actual_q = (
        f"[SKILL: {_state.skill}]\n{_state.skill_content}\n---\n{query}"
        if _state.skill else query
    )
    tid = str(uuid.uuid4())[:8]
    _state.logger.turn_start(query, tid)
    cb = _WSCallback(q, loop, _state.logger, tid, cancel_event)
    stream_cfg = {**_state.cfg, "callbacks": [cb]}
    try:
        final = run_turn_core(
            _state.app, actual_q, stream_cfg,
            thread_id=_state.thread_id, saver=_state._saver, db_conn=_state._db_conn,
            clear_scratch=_scratch_pad.clear, reset_web_counter=_reset_web_counter,
            on_final=lambda _: cb._put({"type": "phase", "label": "synthesizing…"}),
        )
        cb._cancel_timer()
        _state.logger.final_response(final or "", tid)
        loop.call_soon_threadsafe(
            q.put_nowait, {"type": "response", "content": final or "_(ไม่มีคำตอบ)_"}
        )
        loop.call_soon_threadsafe(q.put_nowait, {"type": "done"})
    except (_CancelledError, ToolCancelled):
        cb._cancel_timer()
        loop.call_soon_threadsafe(q.put_nowait, {"type": "cancelled"})
    except Exception as e:
        cb._cancel_timer()
        _state.logger.error(e, tid)
        log.exception("agent error")
        loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "msg": str(e)})
    finally:
        try:
            _state.logger.flush()
        except Exception:
            pass


# ── Workspace file helpers ─────────────────────────────────────────────────────

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg", ".tiff"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


_IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".ico": "image/x-icon", ".svg": "image/svg+xml", ".tiff": "image/tiff",
}

_IMAGE_MAX_BYTES = 8 * 1024 * 1024  # 8MB cap for inline base64 preview


def _read_image_data_url(path: str) -> str | None:
    """Read an image file and return it as a base64 data: URL, or None on failure/oversize."""
    try:
        if os.path.getsize(path) > _IMAGE_MAX_BYTES:
            return None
        ext = os.path.splitext(path)[1].lower()
        mime = _IMAGE_MIME.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            data = f.read()
        import base64
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    except Exception:
        return None


def _safe_real(path: str) -> str | None:
    """Return realpath only if inside WORKSPACE, else None."""
    ws = os.path.realpath(WORKSPACE)
    real = os.path.realpath(path)
    return real if (real == ws or real.startswith(ws + os.sep)) else None


def _list_dir(path: str) -> list[dict]:
    result = []
    try:
        for fname in sorted(os.listdir(path), key=str.lower):
            fpath = os.path.join(path, fname)
            is_dir = os.path.isdir(fpath)
            stat = os.stat(fpath)
            result.append({
                "name": fname,
                "path": fpath,
                "type": "dir" if is_dir else "file",
                "size": 0 if is_dir else stat.st_size,
                "mtime": stat.st_mtime,
            })
    except Exception:
        pass
    dirs  = sorted([e for e in result if e["type"] == "dir"],  key=lambda x: x["name"].lower())
    files = sorted([e for e in result if e["type"] == "file"], key=lambda x: x["mtime"], reverse=True)
    return dirs + files


def _list_workspace() -> list[dict]:
    return _list_dir(WORKSPACE)


def _read_file(path: str) -> str:
    real = _safe_real(path)
    if real is None:
        return "[error] Access denied"
    try:
        with open(real, encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"[error] {e}"


# ── FastAPI app ────────────────────────────────────────────────────────────────

api = FastAPI(title="ENDEAVOR Agent Server")
# No CORS middleware: browser same-origin policy then blocks cross-origin
# reads from drive-by sites. A custom UI uses token-gated WS/REST (see below),
# so neither needs CORS headers.


@api.get("/ui")
def get_ui():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html")
    return FileResponse(html_path, media_type="text/html")


@api.get("/ui-token")
def get_ui_token(request: Request):
    if not _origin_ok(request.headers.get("origin")):
        raise HTTPException(status_code=403, detail="forbidden origin")
    return {"token": _AUTH_TOKEN, "auth_disabled": _AUTH_DISABLED}


@api.get("/status", dependencies=[Depends(_require_token)])
def get_status():
    return {
        "server_up": _mlx_up(),
        "online": _internet_up(retries=1),
        "skill": _state.skill,
        "skills": _list_skills(),
        "builtin_cmds": _load_builtin_cmds(),
        "model": MODEL,
    }


@api.get("/files", dependencies=[Depends(_require_token)])
def get_files():
    return {"files": _list_workspace()}


@api.get("/file", dependencies=[Depends(_require_token)])
def get_file(path: str):
    return {"content": _read_file(path)}


@api.post("/chat", dependencies=[Depends(_require_token)])
async def post_chat(body: dict):
    """Sync endpoint for Telegram / future clients."""
    query = body.get("query", "")
    session_id = body.get("session_id", "default")
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    if _busy.locked():
        return JSONResponse({"error": "agent is busy"}, status_code=503)

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    set_progress_callback(lambda msg: loop.call_soon_threadsafe(q.put_nowait, {"type": "progress", "msg": msg}))
    set_phase_callback(lambda label: loop.call_soon_threadsafe(q.put_nowait, {"type": "phase", "label": label}))

    async with _busy:
        threading.Thread(target=_run_agent_sync, args=(query, q, loop), daemon=True).start()
        response = ""
        while True:
            event = await q.get()
            if event["type"] == "response":
                response = event["content"]
            elif event["type"] in ("done", "error"):
                if event["type"] == "error":
                    response = f"[error] {event['msg']}"
                break

    set_progress_callback(None)
    set_phase_callback(None)
    return {"response": response, "skill": _state.skill}


@api.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    # Token rides in Sec-WebSocket-Protocol (browser WS can't set arbitrary
    # headers, but it CAN offer a subprotocol) — keeps it out of access logs
    # (MN-6). ?token= stays as fallback for older clients/scripts.
    if not _origin_ok(websocket.headers.get("origin")):
        await websocket.close(code=1008)  # policy violation
        return
    _offered = (websocket.headers.get("sec-websocket-protocol") or "").split(",")[0].strip() or None
    if not _AUTH_DISABLED:
        tok = _offered or websocket.query_params.get("token", "")
        if not (tok and secrets.compare_digest(tok, _AUTH_TOKEN)):
            await websocket.close(code=1008)  # policy violation
            return
    # Must echo the offered subprotocol or the browser aborts the connection
    await websocket.accept(subprotocol=_offered)
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    # Dedicated WS reader: one persistent task owns websocket.receive_json().
    # The pump loop reads from ws_in_q instead, so cancelling ws_in_q.get() is
    # safe and never corrupts Starlette's internal WebSocket receive state.
    ws_in_q: asyncio.Queue = asyncio.Queue()

    async def _ws_reader():
        try:
            while True:
                await ws_in_q.put(await websocket.receive_json())
        except Exception:
            await ws_in_q.put(None)  # sentinel: connection closed / error

    ws_reader_task = asyncio.create_task(_ws_reader())

    # Per-connection progress lambdas — passed to each agent thread as args and
    # bound as ContextVars inside the thread (D-1). Never written to globals, so
    # a second concurrent WS connection (future) can't clobber another's callbacks.
    _ws_sub_cb = lambda msg: loop.call_soon_threadsafe(q.put_nowait, {"type": "progress", "msg": msg})
    _ws_phase_cb = lambda label: loop.call_soon_threadsafe(q.put_nowait, {"type": "phase", "label": label})

    def _ws_plan_cb(plan_text: str):
        steps = _parse_plan_steps(plan_text)
        loop.call_soon_threadsafe(q.put_nowait, {"type": "tool", "name": "create_plan", "detail": ""})
        if steps:
            loop.call_soon_threadsafe(q.put_nowait, {"type": "plan", "steps": steps})

    ws_root = os.path.realpath(WORKSPACE)
    await websocket.send_json({"type": "status", **get_status()})
    await websocket.send_json({
        "type": "files", "files": _list_workspace(),
        "path": ws_root, "root": ws_root,
    })

    try:
        while True:
            data = await ws_in_q.get()
            if data is None:
                break  # WS closed / reader error
            msg_type = data.get("type")

            if msg_type == "query":
                if await _ws_busy_guard(websocket): continue
                async with _busy:
                    run_id = str(uuid.uuid4())[:8]
                    cancel_event = threading.Event()
                    await websocket.send_json({"type": "start", "run_id": run_id})
                    threading.Thread(
                        target=_run_agent_sync,
                        args=(data.get("content", ""), q, loop,
                              _ws_sub_cb, _ws_phase_cb, _ws_plan_cb, cancel_event),
                        daemon=True,
                    ).start()
                    ws_alive = True
                    while True:
                        if ws_alive:
                            q_fut = asyncio.ensure_future(q.get())
                            ws_fut = asyncio.ensure_future(ws_in_q.get())
                            done, _ = await asyncio.wait(
                                {q_fut, ws_fut}, return_when=asyncio.FIRST_COMPLETED
                            )
                            # Handle incoming WS message (cancel or other command)
                            if ws_fut in done:
                                try:
                                    msg = ws_fut.result()
                                    if msg is None:  # WS closed mid-turn
                                        ws_alive = False
                                    elif (msg.get("type") == "command" and
                                            msg.get("cmd") == "cancel" and
                                            msg.get("run_id") == run_id):
                                        cancel_event.set()
                                except Exception:
                                    ws_alive = False
                            else:
                                ws_fut.cancel()
                                try: await ws_fut
                                except (asyncio.CancelledError, Exception): pass
                            # Handle queue event
                            if q_fut in done:
                                event = q_fut.result()
                                if ws_alive:
                                    try:
                                        ev_type = event["type"]
                                        if cancel_event.is_set() and ev_type == "token":
                                            pass  # swallow post-cancel tokens
                                        elif cancel_event.is_set() and ev_type == "done":
                                            await websocket.send_json({"type": "cancelled"})
                                        else:
                                            await websocket.send_json(event)
                                    except Exception:
                                        ws_alive = False
                                if event["type"] in ("done", "error", "cancelled"):
                                    break
                            else:
                                q_fut.cancel()
                                try: await q_fut
                                except (asyncio.CancelledError, Exception): pass
                        else:
                            # WS gone — drain queue until agent thread finishes
                            event = await q.get()
                            if event["type"] in ("done", "error", "cancelled"):
                                break
                    if not ws_alive:
                        return  # WS is gone; skip post-turn file/ctx_update sends
                ws_root = os.path.realpath(WORKSPACE)
                await websocket.send_json({
                    "type": "files", "files": _list_workspace(),
                    "path": ws_root, "root": ws_root,
                })
                compact_n = _ctx_stats.get("compact_msg")
                if compact_n:
                    _ctx_stats["compact_msg"] = None
                    await websocket.send_json({
                        "type": "compact_result",
                        "cut": compact_n,
                        "before": _ctx_stats.get("compact_before", 0),
                        "after": _ctx_stats.get("chars", 0),
                    })
                await websocket.send_json({
                    "type": "ctx_update",
                    "chars": _ctx_stats.get("chars", 0),
                    "max_chars": CONTEXT_MAX_CHARS,
                })

            elif msg_type == "command":
                cmd = data.get("cmd", "").strip().lower()
                if cmd == "/clear":
                    if await _ws_busy_guard(websocket): continue
                    async with _busy:
                        _state.clear()
                        _ctx_stats["chars"] = 0
                    await websocket.send_json({"type": "clear_ok"})
                    await websocket.send_json({
                        "type": "ctx_update", "chars": 0,
                        "max_chars": CONTEXT_MAX_CHARS,
                    })
                elif cmd == "/compact":
                    if await _ws_busy_guard(websocket): continue
                    async with _busy:
                        result = await asyncio.get_running_loop().run_in_executor(
                            None, force_compact, _state.app, _state.cfg
                        )
                    if "error" in result:
                        await websocket.send_json({"type": "compact_result", "error": result["error"]})
                    else:
                        _ctx_stats["chars"] = result["after"]
                        await websocket.send_json({
                            "type": "compact_result",
                            "cut": result["cut"],
                            "before": result["before"],
                            "after": result["after"],
                        })
                        await websocket.send_json({
                            "type": "ctx_update",
                            "chars": result["after"],
                            "max_chars": CONTEXT_MAX_CHARS,
                        })
                        _seed_msgs = list(_state.app.get_state(_state.cfg).values.get("messages", []))
                        threading.Thread(
                            target=rewarm_after_compact,
                            args=(_state.app, _state._db_conn, _seed_msgs),
                            daemon=True,
                        ).start()
                elif cmd == "/history":
                    if await _ws_busy_guard(websocket): continue
                    def _do_history():
                        # Blocking: sqlite reads + update_state + LLM topic summary —
                        # must not run on the event loop (freezes every connection).
                        loaded_msgs, loaded_chars, total_pairs = _load_history_pairs(_state._saver)
                        if loaded_msgs:
                            _state.app.update_state(_state.cfg, {"messages": loaded_msgs})
                        loaded_pairs = sum(1 for m in loaded_msgs if hasattr(m, "type") and m.type == "human")
                        _hist_cfg = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": _MEMORY_THREAD}}
                        hist = summarize_history(_state.app, _hist_cfg)
                        return loaded_pairs, loaded_chars, total_pairs, hist
                    async with _busy:
                        try:
                            loaded_pairs, loaded_chars, total_pairs, hist = (
                                await asyncio.get_running_loop().run_in_executor(None, _do_history)
                            )
                        except Exception as e:
                            await websocket.send_json({"type": "error", "msg": f"/history ล้มเหลว: {e}"})
                            continue
                    await websocket.send_json({
                        "type": "memory_ok",
                        "total": hist["total"],
                        "turns": hist["turns"],
                        "total_pairs": total_pairs,
                        "loaded_pairs": loaded_pairs,
                        "loaded_chars": loaded_chars,
                        "topics": hist["topics"],
                    })
                elif cmd in ("/status", "status"):
                    await websocket.send_json({"type": "status", **get_status()})
                elif cmd == "cancel":
                    pass  # stale cancel — agent already finished; safe to ignore
                else:
                    result = _state.toggle_skill(cmd)
                    await websocket.send_json({"type": "skill_change", **result})

            elif msg_type == "get_files":
                req_path = data.get("path", WORKSPACE)
                real = _safe_real(req_path) or os.path.realpath(WORKSPACE)
                ws_root = os.path.realpath(WORKSPACE)
                await websocket.send_json({
                    "type": "files", "files": _list_dir(real),
                    "path": real, "root": ws_root,
                })

            elif msg_type == "open_file":
                path = data.get("path", "")
                real = _safe_real(path)
                if real is None:
                    await websocket.send_json({"type": "error", "msg": "Access denied"})
                elif os.path.isdir(real):
                    ws_root = os.path.realpath(WORKSPACE)
                    await websocket.send_json({
                        "type": "files", "files": _list_dir(real),
                        "path": real, "root": ws_root,
                    })
                elif _is_image(real):
                    data_url = await asyncio.get_running_loop().run_in_executor(
                        None, _read_image_data_url, real
                    )
                    if data_url is None:
                        await websocket.send_json({"type": "error", "msg": "เปิดรูปไม่ได้ (ไฟล์ใหญ่เกินไปหรืออ่านไม่ได้)"})
                    else:
                        await websocket.send_json({"type": "file_image", "path": real, "data_url": data_url})
                else:
                    await websocket.send_json({
                        "type": "file_content", "path": real,
                        "content": _read_file(real),
                    })

            elif msg_type == "delete_file":
                path = data.get("path", "")
                real = _safe_real(path)
                if real is None or os.path.isdir(real):
                    await websocket.send_json({"type": "error", "msg": "ลบไม่ได้: path ไม่ถูกต้อง"})
                else:
                    try:
                        os.remove(real)
                        ws_root = os.path.realpath(WORKSPACE)
                        parent = os.path.dirname(real)
                        await websocket.send_json({
                            "type": "files", "files": _list_dir(parent),
                            "path": parent, "root": ws_root,
                            "deleted": real,
                        })
                    except Exception as e:
                        await websocket.send_json({"type": "error", "msg": f"ลบไม่ได้: {e}"})

            elif msg_type == "open_with_os":
                import subprocess
                path = data.get("path", "")
                real = _safe_real(path)
                if real is None:
                    await websocket.send_json({"type": "error", "msg": "Access denied"})
                else:
                    try:
                        subprocess.Popen(["open", real])
                        await websocket.send_json({"type": "open_with_os_ok", "path": real})
                    except Exception as e:
                        await websocket.send_json({"type": "error", "msg": f"เปิดไม่ได้: {e}"})

            elif msg_type == "get_status":
                await websocket.send_json({"type": "status", **get_status()})

            elif msg_type == "get_history":
                # Open a fresh read-only connection to avoid thread-safety issues
                # with the shared _state._saver connection. Must close it in finally —
                # in WAL mode an unclosed reader connection blocks WAL checkpointing,
                # so the .wal file grows for every history-panel open over a session.
                fresh_conn = None
                try:
                    fresh_conn, fresh_saver = _open_memory_store(_MEMORY_DB, verbose=False)
                    msgs, _, total = _load_history_pairs(fresh_saver, max_chars=999_999)
                    pairs, i = [], 0
                    while i < len(msgs):
                        m = msgs[i]
                        if hasattr(m, "type") and m.type == "human":
                            q_text = str(m.content or "").strip()
                            a_text = ""
                            ts = None
                            if i + 1 < len(msgs) and hasattr(msgs[i + 1], "type") and msgs[i + 1].type == "ai":
                                a_msg = msgs[i + 1]
                                a_text = str(a_msg.content or "").strip()
                                ts = (a_msg.additional_kwargs or {}).get("saved_at")
                                i += 1
                            if q_text:
                                pairs.append({"q": q_text, "a": a_text, "ts": ts})
                        i += 1
                    await websocket.send_json({"type": "history_list", "pairs": pairs, "total": total})
                except Exception as e:
                    log.warning(f"get_history error: {e}")
                    await websocket.send_json({"type": "history_list", "pairs": [], "total": 0})
                finally:
                    if fresh_conn is not None:
                        try:
                            fresh_conn.close()
                        except Exception:
                            pass

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    finally:
        ws_reader_task.cancel()
        try:
            await ws_reader_task
        except (asyncio.CancelledError, Exception):
            pass


if __name__ == "__main__":
    log.info(f"Starting ENDEAVOR agent server on port {PORT}")
    if _AUTH_DISABLED:
        log.warning("AUTH DISABLED (AGENT_AUTH_DISABLED=1) — dev only, do not run alongside a browser")
    elif not os.getenv("AGENT_SERVER_TOKEN"):
        log.info(f"Auth token persisted at {_TOKEN_FILE} (0600) — clients must send X-Auth-Token / ?token=")
    uvicorn.run(api, host="127.0.0.1", port=PORT, log_level="warning")
