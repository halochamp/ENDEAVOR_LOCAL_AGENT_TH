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
from collections import deque
import json
import logging
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from langchain_core.callbacks import BaseCallbackHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_log import AgentLogger
from awake_engine import AwakeEngine, Registry, drain_notices, restore_notices
import config as _config
from config import (
    RECURSION_LIMIT, WORKSPACE, CONTEXT_MAX_CHARS, SERVER_PORT, AUTH_DISABLED,
    READ_FILE_MAX_BYTES, READ_FILE_AUDIO_VIDEO_MAX_BYTES,
    get_model, get_thinking_budget, get_runtime_settings, set_runtime_settings,
    get_mlx_base_url, get_server_mode, get_server_port,
)
from model_runtime import (
    reconcile_runtime_server as _reconcile_runtime_server,
    model_server_status as _model_server_status,
    shared_max_server_status as _shared_max_server_status,
    auto_attach_max_test_server_if_present as _auto_attach_max_test_server_if_present,
    start_owner_model_server as _start_owner_model_server,
    stop_owner_model_server as _stop_owner_model_server,
    restart_owner_model_server as _restart_owner_model_server,
    stop_model_server as _stop_owned_model_server,
    get_model_server_control_settings as _get_model_server_control_settings,
    set_model_server_watchdog as _set_model_server_watchdog,
)
import graph as _graph
import planner as _planner
from graph import build_graph, force_compact, rewarm_after_compact, summarize_history
from react import get_system_prompt, ctx_stats as _ctx_stats
from runtime_model import (
    active_local_mlx_model as _active_local_mlx_model,
    model_from_process_command as _model_from_process_command,
)
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
from tools import _summarize as _summarize_tool
from tools._progress import (
    set_callback as set_progress_callback,
    set_phase_callback,
    set_plan_callback,
    set_run_callbacks,
    ToolCancelled,
)
from tools._safety import resolve_path as _resolve_write_path, check_path as _check_write_path
from tools._transcribe import _AUDIO_EXT, _VIDEO_EXT
from tools.read_image import _KNOWN_IMG_EXTS
from tools.web_cache import web_count_reset as _reset_web_counter

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Desktop telemetry is intentionally self-contained in this public project.
# psutil is part of the documented install set, but keep collection fail-soft so
# the Agent remains usable even in a minimal/custom environment.
try:
    import psutil as _psutil
    _HAS_PSUTIL = True
    _psutil.cpu_percent(interval=None)  # prime the non-blocking CPU baseline
    _SYSTEM_RAM_TOTAL = int(_psutil.virtual_memory().total)
except Exception:
    _psutil = None
    _HAS_PSUTIL = False
    _SYSTEM_RAM_TOTAL = 0

try:
    _SYSTEM_TELEMETRY_INTERVAL_SECONDS = max(
        1.0, float(os.getenv("V2_SYSTEM_TELEMETRY_INTERVAL_SECONDS", "5")),
    )
except (TypeError, ValueError):
    _SYSTEM_TELEMETRY_INTERVAL_SECONDS = 5.0
try:
    _SYSTEM_TELEMETRY_PUBLISH_SECONDS = max(
        0.25, float(os.getenv("V2_SYSTEM_TELEMETRY_PUBLISH_SECONDS", "1")),
    )
except (TypeError, ValueError):
    _SYSTEM_TELEMETRY_PUBLISH_SECONDS = 1.0

_system_net_prev: tuple[int, int, float] | None = None
_system_net_lock = threading.Lock()
_system_telemetry_task: asyncio.Task | None = None

_GENERATION_RATE_WINDOW_SECONDS = 5.0
_generation_token_times: deque[float] = deque()
_generation_active_runs: set[str] = set()
_generation_token_lock = threading.Lock()

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
# Electron generates a token and passes it via env AGENT_SERVER_TOKEN; explicit custom clients may provide one too.
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


_OPAQUE_ORIGINS = {"null", "file://"}


def _origin_ok(origin: str | None) -> bool:
    """Reject cross-origin browser requests (DNS rebinding to 127.0.0.1).

    Browsers always send Origin on fetch()/WS; non-browser clients (curl, bots)
    typically omit it, so a missing header is allowed. A page loaded from
    file:// (AGENT_UI's Electron renderer) has an opaque origin — live-reproduced
    2026-08-16: every WS connection from AGENT_UI got HTTP 403'd here, the
    loading screen stuck forever (confirmed via Chrome DevTools Protocol
    against the actual running renderer, not assumed): this Electron build
    sends the literal string "file://", not the spec-standard "null" a regular
    browser would send for the same opaque-origin case — first fixing only
    "null" was verified insufficient live before landing this. Allowing either
    doesn't weaken the DNS-rebinding protection this exists for — that attack
    needs a real attacker-controlled domain as Origin, never one of these two
    literal opaque-origin strings.
    """
    return origin is None or origin in _OPAQUE_ORIGINS or origin in _ALLOWED_ORIGINS


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


def _active_mlx_model() -> str:
    """Resolve active local model from the current owner/shared endpoint."""
    return _active_local_mlx_model(get_mlx_base_url())


_VLM_SWITCH_TIMEOUT_SECONDS = 240.0
_MODEL_WATCHDOG_INTERVAL_SECONDS = 5.0
_MODEL_WATCHDOG_RETRY_SECONDS = 20.0
_model_watchdog_task: asyncio.Task | None = None
_model_server_action_lock = asyncio.Lock()


def _runtime_settings_payload(
    *,
    switch_state: str = "idle",
    error: str = "",
    confirmation_required: bool = False,
    ram_gb: int = 0,
) -> dict:
    state = str(switch_state or "idle")
    if error and state == "idle":
        state = "error"
    return {
        "type": "runtime_settings",
        **get_runtime_settings(),
        "switching": state == "switching",
        "switch_state": state,
        "confirmation_required": bool(confirmation_required),
        "ram_gb": int(ram_gb or 0),
        "error": str(error or ""),
    }


def _model_server_progress_payload(action_state: str) -> dict:
    control = _get_model_server_control_settings()
    shared = get_server_mode() == "shared_max"
    return {
        "type": "model_server_status",
        "owner": "agent_max_vlm" if shared else "agent_th",
        "mode": get_server_mode(),
        "managed_by_th": not shared,
        "port": get_server_port(),
        "selected_model": get_model(),
        "watchdog_enabled": False if shared else bool(control.get("watchdog_enabled", True)),
        "desired_state": "shared" if shared else str(control.get("desired_state") or "running"),
        "action_state": str(action_state or ""),
        "error": "",
    }


async def _model_server_status_payload(*, error: str = "", action_state: str = "") -> dict:
    try:
        status = await asyncio.to_thread(_model_server_status)
    except Exception as exc:
        shared = get_server_mode() == "shared_max"
        control = _get_model_server_control_settings()
        status = {
            "owner": "agent_max_vlm" if shared else "agent_th",
            "mode": get_server_mode(),
            "managed_by_th": not shared,
            "port": get_server_port(),
            "selected_model": get_model(),
            "watchdog_enabled": False if shared else bool(control.get("watchdog_enabled", True)),
            "desired_state": "shared" if shared else str(control.get("desired_state") or "running"),
            "running": False,
            "owned": False,
            "healthy": False,
            "state": "error",
            "loaded_model": "",
            "error": str(exc),
        }
    if error:
        status["error"] = str(error)
    return {"type": "model_server_status", **status, "action_state": str(action_state or "")}


def _rebuild_runtime_llms() -> None:
    """Drop every cached model-bound client, then rebuild the active agent graph."""
    _graph.invalidate_llm_cache()
    _planner.invalidate_llm_cache()
    _summarize_tool.invalidate_llm_cache()
    _state._rebuild_app(_state._get_skill_tools(_state.skill))


def _sync_runtime_settings_from_owner_file_if_idle() -> bool:
    """Adopt CLI-written owner settings before the next idle Desktop action."""
    if _busy.locked():
        return False
    if not _config.refresh_runtime_settings_from_file():
        return False
    try:
        status = _model_server_status()
        if status.get("healthy") and get_server_mode() == "shared_max":
            _config.adopt_shared_model(str(status.get("loaded_model") or ""))
    except Exception:
        pass
    _rebuild_runtime_llms()
    log.info(
        "runtime settings synced model=%s budget=%s mode=%s port=%s",
        get_model(), get_thinking_budget(), get_server_mode(), get_server_port(),
    )
    return True


async def _apply_model_server_action(action: str) -> dict:
    action = str(action or "").strip().lower()
    if action not in {"start", "stop", "reset"}:
        return await _model_server_status_payload(error=f"unsupported model server action: {action}")
    if _busy.locked():
        return await _model_server_status_payload(error="agent is busy")
    if _model_server_action_lock.locked():
        return await _model_server_status_payload(error="model server action is busy")

    async with _model_server_action_lock:
        fn = {
            "start": _start_owner_model_server,
            "stop": _stop_owner_model_server,
            "reset": _restart_owner_model_server,
        }[action]
        try:
            if action == "stop":
                await asyncio.to_thread(fn)
            else:
                await asyncio.to_thread(fn, _VLM_SWITCH_TIMEOUT_SECONDS)
        except Exception as exc:
            return await _model_server_status_payload(error=f"{action} failed: {exc}")
    return await _model_server_status_payload()


async def _apply_model_server_watchdog(enabled: bool) -> dict:
    try:
        await asyncio.to_thread(_set_model_server_watchdog, bool(enabled))
    except Exception as exc:
        return await _model_server_status_payload(error=f"watchdog update failed: {exc}")
    return await _model_server_status_payload()


async def _apply_runtime_settings(
    model: str,
    thinking_budget: int,
    server_mode: str,
    server_port: int,
    *,
    confirmed_low_ram: bool = False,
) -> dict:
    """Apply Agent TH owner settings or attach to MAX's read-only test server."""
    if _busy.locked():
        return _runtime_settings_payload(error="agent is busy")
    if _model_server_action_lock.locked():
        return _runtime_settings_payload(error="model server action is busy")

    requested_model = str(model or "").strip()
    requested_mode = str(server_mode or "standalone").strip()
    try:
        requested_port = int(server_port)
        requested_budget = int(thinking_budget)
    except (TypeError, ValueError):
        return _runtime_settings_payload(error="invalid runtime settings")

    old = get_runtime_settings()
    old_model = str(old.get("model") or get_model())
    old_budget = int(old.get("thinking_budget") or get_thinking_budget())
    old_mode = str(old.get("server_mode") or get_server_mode())
    old_port = int(old.get("server_port") or get_server_port())
    old_standalone_model = str(old.get("standalone_model") or _config.DEFAULT_MODEL)
    if requested_mode == "standalone" and old_mode == "shared_max":
        requested_model = old_standalone_model

    if (
        requested_mode == "standalone"
        and _config.high_quality_model_warning_required(requested_model)
        and not confirmed_low_ram
    ):
        ram_bytes = _config.physical_memory_bytes()
        ram_gb = max(1, round(ram_bytes / (1024 ** 3))) if ram_bytes else 0
        return _runtime_settings_payload(
            confirmation_required=True,
            ram_gb=ram_gb,
        )

    control = _get_model_server_control_settings()
    desired_running = control.get("desired_state") == "running"

    async with _model_server_action_lock:
        try:
            if requested_mode == "shared_max":
                shared = await asyncio.to_thread(
                    _shared_max_server_status, port=requested_port,
                )
                if not shared.get("healthy"):
                    return _runtime_settings_payload(
                        error=shared.get("error") or "MAX test server is unavailable"
                    )
                requested_model = str(shared.get("loaded_model") or "")
                if old_mode == "standalone" and old_port != requested_port:
                    old_status = await asyncio.to_thread(_model_server_status)
                    if old_status.get("running") and old_status.get("owned"):
                        await asyncio.to_thread(_stop_owned_model_server, port=old_port)
                changed = set_runtime_settings(
                    model=requested_model,
                    thinking_budget=requested_budget,
                    server_mode="shared_max",
                    server_port=requested_port,
                )
                _config.adopt_shared_model(requested_model)
            else:
                changed = set_runtime_settings(
                    model=requested_model,
                    thinking_budget=requested_budget,
                    server_mode="standalone",
                    server_port=requested_port,
                )
                server_changed = any(changed.get(key) for key in (
                    "model_changed", "server_mode_changed", "server_port_changed",
                ))
                if desired_running and server_changed:
                    if old_mode == "standalone":
                        await asyncio.to_thread(
                            _restart_owner_model_server,
                            _VLM_SWITCH_TIMEOUT_SECONDS,
                            previous_port=old_port,
                        )
                    else:
                        await asyncio.to_thread(
                            _start_owner_model_server,
                            _VLM_SWITCH_TIMEOUT_SECONDS,
                        )
        except Exception as exc:
            rollback_error = ""
            try:
                set_runtime_settings(
                    model=old_model,
                    thinking_budget=old_budget,
                    server_mode=old_mode,
                    server_port=old_port,
                )
                if old_mode == "shared_max":
                    _config.adopt_shared_model(old_model)
                elif desired_running:
                    await asyncio.to_thread(
                        _start_owner_model_server,
                        _VLM_SWITCH_TIMEOUT_SECONDS,
                    )
            except Exception as rollback_exc:
                rollback_error = f"; rollback failed: {rollback_exc}"
            return _runtime_settings_payload(
                switch_state="error",
                error=f"runtime/server switch failed: {exc}{rollback_error}",
            )

        if any(changed.get(key) for key in (
            "model_changed", "thinking_budget_changed", "server_mode_changed", "server_port_changed",
        )):
            _rebuild_runtime_llms()
            log.info(
                "runtime settings applied model=%s budget=%s mode=%s port=%s",
                get_model(), get_thinking_budget(), get_server_mode(), get_server_port(),
            )

    return _runtime_settings_payload(
        switch_state="ready"
        if any(changed.get(key) for key in ("model_changed", "server_mode_changed", "server_port_changed"))
        else "idle"
    )


async def _model_server_watchdog_loop() -> None:
    """Recover only a missing Agent TH-owned standalone server."""
    retry_at = 0.0
    loop = asyncio.get_running_loop()
    while True:
        try:
            await asyncio.sleep(_MODEL_WATCHDOG_INTERVAL_SECONDS)
            if get_server_mode() != "standalone":
                continue
            control = _get_model_server_control_settings()
            if not control.get("watchdog_enabled", True) or control.get("desired_state") != "running":
                continue
            if _busy.locked() or _model_server_action_lock.locked():
                continue
            status = await asyncio.to_thread(_model_server_status)
            if status.get("running"):
                continue
            now = loop.time()
            if now < retry_at:
                continue
            retry_at = now + _MODEL_WATCHDOG_RETRY_SECONDS
            async with _model_server_action_lock:
                latest = _get_model_server_control_settings()
                if get_server_mode() != "standalone":
                    continue
                if not latest.get("watchdog_enabled", True) or latest.get("desired_state") != "running":
                    continue
                try:
                    await asyncio.to_thread(
                        _reconcile_runtime_server,
                        _VLM_SWITCH_TIMEOUT_SECONDS,
                    )
                    retry_at = 0.0
                    log.info("Standalone Agent TH model-server watchdog recovered owner server")
                except Exception as exc:
                    log.warning("Standalone Agent TH model-server recovery failed: %s", exc)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Agent TH model-server watchdog iteration failed")


def _generation_run_start(run_id) -> None:
    key = str(run_id or "default")
    with _generation_token_lock:
        # One Agent turn can contain several adjacent LLM runs. Keep one true
        # process-level rolling 5s window across those run boundaries; samples
        # expire only by timestamp, never because another run starts.
        _generation_active_runs.add(key)


def _generation_run_end(run_id) -> None:
    key = str(run_id or "default")
    with _generation_token_lock:
        _generation_active_runs.discard(key)


def _generation_run_cleanup(run_ids) -> None:
    with _generation_token_lock:
        for run_id in run_ids:
            _generation_active_runs.discard(str(run_id or "default"))


def _mark_generation_token(now: float | None = None) -> None:
    stamp = time.monotonic() if now is None else float(now)
    cutoff = stamp - _GENERATION_RATE_WINDOW_SECONDS
    with _generation_token_lock:
        _generation_token_times.append(stamp)
        while _generation_token_times and _generation_token_times[0] < cutoff:
            _generation_token_times.popleft()


def _generation_tokens_per_second(now: float | None = None) -> float:
    """Return the current model's real streamed-token rate over a rolling 5s window.

    The meter counts generation callbacks, not characters and not a model-specific
    tokenizer. mlx/OpenAI-compatible streaming emits one callback for each output
    token/chunk produced by the active model; reasoning/tool-call chunks are counted
    too when the callback carries generation payload. No model ID is assumed.
    """
    stamp = time.monotonic() if now is None else float(now)
    cutoff = stamp - _GENERATION_RATE_WINDOW_SECONDS
    with _generation_token_lock:
        while _generation_token_times and _generation_token_times[0] < cutoff:
            _generation_token_times.popleft()
        return len(_generation_token_times) / _GENERATION_RATE_WINDOW_SECONDS


def _callback_has_generation_payload(token: str, kwargs: dict) -> bool:
    if token:
        return True
    chunk = kwargs.get("chunk")
    message = getattr(chunk, "message", None)
    if message is None:
        return False
    content = getattr(message, "content", None)
    if content:
        return True
    extra = getattr(message, "additional_kwargs", None) or {}
    if extra.get("reasoning_content") or extra.get("reasoning"):
        return True
    return bool(
        getattr(message, "tool_call_chunks", None)
        or getattr(message, "tool_calls", None)
    )


def _portable_ram_used_bytes() -> int | None:
    """Best-effort host RAM usage without machine-specific constants.

    On macOS, ``vm_stat`` includes unified-memory pressure that ``psutil.used``
    may under-report for MLX/Metal workloads. Other platforms fall back to the
    portable psutil value. Every executable is capability-discovered at runtime;
    no private path or host identity is assumed.
    """
    if sys.platform == "darwin":
        vm_stat = shutil.which("vm_stat")
        if vm_stat:
            try:
                out = subprocess.check_output([vm_stat], text=True, timeout=2)
                page_size = 4096
                match = re.search(r"page size of (\d+) bytes", out)
                if match:
                    page_size = int(match.group(1))

                def pages(pattern: str) -> int:
                    m = re.search(pattern, out)
                    return int(m.group(1).replace(",", "").replace(".", "")) if m else 0

                anonymous = pages(r"Anonymous pages:\s+([\d,.]+)")
                wired = pages(r"Pages wired down:\s+([\d,.]+)")
                occupied = pages(r"Pages occupied by compressor:\s+([\d,.]+)")
                free = pages(r"Pages free:\s+([\d,.]+)")
                speculative = pages(r"Pages speculative:\s+([\d,.]+)")
                active = pages(r"Pages active:\s+([\d,.]+)")
                inactive = pages(r"Pages inactive:\s+([\d,.]+)")
                throttled = pages(r"Pages throttled:\s+([\d,.]+)")
                total_pages = _SYSTEM_RAM_TOTAL // page_size if _SYSTEM_RAM_TOTAL else 0
                accounted = free + speculative + active + inactive + wired + occupied + throttled
                unaccounted = max(0, total_pages - accounted)
                return (anonymous + wired + occupied + unaccounted) * page_size
            except Exception:
                pass
    if _HAS_PSUTIL and _psutil is not None:
        try:
            return int(_psutil.virtual_memory().used)
        except Exception:
            pass
    return None


def _portable_gpu_percent() -> float | None:
    """Best-effort GPU utilisation using only locally available platform tools.

    Apple Silicon uses ``ioreg`` when present; NVIDIA hosts use ``nvidia-smi``.
    Unsupported systems simply report ``None`` so this public UI degrades to
    ``GPU —`` instead of assuming a particular machine or external monitor.
    """
    if sys.platform == "darwin":
        ioreg = shutil.which("ioreg")
        if ioreg:
            for cls in ("AGXAccelerator", "IOAccelerator"):
                try:
                    out = subprocess.check_output(
                        [ioreg, "-r", "-c", cls], text=True, timeout=2,
                    )
                    m = re.search(r'"Device Utilization %"\s*=\s*(\d+(?:\.\d+)?)', out)
                    if m:
                        return float(m.group(1))
                except Exception:
                    continue

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            )
            values = [float(line.strip()) for line in out.splitlines() if line.strip()]
            if values:
                return max(values)
        except Exception:
            pass
    return None


def _collect_system_telemetry() -> dict:
    """Collect one portable host-telemetry snapshot for the Electron status bar."""
    global _system_net_prev
    data = {
        "type": "system_telemetry",
        "cpu_percent": None,
        "gpu_percent": None,
        "ram_percent": None,
        "ram_used_bytes": None,
        "ram_total_bytes": _SYSTEM_RAM_TOTAL or None,
        "network_up_bytes_per_second": None,
        "network_down_bytes_per_second": None,
        "tokens_per_second_5s": _generation_tokens_per_second(),
    }
    if _HAS_PSUTIL and _psutil is not None:
        try:
            data["cpu_percent"] = float(_psutil.cpu_percent(interval=None))
            vm = _psutil.virtual_memory()
            used = _portable_ram_used_bytes()
            total = int(_SYSTEM_RAM_TOTAL or getattr(vm, "total", 0) or 0)
            data["ram_used_bytes"] = used
            data["ram_total_bytes"] = total or None
            if used is not None and total > 0:
                data["ram_percent"] = used / total * 100
            else:
                data["ram_percent"] = float(getattr(vm, "percent", 0.0))
        except Exception:
            pass

        try:
            nio = _psutil.net_io_counters()
            now = time.monotonic()
            with _system_net_lock:
                if _system_net_prev is not None:
                    previous_up, previous_down, previous_at = _system_net_prev
                    dt = now - previous_at
                    delta_up = int(nio.bytes_sent) - previous_up
                    delta_down = int(nio.bytes_recv) - previous_down
                    if dt > 0 and delta_up >= 0 and delta_down >= 0:
                        data["network_up_bytes_per_second"] = delta_up / dt
                        data["network_down_bytes_per_second"] = delta_down / dt
                _system_net_prev = (int(nio.bytes_sent), int(nio.bytes_recv), now)
        except Exception:
            pass

    data["gpu_percent"] = _portable_gpu_percent()
    return data


async def _system_telemetry_loop() -> None:
    """Publish desktop telemetry cheaply while keeping generation rate responsive."""
    cached: dict | None = None
    next_system_sample_at = 0.0
    loop = asyncio.get_running_loop()
    while True:
        try:
            await asyncio.sleep(_SYSTEM_TELEMETRY_PUBLISH_SECONDS)
            target = _active_ws
            if target is None or _ws_transport(target) != "desktop":
                continue

            now = loop.time()
            if cached is None or now >= next_system_sample_at:
                cached = await asyncio.to_thread(_collect_system_telemetry)
                next_system_sample_at = now + _SYSTEM_TELEMETRY_INTERVAL_SECONDS
            else:
                cached = {
                    **cached,
                    "tokens_per_second_5s": _generation_tokens_per_second(),
                }

            if _active_ws is not target:
                continue
            try:
                await target.send_json(cached)
            except Exception:
                # Connection cleanup remains owned by ws_endpoint's reader/finally.
                pass
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("System telemetry iteration failed")


# ── Agent Awake — host wiring ──────────────────────────────────────────────────
# This server owns ONE shared conversation thread (_state.thread_id) across
# every /chat and /ws caller. An awake-fired turn is deliberately run on that
# SAME thread via the SAME _run_agent_sync path post_chat/ws_endpoint use —
# NOT a separate "awake" thread — so it becomes an ordinary turn in the one
# live conversation. The macOS notification in _post_awake_notification is
# only a "something happened" nudge for when nobody is looking at the UI right
# now, not a transcript channel.
#
# Only the most-recently-connected WS is tracked (consistent with this file's
# single-session design elsewhere — no multi-connection fan-out).
_active_ws: WebSocket | None = None
_active_ws_loop: asyncio.AbstractEventLoop | None = None


def _ws_transport(websocket: WebSocket | None) -> str:
    if websocket is None:
        return ""
    try:
        return str(websocket.query_params.get("transport") or "").strip().lower()
    except Exception:
        return ""


# Ceiling for one fired turn's total wall time. Nobody is watching a fired
# turn — post_chat/ws_endpoint have no such timeout because a human notices a
# stuck request and can restart, but a wedged LLM server (Metal-crash/GPU-hang
# class scripts_max's monitor watches for) would otherwise hold _busy forever
# on this path, silently 503-ing every future /chat and WS query with nobody
# around to notice. Generous — real multi-tool research turns can run several
# minutes — but bounded.
_AWAKE_TURN_TIMEOUT_SEC = 600

# Captured once in the FastAPI startup hook so AwakeEngine's fire callback —
# invoked from the engine's OWN daemon thread (awake_engine.AwakeEngine._run)
# — has a live loop to schedule onto.
_awake_loop: asyncio.AbstractEventLoop | None = None
_awake_engine: AwakeEngine | None = None

# FIFO queue of pending awake fires (holds watch_ids only — the message for
# each lives in _awake_pending, see _enqueue_awake) + watch_id -> latest
# message for ids currently queued or running (dedup/coalescing guard) + the
# single consumer task draining the queue (see _awake_queue_worker). All
# three set together in _start_awake_engine, once a running loop exists.
_awake_queue: asyncio.Queue | None = None
_awake_pending: dict[str, str] = {}
_awake_worker_task: asyncio.Task | None = None


def _notification_snippet(text: str, limit: int = 80) -> str:
    """First line of `text`, minus a leading [AWAKE:id] tag (redundant — the
    caller already puts watch_id in the notification separately), sanitized
    for a literal AppleScript string and capped at `limit` chars."""
    first_line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    first_line = re.sub(r"^\[AWAKE:[^\]]+\]\s*", "", first_line)
    first_line = first_line.replace("\\", "").replace('"', "'")
    return first_line[:limit] + ("…" if len(first_line) > limit else "")


def _post_awake_notification(watch_id: str, snippet: str) -> None:
    """Best-effort macOS notification for when no WS client is connected to
    receive the fired turn's result — the user learns something happened even
    with the UI closed. Runs via subprocess (blocking) — callers MUST invoke
    this off the event loop (run_in_executor), never awaited directly.
    try/except-everything: a failed notification must never take down the
    awake turn or the server."""
    try:
        text = f"awake {watch_id}: {snippet}" if snippet else f"awake {watch_id} fired"
        script = f'display notification "{text}" with title "ENDEAVOR TH"'
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except Exception as e:
        log.warning("awake: notification failed for %s: %s", watch_id, e)


def _awake_fire(watch_id: str, message: str) -> None:
    """AwakeEngine fire callback — called SYNCHRONOUSLY from the engine's own
    daemon thread (awake_engine.py: tick() calls self._fire(wid, message)
    directly, not awaited, wrapped only in a broad try/except). Must therefore
    schedule-and-return immediately: running the turn inline here would stall
    every OTHER watcher's check for as long as this turn takes, on every tick.
    Schedules _enqueue_awake (dedup + FIFO put, see its docstring) onto the
    server's own asyncio loop (captured at startup) — the actual turn runs
    later, drained one at a time by _awake_queue_worker."""
    loop = _awake_loop
    if loop is None:
        log.warning("awake: fire(%s) dropped — server loop not captured yet (starting up?)", watch_id)
        return
    try:
        asyncio.run_coroutine_threadsafe(_enqueue_awake(watch_id, message), loop)
    except Exception as e:
        # Defensive belt-and-suspenders: the engine already try/excepts fire(),
        # but a scheduling failure here (e.g. loop closing during shutdown)
        # must never propagate back onto the engine thread.
        log.warning("awake: could not schedule fire(%s): %s", watch_id, e)


async def _enqueue_awake(watch_id: str, message: str) -> None:
    """Runs on the server's own event loop (scheduled via
    run_coroutine_threadsafe from the engine's daemon thread, see
    _awake_fire). Dedup/coalescing guard: if `watch_id` is already queued or
    running, don't enqueue a second copy — but DO overwrite its stored
    message with this newer one, so the eventual run uses whatever was most
    recent rather than silently discarding a distinct occurrence.

    Needed because "every" watchers re-detect due on every tick while queued
    behind a long turn — e.g. interval_minutes=5 stuck behind a 20-minute user
    turn would otherwise re-trigger at t=5,10,15 and pile up 4 near-identical
    copies of the same task by the time the queue finally drains. The worker
    reads _awake_pending[watch_id] at the LATEST possible moment (after
    acquiring _busy, right before running) so a coalesce that lands anywhere
    from "still queued" to "still waiting for the lock" is honored — only a
    coalesce that lands after the turn has already started running is too
    late to change anything."""
    if watch_id in _awake_pending:
        log.info("awake: fire(%s) coalesced — already queued/running, message updated to latest", watch_id)
        _awake_pending[watch_id] = message
        return
    _awake_pending[watch_id] = message
    await _awake_queue.put(watch_id)


async def _awake_queue_worker() -> None:
    """Single consumer draining _awake_queue in FIFO order, one turn at a
    time — this is what gives colliding awake fires in-order execution
    without a separate lock (a second/third queued fire just waits its turn
    behind this loop). Started once by _start_awake_engine, runs for the
    process lifetime, cancelled in _stop_awake_engine.

    Each dequeued item still competes for _busy against real user turns
    (post_chat/ws_endpoint) exactly as before — a queued fire waits as long as
    it takes for the current user turn to finish, it is never dropped for
    waiting too long (single-GPU rule — awake and user turns still never run
    concurrently)."""
    while True:
        watch_id = await _awake_queue.get()
        ran = False
        message = None
        try:
            if getattr(_state, "app", None) is None:
                log.warning("awake: fire(%s) dropped — _state.app not ready (server booting/offline)", watch_id)
                continue
            await _busy.acquire()
            try:
                message = _awake_pending.get(watch_id, "")
                await _run_awake_turn(watch_id, message)
                ran = True
            finally:
                _busy.release()
        except Exception:
            log.exception("awake: queue worker crashed handling fire(%s)", watch_id)
        finally:
            # A coalesce that lands WHILE _run_awake_turn above is already
            # executing updates _awake_pending with nothing left to consume it
            # — popping unconditionally here would silently discard that newer
            # fire (a times= watcher's second slot lost for the whole day,
            # since awake_engine already marked it fired at detection time).
            # Re-queue once instead: if the pending message no longer matches
            # what we just ran, a coalesce happened during the run. Gated on
            # `ran` — without it, the "dropped, _state.app not ready" continue
            # path leaves message=None while _awake_pending[watch_id] is a
            # real string, so None != string is always true and the watch_id
            # re-queues itself forever with no delay (tight spin).
            if ran and watch_id in _awake_pending and _awake_pending[watch_id] != message:
                log.info("awake: fire(%s) coalesced during its own run — re-queuing the newer message", watch_id)
                await _awake_queue.put(watch_id)
            else:
                _awake_pending.pop(watch_id, None)
            _awake_queue.task_done()


async def _deliver_awake_result(watch_id: str, parts: list[str], message: str) -> None:
    """Deliver a finished awake turn — forward finished response part(s) + a
    trailing 'done' to whatever WS is currently connected, reusing event types
    the renderer already handles. No parts (error/cancelled with nothing to
    show) or a dead WS both fall back to the notification: even with a live
    WS, a silent 'done' with zero text is not useful feedback on its own.

    Split out from _run_awake_turn so this branch is unit-testable with a
    fake WS object — it does not touch _busy/_run_agent_sync/_state at all."""
    delivered = False
    if _active_ws is not None and parts:
        try:
            for part in parts:
                await _active_ws.send_json({"type": "response", "content": part})
            await _active_ws.send_json({"type": "done"})
            delivered = True
        except Exception:
            delivered = False  # WS died between fire-start and now

    if not delivered:
        snippet = _notification_snippet(parts[0] if parts else message)
        await asyncio.get_running_loop().run_in_executor(None, _post_awake_notification, watch_id, snippet)


async def _run_awake_turn(watch_id: str, message: str) -> None:
    """Executes one awake-fired turn's body and delivers the result. `message`
    is the engine's own text verbatim ("[AWAKE:id] <task>" + any file/screen
    context hints) — passed straight through to _run_agent_sync as the query,
    same as a real user message.

    Caller (_awake_queue_worker) must already hold _busy and is responsible
    for releasing it — this function neither acquires nor releases it, so it
    stays a pure run-and-deliver unit, independently testable the same way
    _deliver_awake_result is. Bounded by _AWAKE_TURN_TIMEOUT_SEC — with a
    human on a real user turn, a stuck request is noticed and the server
    restarted; a fired turn has nobody watching, so an unattended hang here
    would hold _busy forever and silently 503 every future query."""
    parts: list[str] = []
    try:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        cancel_event = threading.Event()
        threading.Thread(
            target=_run_agent_sync, args=(message, q, loop, None, None, None, cancel_event),
            daemon=True,
        ).start()
        deadline = loop.time() + _AWAKE_TURN_TIMEOUT_SEC
        while True:
            timeout = max(0.01, deadline - loop.time())
            try:
                event = await asyncio.wait_for(q.get(), timeout=timeout)
            except asyncio.TimeoutError:
                log.error("awake: fire(%s) gave no done/error within %ss — abandoning wait",
                          watch_id, _AWAKE_TURN_TIMEOUT_SEC)
                cancel_event.set()
                break
            etype = event.get("type")
            if etype == "response":
                parts.append(event.get("content", ""))
            elif etype in ("done", "error", "cancelled"):
                if etype == "error":
                    log.warning("awake: fire(%s) turn error: %s", watch_id, event.get("msg"))
                break
            # progress/phase — discarded here; no live streaming feed for an
            # awake turn yet.
    except Exception as e:
        log.warning("awake: fire(%s) run failed: %s", watch_id, e)

    await _deliver_awake_result(watch_id, parts, message)


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
        self._generation_runs: set[str] = set()
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
        run_key = str(run_id or "default")
        self._generation_runs.add(run_key)
        _generation_run_start(run_key)
        self._timer.start()
        if self._timer.had_tool:
            self._put({"type": "synthesis_start"})

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._check_cancel()
        if _callback_has_generation_payload(token, kwargs):
            _mark_generation_token()
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
        _generation_run_end(run_key)
        self._generation_runs.discard(run_key)
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
        _generation_run_end(run_key)
        self._generation_runs.discard(run_key)
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
            reset_web_counter=_reset_web_counter,
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
        _generation_run_cleanup(cb._generation_runs)
        cb._generation_runs.clear()
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


@api.get("/status", dependencies=[Depends(_require_token)])
def get_status():
    return {
        "server_up": _mlx_up(),
        "online": _internet_up(retries=1),
        "skill": _state.skill,
        "skills": _list_skills(),
        "builtin_cmds": _load_builtin_cmds(),
        "model": get_model(),
        "model_server_mode": get_server_mode(),
        "model_server_port": get_server_port(),
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
    _sync_runtime_settings_from_owner_file_if_idle()
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


def _sanitize_upload_name(name: str) -> str:
    """basename only — no directory separators or traversal can survive this,
    so the caller never needs to think about a hostile filename escaping uploads/"""
    base = os.path.basename((name or "").strip().replace("\x00", ""))
    base = re.sub(r"[^\w.\-]", "_", base) or "upload"
    if base in (".", ".."):
        base = "upload"
    return base[:200]


_upload_lock = asyncio.Lock()  # serializes _unique_upload_path()'s check against the write
                                # that claims it — both run under this lock in post_upload,
                                # so two concurrent uploads of the same filename can never
                                # race into the same path between the check and the write.


def _unique_upload_path(base: str) -> str:
    """avoid clobbering an existing upload or a file the agent already produced.
    Caller MUST hold _upload_lock across this call and the write that follows it —
    on its own this is a check-then-act race."""
    stem, ext = os.path.splitext(base)
    candidate = os.path.join("uploads", base)
    n = 1
    while os.path.exists(_resolve_write_path(candidate)):
        candidate = os.path.join("uploads", f"{stem}_{n}{ext}")
        n += 1
    return candidate


def _write_upload_file(abs_path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "wb") as f:
        f.write(data)


def _attach_hint(rel_path: str) -> str:
    """Text hint injected into the query so the model knows an attached file exists
    and which tool reads it — mirrors the Electron upload hint, with
    direct-vision read_image and optional OCR/detail sensors."""
    ext = os.path.splitext(rel_path)[1].lower()
    if ext in _KNOWN_IMG_EXTS:
        return (f"[ไฟล์แนบ (รูปภาพ): {rel_path}]\n"
                f"ใช้ tool: read_image (ส่งภาพต้นฉบับให้ VLM โดยตรง; ถ้าต้องการอ่านข้อความ/ตาราง/QR ค่อยใช้ detail=text)")
    if ext in _AUDIO_EXT:
        return f"[ไฟล์แนบ (เสียง): {rel_path}]\nใช้ tool: read_file (จะถอดเสียงเป็นข้อความอัตโนมัติ)"
    if ext in _VIDEO_EXT:
        return f"[ไฟล์แนบ (วิดีโอ): {rel_path}]\nใช้ tool: read_file (จะถอดเสียงจากวิดีโอเป็นข้อความอัตโนมัติ)"
    return f"[ไฟล์แนบ: {rel_path}]\nใช้ tool: read_file"


@api.post("/upload", dependencies=[Depends(_require_token)])
async def post_upload(request: Request):
    """Save a client-uploaded file into workspace/uploads/ and return a text hint.

    The Electron renderer and any explicit custom client use this backend endpoint
    as the attach mechanism: bytes cross the local loopback transport once, land in
    workspace, then read_file/read_image (already sandboxed to that boundary) do
    the actual reading. There is no bundled browser HTML UI.

    Takes raw Request rather than `file: UploadFile = File(...)` on purpose: FastAPI
    resolves File() params (i.e. parses the full multipart body into a spooled temp
    file) BEFORE the endpoint body runs, and Starlette's parser has no size limit on
    individual file parts — so a size check placed inside the function body, no matter
    where, is always too late to stop the full upload from being received and spooled
    to disk first. Reading Content-Length ourselves before touching the body is the
    only way to reject an oversized upload before paying for it."""
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > READ_FILE_AUDIO_VIDEO_MAX_BYTES:
                return JSONResponse(
                    {"error": f"request too large (max {READ_FILE_AUDIO_VIDEO_MAX_BYTES // (1024 * 1024)}MB)"},
                    status_code=413,
                )
        except ValueError:
            pass  # malformed header — let the body-size check below catch it instead

    form = await request.form()
    file = form.get("file")
    if file is None or not hasattr(file, "filename"):
        return JSONResponse({"error": "file field required"}, status_code=400)

    safe_name = _sanitize_upload_name(file.filename or "upload")
    ext = os.path.splitext(safe_name)[1].lower()
    cap = READ_FILE_AUDIO_VIDEO_MAX_BYTES if ext in _AUDIO_EXT or ext in _VIDEO_EXT else READ_FILE_MAX_BYTES
    data = await file.read(cap + 1)
    await file.close()
    if len(data) > cap:
        return JSONResponse({"error": f"file too large (max {cap // (1024 * 1024)}MB)"}, status_code=413)
    if not data:
        return JSONResponse({"error": "empty file"}, status_code=400)

    async with _upload_lock:
        rel_path = _unique_upload_path(safe_name)
        err = _check_write_path(rel_path)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        abs_path = _resolve_write_path(rel_path)
        await run_in_threadpool(_write_upload_file, abs_path, data)
    return {"path": rel_path, "hint": _attach_hint(rel_path)}


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
    global _active_ws, _active_ws_loop
    _active_ws = websocket
    _active_ws_loop = asyncio.get_running_loop()
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
                _sync_runtime_settings_from_owner_file_if_idle()
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
                elif cmd == "/build_kb":
                    if await _ws_busy_guard(websocket): continue
                    def _do_build_kb():
                        from tools.rag_tool import _rag_engine_available, _missing_engine_message, _ensure_rag_path
                        if not _rag_engine_available():
                            return None, _missing_engine_message()
                        _ensure_rag_path()
                        import ingestor
                        import llm_client
                        llm_client.ensure_mlx_server()
                        result = ingestor.sync_knowledge_base()
                        return result, None
                    async with _busy:
                        try:
                            result, engine_error = await asyncio.get_running_loop().run_in_executor(None, _do_build_kb)
                        except Exception as e:
                            await websocket.send_json({"type": "error", "msg": f"build_kb ล้มเหลว: {e}"})
                            continue
                        if engine_error:
                            await websocket.send_json({"type": "error", "msg": engine_error})
                            continue
                        counts: dict[str, int] = {}
                        for row in result["rows"]:
                            counts[row["status"]] = counts.get(row["status"], 0) + 1
                        await websocket.send_json({
                            "type": "build_kb_ok",
                            "data_dir": result["data_dir"],
                            "total": result["total_found"],
                            "counts": counts,
                            "elapsed": result["elapsed"],
                            "health_issues": result["health_issues"],
                            "ghost_count": result["ghost_count"],
                        })
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

            elif msg_type == "get_runtime_settings":
                _sync_runtime_settings_from_owner_file_if_idle()
                await websocket.send_json(_runtime_settings_payload())

            elif msg_type == "set_runtime_settings":
                _sync_runtime_settings_from_owner_file_if_idle()
                result = await _apply_runtime_settings(
                    str(data.get("model", "")),
                    data.get("thinking_budget", get_thinking_budget()),
                    str(data.get("server_mode", get_server_mode())),
                    data.get("server_port", get_server_port()),
                    confirmed_low_ram=bool(data.get("confirmed_low_ram", False)),
                )
                await websocket.send_json(result)
                await websocket.send_json({"type": "status", **get_status()})
                await websocket.send_json(await _model_server_status_payload())

            elif msg_type == "get_model_server_status":
                await websocket.send_json(await _model_server_status_payload())

            elif msg_type == "set_model_server_watchdog":
                await websocket.send_json(
                    await _apply_model_server_watchdog(bool(data.get("enabled", False)))
                )

            elif msg_type == "model_server_action":
                action = str(data.get("action", "")).strip().lower()
                if action in {"start", "stop", "reset"}:
                    await websocket.send_json(_model_server_progress_payload({
                        "start": "starting", "stop": "stopping", "reset": "resetting",
                    }[action]))
                await websocket.send_json(await _apply_model_server_action(action))

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
        if _active_ws is websocket:
            _active_ws = None
            _active_ws_loop = None
        ws_reader_task.cancel()
        try:
            await ws_reader_task
        except (asyncio.CancelledError, Exception):
            pass


# ── Agent Awake — engine lifecycle ─────────────────────────────────────────────

@api.on_event("startup")
async def _start_awake_engine() -> None:
    """Start standalone model ownership/watchdog, telemetry, then Awake."""
    global _awake_loop, _awake_engine, _awake_queue, _awake_worker_task, _model_watchdog_task, _system_telemetry_task
    _awake_loop = asyncio.get_running_loop()

    runtime = get_runtime_settings()
    if not runtime.get("runtime_locked"):
        try:
            shared = await asyncio.to_thread(_auto_attach_max_test_server_if_present)
            if shared is not None:
                log.info(
                    "Detected Agent MAX VLM test server on :%s; using read-only shared mode",
                    get_server_port(),
                )
            startup_model = await asyncio.to_thread(_reconcile_runtime_server)
            log.info(
                "Model server startup mode=%s state=%s port=%s",
                get_server_mode(), startup_model.get("state"), get_server_port(),
            )
        except Exception as exc:
            # Keep the control plane/UI available even if the model is missing,
            # occupied, loading, or the optional MAX test server is offline.
            log.error("Model server startup reconcile failed: %s", exc)
        _model_watchdog_task = asyncio.create_task(_model_server_watchdog_loop())

    _system_telemetry_task = asyncio.create_task(_system_telemetry_loop())
    _awake_queue = asyncio.Queue()
    _awake_worker_task = asyncio.create_task(_awake_queue_worker())
    _awake_engine = AwakeEngine(fire=_awake_fire, registry=Registry(), owner="agent_server")
    _awake_engine.start()
    restore_notices(drain_notices())


@api.on_event("shutdown")
async def _stop_awake_engine() -> None:
    global _awake_worker_task, _model_watchdog_task, _system_telemetry_task
    if _awake_engine is not None:
        _awake_engine.stop()
    if _awake_worker_task is not None:
        _awake_worker_task.cancel()
        _awake_worker_task = None
    if _model_watchdog_task is not None:
        _model_watchdog_task.cancel()
        _model_watchdog_task = None
    if _system_telemetry_task is not None:
        _system_telemetry_task.cancel()
        _system_telemetry_task = None
    # A pending coalesced fire left in _awake_pending would coalesce-skip
    # every future fire for that watch_id forever (the guard checks
    # `if watch_id in _awake_pending`) — clear it so a restart starts clean.
    _awake_pending.clear()


if __name__ == "__main__":
    log.info(f"Starting ENDEAVOR agent server on port {PORT}")
    if _AUTH_DISABLED:
        log.warning("AUTH DISABLED (AGENT_AUTH_DISABLED=1) — dev only, do not run alongside a browser")
    elif not os.getenv("AGENT_SERVER_TOKEN"):
        log.info(f"Auth token persisted at {_TOKEN_FILE} (0600) — clients must send X-Auth-Token / ?token=")
    uvicorn.run(api, host="127.0.0.1", port=PORT, log_level="warning")
