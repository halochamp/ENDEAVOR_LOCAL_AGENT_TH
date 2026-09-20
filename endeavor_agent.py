# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""endeavor_agent.py — ENDEAVOR_AGENT_V2 interactive CLI

รัน:  conda activate mlx && python endeavor_agent.py
Model + Think Budget + Server Mode + Model Server Port ใช้ config กลางเดียวกับ Electron UI (ดู config.py)
"""
from __future__ import annotations
import sys
import json as _json
import os
import uuid

if sys.version_info[:2] != (3, 11):
    print(f"\n❌ Python {sys.version_info.major}.{sys.version_info.minor} ไม่รองรับ — ต้องใช้ Python 3.11 (mlx env)")
    print("   รัน:  conda activate mlx && python endeavor_agent.py\n")
    sys.exit(1)

from langchain_core.callbacks import BaseCallbackHandler

import config
import graph as _graph
import planner as _planner
from graph import build_graph, force_compact, rewarm_after_compact, summarize_history
from react import get_system_prompt, ctx_stats as _ctx_stats
from config import RECURSION_LIMIT, CONTEXT_MAX_CHARS
from model_runtime import (
    reconcile_runtime_server,
    restart_owner_model_server,
    start_owner_model_server,
    stop_model_server as _stop_owned_model_server,
    model_server_status,
    shared_max_server_status,
    auto_attach_max_test_server_if_present,
    get_model_server_control_settings,
)
from runtime_common import (
    mlx_up as _server_up, internet_up as _internet_up,
    parse_plan_steps as _parse_plan_steps,
    extract_tool_content as _extract_tool_content,
    parse_tool_args as _parse_tool_args,
    guard_db_schema as _guard_db_schema, open_memory_store as _open_memory_store,
    load_memory_md as _load_memory_md, purge_thread as _purge_thread,
    scan_skill_roles as _scan_skill_roles, first_role_line as _first_role_line,
    ThinkingTimer, run_turn_core, vacuum_db,
    _MEMORY_DB, _MEMORY_MD, _MEMORY_THREAD, _DB_SCHEMA_VERSION,
    _MAX_DB_ROWS, load_history_pairs as _load_history_pairs,
)
import atexit
from tools import ALL_TOOLS, SKILL_TOOLS
from tools import _summarize as _summarize_tool
from tools._progress import set_callback as set_progress_callback, set_phase_callback, set_plan_callback
from tools.web_cache import web_count_reset as _reset_web_counter
from tools._safety import validate_approved_edit_folder as _validate_approved_edit_folder
from tools.edit_access import (
    add_approved_edit_folders as _add_approved_edit_folders,
    load_edit_access_state as _load_edit_access_state,
    remove_approved_edit_folder as _remove_approved_edit_folder,
    save_focus_folder as _save_focus_folder,
)
from agent_log import AgentLogger
from ui_cli import (
    Spinner, print_header, print_divider, print_user_prompt,
    print_tool_step, print_plan, print_synthesizing,
    print_agent_response, print_web_refs,
    extract_refs_from_search_result, prompt_user,
    print_mode_menu, print_special_commands, print_skill_help,
    print_runtime_settings_menu, print_runtime_choices,
    print_edit_access_menu,
    print_startup_hint,
    setup_skill_completer, update_ctx_info, print_compact_notice,
    _SPINNER_LABELS,
    PHASE_THINKING, PHASE_EXECUTING, PHASE_SYNTH,
    C_ARROW, C_GREEN, C_WARN, C_META, R,
)


_WEB_TOOLS = {"web_search", "browse_url", "browser_use", "recall_web",
              "fetch_sitemap", "batch_browse", "scrape_table",
              "research_orchestrator"}


def _get_skill_tools(skill: str, online: bool) -> list:
    tools = SKILL_TOOLS.get(skill, [])
    if not online:
        return [t for t in tools if t.name not in _WEB_TOOLS]
    return tools


def _ensure_server_alive(timeout: int = 240, interval: int = 3) -> bool:
    """Reconcile Agent TH's standalone owner or verify read-only MAX sharing."""
    del interval  # compatibility with older callers/tests
    try:
        status = reconcile_runtime_server(timeout=float(timeout))
    except Exception as exc:
        mode = config.get_server_mode()
        if mode == "shared_max":
            print(f"\n❌ External model server ใช้งานไม่ได้: {exc}\n")
        else:
            print(f"\n❌ เตรียม model server ของ Agent TH ไม่สำเร็จ: {exc}\n")
        return False
    if status.get("healthy"):
        return True
    print(f"\n❌ Model server ยังไม่พร้อม: {status.get('error') or status.get('state')}\n")
    return False


def _get_active_tools(online: bool) -> list:
    if online:
        return ALL_TOOLS
    return [t for t in ALL_TOOLS if t.name not in _WEB_TOOLS]


def _get_tool_detail(name: str, args: dict) -> str:
    if name == "web_search":
        return f'"{args.get("query", "")[:55]}"'
    if name == "create_plan":
        q = args.get("query", "")
        return q[:55] if q else ""
    if name in ("read_file", "write_file"):
        path = args.get("file_path", "")
        return path.split("/")[-1] if "/" in path else path
    if name == "bash":
        return args.get("command", "")[:55]
    if name == "grep":
        return f'"{args.get("pattern", "")[:40]}"'
    if name == "python_exec":
        code = args.get("code", "")
        for line in code.splitlines():
            if line.strip():
                return line.strip()[:55]
    if name in ("browse_url", "recall_web"):
        url = args.get("url", "")
        # แสดง domain + path สั้นๆ
        return url.replace("https://", "").replace("http://", "")[:55] if url else ""
    if name == "remember":
        return str(args.get("fact", ""))[:55]
    if name == "tool_loop":
        action = args.get("action", "")
        n = len(args.get("items", []))
        ctx = args.get("context", "")
        return f"{action}  {n} items" + (f"  [{ctx[:40]}]" if ctx else "")
    return ""


def _make_label(name: str, args: dict) -> str:
    """Spinner label สำหรับ tool — base label + detail (เช่น query/URL)."""
    base = _SPINNER_LABELS.get(name, f"⚙ {name}")
    detail = _get_tool_detail(name, args)
    if detail:
        # ตัด quotes ของ web_search/grep ที่ใส่มา
        clean = detail.strip('"').strip("'")
        return f"{base}: {clean[:50]}"
    return base


_PLAN_SKIP = {"create_plan"}


class _Turn:
    """Mutable UI state shared between _run_turn and _UICallback."""

    def __init__(self) -> None:
        self.spinner: "Spinner | None" = None
        self.active = False
        self.web_refs: list = []
        self.final = ""
        self.plan_steps: list[str] = []   # from create_plan output
        self.plan_step_idx: int = 0       # tool calls since create_plan

    def _bind(self) -> None:
        s = self.spinner
        if s:
            set_progress_callback(lambda msg: s.update_sub(msg))
            set_phase_callback(lambda label: s.update(label))

    def stop(self) -> None:
        if self.active and self.spinner:
            self.spinner.__exit__(None, None, None)
            self.active = False

    def start(self, label: str) -> None:
        self.spinner = Spinner(label)
        self.spinner.__enter__()
        self.active = True
        self._bind()

    def set_label(self, label: str) -> None:
        """Change the phase label of the running spinner (no thread churn)."""
        if self.active and self.spinner:
            self.spinner.update(label)

    def live_print(self, fn) -> None:
        """Print permanent output without racing the running spinner."""
        if self.active and self.spinner:
            self.spinner.live_print(fn)
        else:
            fn()


class _UICallback(BaseCallbackHandler):
    """Real-time tool UI events fired from inside _REACT.invoke() via RunnableConfig propagation."""
    raise_error = False

    def __init__(self, turn: _Turn, logger, turn_id) -> None:
        super().__init__()
        self._t = turn
        self._log = logger
        self._tid = turn_id
        self._run_to_name: dict = {}
        self._timer = ThinkingTimer(emit=self._t.set_label)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._timer.start()

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        self._timer.cancel()
        label = PHASE_EXECUTING if self._timer.had_tool else PHASE_THINKING
        self._t.set_label(label)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._timer.cancel()

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs) -> None:
        name = serialized.get("name", "")
        args = _parse_tool_args(input_str, kwargs)
        self._run_to_name[str(run_id)] = name
        if self._log:
            self._log.tool_call(name, args, self._tid)
        # Track plan step progress
        if self._t.plan_steps and name not in _PLAN_SKIP:
            self._t.plan_step_idx += 1
            idx = self._t.plan_step_idx
            total = len(self._t.plan_steps)
            step_desc = self._t.plan_steps[idx - 1] if idx <= total else ""
            step_tag = f" [{idx}/{total}]"
        else:
            step_tag = ""
        # Single persistent spinner: print the step line through the lock, then
        # just relabel — never start/stop (concurrent tool dispatch would orphan).
        self._t.live_print(lambda: print_tool_step(name, _get_tool_detail(name, args)))
        base = _make_label(name, args)
        self._t.set_label(f"{base}{step_tag}")
        self._timer.mark_tool()

    def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        name = self._run_to_name.get(str(run_id), "")
        content = _extract_tool_content(output)
        if self._log:
            self._log.tool_result(name, content, self._tid)
        if name == "create_plan":
            steps = _parse_plan_steps(content)
            if steps:
                self._t.plan_steps = steps
                self._t.plan_step_idx = 0
                self._t.live_print(lambda: print_plan(steps))
        elif name == "web_search":
            refs = extract_refs_from_search_result(content)
            self._t.web_refs.extend(refs)
        label = PHASE_EXECUTING if self._timer.had_tool else PHASE_THINKING
        self._t.set_label(label)

    def on_tool_error(self, error, *, run_id, **kwargs) -> None:
        label = PHASE_EXECUTING if self._timer.had_tool else PHASE_THINKING
        self._t.set_label(label)



def _load_skill_registry() -> dict:
    """โหลด skills/skill.json (usage + hint) แล้ว auto-detect skills จาก *.md"""
    skills_dir = os.path.join(os.path.dirname(__file__), "skills")
    try:
        with open(os.path.join(skills_dir, "skill.json"), encoding="utf-8") as f:
            data = _json.load(f)
    except Exception:
        data = {}

    data["skills"] = [{"name": n, "description": role or "—"} for n, role in _scan_skill_roles(skills_dir)]
    return data


def _activate_skill(sname: str) -> tuple[str, str]:
    """โหลด skill และแสดง role hint. คืน (active_skill, content) หรือ ("","") ถ้าไม่พบ."""
    if not sname or any(c in sname for c in ("/", "\\")) or ".." in sname:
        print(f" ไม่พบ skills/{sname}.md\n")
        return "", ""
    spath = os.path.join(os.path.dirname(__file__), "skills", f"{sname}.md")
    if not os.path.exists(spath):
        print(f" ไม่พบ skills/{sname}.md\n")
        return "", ""
    with open(spath, encoding="utf-8") as sf:
        content = sf.read().strip()
    role_hint = _first_role_line(content)
    print(f" เปิด {sname} mode — {role_hint}" if role_hint else f" เปิด {sname} mode")
    print(f" พิมพ์ /{sname} อีกครั้งหรือ /exit เพื่อออก\n")
    return sname, content


def _run_build_kb() -> None:
    """/build_kb — real chunk+embed+BM25 ingestion of the sibling ENDEAVOR_RAG_LITE
    engine's knowledge base, driven straight from this CLI (no need to open
    ENDEAVOR_RAG_LITE's own main.py separately). Reuses ingestor.sync_knowledge_base()
    — the same pipeline ENDEAVOR_RAG_LITE's own main.py runs.

    Unlike agent_max_vlm's /build_kb, there is no topics/tags rebuild step
    here — this fork has no rag_index_builder.py / rag_rebuild_index tool, so
    /build_kb only ingests and reports a health check."""
    from tools.rag_tool import _RAG_DIR, _ensure_rag_path, _rag_engine_available, _missing_engine_message
    if not _rag_engine_available():
        print(f"\n   {C_WARN}{_missing_engine_message()}{R}\n")
        return
    _ensure_rag_path()
    import ingestor
    import llm_client

    total_preview = sum(1 for f in ingestor.DATA_DIR.rglob("*") if ingestor._is_indexable_file(f))
    print()
    print(f"   {C_META}folder: {C_ARROW}{ingestor.DATA_DIR}{R}")
    print(f"   {C_META}found:  {total_preview} file(s){R}")
    print()

    if total_preview == 0:
        print(f"   {C_WARN}No supported files found in knowledge/{R}\n")
        return

    with Spinner("⚙️  กำลังเปิด RAG's LLM (novelty check)…") as sp:
        llm_client.ensure_mlx_server(status_cb=sp.update_sub)

    def _on_start(idx, total, name):
        print(f"   {C_META}[{idx}/{total}]{R} {name}…")

    def _on_file(idx, total, name, status, n_chunks, err):
        if err:
            print(f"   {C_ARROW}{name.ljust(32)}{R}  {C_WARN}✗ error{R}  {C_META}{err[:60]}{R}")
        else:
            tag = f"{C_GREEN}✓ {status}{R}"
            print(f"   {C_ARROW}{name.ljust(32)}{R}  {tag}  {C_META}{n_chunks} chunks{R}")

    result = ingestor.sync_knowledge_base(on_file=_on_file, on_file_start=_on_start)
    print()
    issues, ghost_count = result["health_issues"], result["ghost_count"]
    if not issues and not ghost_count:
        print(f"   {C_GREEN}✓ self-check: no anomalies{R}\n")
    else:
        print(f"   {C_WARN}⚠ self-check found anomalies:{R}")
        for issue in issues:
            print(f"     {C_WARN}- {issue}{R}")
        if ghost_count:
            print(f"     {C_META}- {ghost_count} registered file(s) have zero chunks (dedup ghosts, informational){R}")
        print()


def _run_turn(app, q: str, cfg: dict, *, thread_id: str, saver, db_conn,
              logger: "AgentLogger | None" = None, system_prompt: str = "") -> None:
    """รัน 1 turn พร้อม UI: spinner, tool steps, plan, web refs, final"""
    turn_id = logger.new_turn_id() if logger else None
    if logger:
        logger.turn_start(q, turn_id)

    t = _Turn()
    t.start(PHASE_THINKING)

    ui_cb = _UICallback(t, logger, turn_id)
    stream_cfg = {**cfg, "callbacks": [ui_cb]}

    def _on_plan(plan_text: str) -> None:
        """Forced create_plan is pre-seeded before _REACT.invoke(), so on_tool_start
        never fires for it — emit_plan() (called from graph.py::_force_plan_or_directive
        BEFORE the react node runs) is the only way to show the plan ahead of the
        web-tool prints it precedes. Mirrors agent_server.py::_on_plan."""
        t.live_print(lambda: print_tool_step("create_plan", ""))
        ui_cb._timer.mark_tool()
        steps = _parse_plan_steps(plan_text)
        if steps:
            t.plan_steps = steps
            t.plan_step_idx = 0
            t.live_print(lambda s=steps: print_plan(s))

    set_plan_callback(_on_plan)

    def _on_final(content: str) -> None:
        t.set_label(PHASE_SYNTH)
        t.live_print(print_synthesizing)
        t.final = content

    try:
        run_turn_core(
            app, q, stream_cfg,
            thread_id=thread_id, saver=saver, db_conn=db_conn,
            reset_web_counter=_reset_web_counter,
            on_final=_on_final,
        )

    except Exception as e:
        t.stop()
        set_progress_callback(None)
        set_phase_callback(None)
        set_plan_callback(None)
        if logger:
            logger.error(e, turn_id)
            logger.flush()
        print(f"\n[error] {e}\n")
        return

    t.stop()
    set_progress_callback(None)
    set_phase_callback(None)
    set_plan_callback(None)

    print_agent_response(t.final if t.final else "_(ไม่มีคำตอบ)_")

    if logger:
        if t.final:
            logger.final_response(t.final, turn_id)
        if not ui_cb._timer.had_tool and t.final:
            logger.log("direct_answer", {}, turn_id=turn_id)
        logger.flush()

    if t.web_refs:
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for title, url in t.web_refs:
            if url not in seen:
                seen.add(url)
                unique.append((title, url))
        print_web_refs(unique)


_LOAD_CMDS = {"โหลดความจำ", "load history", "/history", "จำเก่า", "โหลด history"}


def _is_load_cmd(q: str) -> bool:
    return q.strip().lower() in _LOAD_CMDS


def _invalidate_runtime_llms() -> None:
    _graph.invalidate_llm_cache()
    _planner.invalidate_llm_cache()
    _summarize_tool.invalidate_llm_cache()


def _runtime_model_matches_active_server() -> tuple[bool, str]:
    """Return whether the current standalone/shared server matches TH runtime state."""
    runtime = config.get_runtime_settings()
    if runtime.get("runtime_locked"):
        return True, config.get_model()
    status = model_server_status()
    active = str(status.get("loaded_model") or "")
    return bool(status.get("healthy")), active


def _confirm_high_quality_model_on_low_ram(model: str) -> bool:
    """Ask before selecting 35B on <24 GB RAM; warning is never a hard block."""
    if not config.high_quality_model_warning_required(model):
        return True
    ram_bytes = config.physical_memory_bytes()
    ram_gb = max(1, round(ram_bytes / (1024 ** 3))) if ram_bytes else 0
    shown = f"ประมาณ {ram_gb}GB" if ram_gb else "ต่ำกว่า 24GB"
    print(f" {C_WARN}⚠ เครื่องนี้มี RAM {shown}; Qwen3.6 35B · VLM เป็นโมเดลขนาดใหญ่{R}")
    print(f" {C_WARN}  การดาวน์โหลด/โหลดอาจใช้พื้นที่และ swap สูง แต่ยังเลือกใช้ได้{R}")
    print(f" {C_META}  ต้องการดาวน์โหลด/ใช้ Qwen3.6 35B · VLM ต่อหรือไม่? [y/N]{R}")
    answer = (prompt_user() or "").strip().lower()
    return answer in {"y", "yes", "ใช่", "ตกลง"}


def _apply_cli_runtime_settings(
    model: str,
    thinking_budget: int,
    server_port: int | None = None,
) -> dict:
    """Apply CLI state through the same owner config used by Electron Settings."""
    runtime = config.get_runtime_settings()
    requested = str(model or "").strip()
    old_model = config.get_model()
    old_budget = config.get_thinking_budget()
    old_mode = config.get_server_mode()
    old_port = config.get_server_port()
    requested_port = old_port if server_port is None else int(server_port)

    if runtime.get("shared_server"):
        if requested != old_model:
            raise ValueError("external server is read-only; Model is controlled by the connected server")
        if requested_port != old_port:
            shared = shared_max_server_status(port=requested_port)
            if not shared.get("healthy"):
                raise RuntimeError(shared.get("error") or "MAX test server is unavailable")
            requested = str(shared.get("loaded_model") or "")
        changed = config.set_runtime_settings(
            model=requested,
            thinking_budget=int(thinking_budget),
            server_mode="shared_max",
            server_port=requested_port,
        )
        config.adopt_shared_model(requested)
        return changed

    changed = config.set_runtime_settings(
        model=requested,
        thinking_budget=int(thinking_budget),
        server_mode="standalone",
        server_port=requested_port,
    )
    server_changed = bool(changed.get("model_changed") or changed.get("server_port_changed"))
    if not server_changed:
        return changed
    desired_running = get_model_server_control_settings().get("desired_state") == "running"
    if not desired_running:
        return changed

    try:
        restart_owner_model_server(timeout=240.0, previous_port=old_port)
    except Exception as exc:
        rollback_error = ""
        try:
            failed_port = config.get_server_port()
            config.set_runtime_settings(
                model=old_model,
                thinking_budget=old_budget,
                server_mode=old_mode,
                server_port=old_port,
            )
            if failed_port != old_port:
                try:
                    _stop_owned_model_server(port=failed_port)
                except Exception:
                    pass
            start_owner_model_server(timeout=240.0)
        except Exception as rollback_exc:
            rollback_error = f"; rollback failed: {rollback_exc}"
        raise RuntimeError(f"model/port switch failed: {exc}{rollback_error}") from exc
    return changed


def _apply_cli_edit_access_action(action: str, value: str = "") -> dict:
    """Apply one CLI edit-access action through the shared backend state."""
    action = str(action or "").strip().lower()
    if action == "add":
        folder = _validate_approved_edit_folder(value)
        return _add_approved_edit_folders([folder])
    if action == "remove":
        return _remove_approved_edit_folder(value)
    if action == "focus":
        folder = _validate_approved_edit_folder(value) if str(value or "").strip() else ""
        return _save_focus_folder(folder)
    if action == "clear_focus":
        return _save_focus_folder("")
    raise ValueError(f"unsupported edit-access action: {action}")


def main() -> None:
    runtime = config.get_runtime_settings()
    if not runtime.get("runtime_locked"):
        shared = auto_attach_max_test_server_if_present()
        if shared is not None:
            print(
                f"[external] ใช้ model server ภายนอกแบบ read-only "
                f":{config.get_server_port()} · {config.get_model_label()}"
            )
        if not _ensure_server_alive():
            return
    elif not _server_up():
        print(f"[!] เชื่อม custom MLX backend ไม่ได้ที่ {config.get_mlx_base_url()}")
        return

    matches, active_model = _runtime_model_matches_active_server()
    if not matches:
        shown = active_model or "unknown"
        print("[!] Runtime config กับ MLX server ไม่ตรงกัน")
        print(f"    config: {config.get_model()}")
        print(f"    active: {shown}")
        return

    online = _internet_up()
    active_tools = _get_active_tools(online)
    if not online:
        print(f" ⚠️  ไม่มีอินเทอร์เน็ต — ปิด web tools ({len(ALL_TOOLS) - len(active_tools)} tools)\n")

    _db_conn, saver = _open_memory_store(_MEMORY_DB)
    atexit.register(lambda: vacuum_db(_db_conn))
    app = build_graph(checkpointer=saver, memory=_load_memory_md(), tools=active_tools)
    logger = AgentLogger()

    # default: fresh session — auto-saves to _MEMORY_THREAD per turn via append_pair_to_history
    thread_id = str(uuid.uuid4())
    cfg = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": thread_id}}
    _active_skill = ""
    _active_skill_content = ""

    def _set_skill(sname: str) -> tuple[str, str]:
        """เปิด/ปิด skill mode + rebuild app เมื่อ skill-only tools เปลี่ยนสถานะ
        (SKILL_TOOLS ไม่อยู่ใน ALL_TOOLS — bind เฉพาะตอน skill mode ที่ตรงกัน active)"""
        nonlocal app
        if sname:
            new_skill, new_content = _activate_skill(sname)
        else:
            new_skill, new_content = "", ""
        old_extra = _get_skill_tools(_active_skill, online)
        new_extra = _get_skill_tools(new_skill, online)
        if old_extra != new_extra:
            app = build_graph(checkpointer=saver, memory=_load_memory_md(), tools=active_tools + new_extra)
        return new_skill, new_content

    def _runtime_tools() -> list:
        return active_tools + _get_skill_tools(_active_skill, online)

    def _rebuild_for_runtime_change() -> None:
        nonlocal app
        _invalidate_runtime_llms()
        app = build_graph(
            checkpointer=saver,
            memory=_load_memory_md(),
            tools=_runtime_tools(),
        )

    import threading as _threading
    # Pre-warm mlx_vlm.server prompt cache with [system_prompt] — runs in background so the
    # user's real first turn hits a cached system segment instead of a cold ~16k prefill.
    _WARM_THREAD_ID = "__cache_warm__"
    _warm_llm_ready = _threading.Event()

    def _warm_llm_cache():
        try:
            _purge_thread(_db_conn, _WARM_THREAD_ID)
            _wcfg = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": _WARM_THREAD_ID}}
            for _ in app.stream({"messages": [{"role": "user", "content": "hi"}]},
                                config=_wcfg, stream_mode="updates"):
                pass
        except Exception:
            pass
        finally:
            try:
                _purge_thread(_db_conn, _WARM_THREAD_ID)
            finally:
                _warm_llm_ready.set()

    _threading.Thread(target=_warm_llm_cache, daemon=True).start()

    def _sync_runtime_config_from_disk() -> bool:
        """Pick up Electron-written config before the next CLI action/turn."""
        persisted = config.get_persisted_runtime_settings()
        if persisted is None:
            return False
        if not config.refresh_runtime_settings_from_file():
            return False
        status = model_server_status()
        if not status.get("healthy"):
            return False
        if config.get_server_mode() == "shared_max":
            config.adopt_shared_model(str(status.get("loaded_model") or ""))
        if not _warm_llm_ready.is_set():
            _warm_llm_ready.wait()
        _rebuild_for_runtime_change()
        return True

    def _print_runtime_header() -> None:
        print_header(
            config.get_model(),
            len(active_tools),
            online=online,
            thinking_budget=config.get_thinking_budget(),
            thinking_label=config.get_thinking_budget_label(),
        )

    def _apply_runtime_selection(
        model: str,
        thinking_budget: int,
        server_port: int | None = None,
    ) -> str:
        requested_model = str(model or "").strip()
        old_port = config.get_server_port()
        requested_port = old_port if server_port is None else int(server_port)
        model_change_requested = requested_model != config.get_model()
        port_change_requested = requested_port != old_port
        if not _warm_llm_ready.is_set():
            print(" กำลังรอ model cache ก่อนเปลี่ยน runtime settings…")
            _warm_llm_ready.wait()
        try:
            spinner_text = (
                f"⏳  กำลังเปลี่ยน Model Server Port → :{requested_port}…"
                if port_change_requested
                else "⏳  กำลังเตรียมเปลี่ยน Model…"
                if model_change_requested
                else "⚙️  กำลังปรับ Think Budget…"
            )
            with Spinner(spinner_text):
                changed = _apply_cli_runtime_settings(model, thinking_budget, requested_port)
        except Exception as exc:
            print(f" {C_WARN}⚠ เปลี่ยน runtime settings ไม่สำเร็จ: {exc}{R}\n")
            return "error"
        if any(changed.get(key) for key in (
            "model_changed", "thinking_budget_changed", "server_port_changed",
        )):
            _rebuild_for_runtime_change()
        print(
            f" {C_GREEN}✓ Runtime:{R} {config.get_model_label()} · "
            f"Think {config.get_thinking_budget_label()} · {config.get_thinking_budget()} · "
            f"{config.get_server_mode()} :{config.get_server_port()}\n"
        )
        return "applied"

    _print_runtime_header()
    print_startup_hint()
    _reg = _load_skill_registry()
    _hard_cmds = [c["name"] for c in _reg.get("builtin_cmds", [])]
    setup_skill_completer([s["name"] for s in _reg.get("skills", [])] + _hard_cmds)
    restart_requested = False

    while True:
        if _sync_runtime_config_from_disk():
            print(
                f" {C_META}↻ synced Electron runtime config:{R} "
                f"{config.get_model_label()} · Think {config.get_thinking_budget_label()} · "
                f"{config.get_thinking_budget()} · {config.get_server_mode()} :{config.get_server_port()}\n"
            )
        q = prompt_user(mode=_active_skill)
        if q is None:
            # Ctrl+D (EOF) — deliberate exit gesture, same as /exit at top level.
            print("\n Bye.\n")
            if thread_id != _MEMORY_THREAD:
                _purge_thread(_db_conn, thread_id)
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "ออก", "บาย"):
            print("\n Bye.\n")
            if thread_id != _MEMORY_THREAD:
                _purge_thread(_db_conn, thread_id)
            break

        if _is_load_cmd(q):
            with Spinner("🧠  กำลังโหลด history…"):
                loaded_msgs, loaded_chars, total_pairs = _load_history_pairs(saver)
                if loaded_msgs:
                    app.update_state(cfg, {"messages": loaded_msgs})
                loaded_pairs = sum(1 for m in loaded_msgs if hasattr(m, "type") and m.type == "human")
                _hist_cfg = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": _MEMORY_THREAD}}
                hist = summarize_history(app, _hist_cfg)
            print(f" โหลด history แล้ว — {loaded_pairs}/{total_pairs} pairs ({loaded_chars:,} chars)\n")
            print(f" หัวข้อใน history:\n{hist['topics']}\n")
            continue

        if q.strip().lower() == "menu":
            print_mode_menu()
            choice = prompt_user() or ""
            if choice == "1":
                print("\n   ฟีเจอร์นี้ไม่รองรับในรุ่นนี้\n")
            elif choice == "2":
                _reg = _load_skill_registry()
                print_skill_help(_reg)
                _sc = prompt_user()
                if _sc:
                    _sn = _sc.lstrip("/").strip().lower()
                    if any(s["name"] == _sn for s in _reg.get("skills", [])):
                        _active_skill, _active_skill_content = _set_skill(_sn)
                    else:
                        print(f" ไม่พบ skill '{_sn}'\n")
            elif choice == "3":
                print_special_commands(_reg.get("builtin_cmds", []))
            elif choice == "4":
                while True:
                    settings = config.get_runtime_settings()
                    print_runtime_settings_menu(settings)
                    sub = (prompt_user() or "").strip().lower()
                    if sub in ("b", "back", "/exit"):
                        break
                    if sub == "1":
                        if settings.get("model_locked"):
                            print(f" {C_WARN}⚠ Model ถูกล็อกโดย shared server หรือ environment override{R}\n")
                            continue
                        options = settings.get("model_options", [])
                        print_runtime_choices("Model", options, settings.get("model"))
                        raw = (prompt_user() or "").strip().lower()
                        if raw in ("b", "back", "/exit"):
                            continue
                        try:
                            idx = int(raw) - 1
                            if idx < 0:
                                raise IndexError
                            selected = options[idx]["value"]
                        except (ValueError, IndexError, KeyError, TypeError):
                            print(f" {C_WARN}⚠ ตัวเลือก model ไม่ถูกต้อง{R}\n")
                            continue
                        if not _confirm_high_quality_model_on_low_ram(selected):
                            print(f" {C_META}ยกเลิกการเปลี่ยน model{R}\n")
                            continue
                        if _apply_runtime_selection(selected, config.get_thinking_budget()) == "restart":
                            restart_requested = True
                            break
                    elif sub == "2":
                        options = settings.get("thinking_options", [])
                        print_runtime_choices("Think Budget", options, settings.get("thinking_budget"))
                        raw = (prompt_user() or "").strip().lower()
                        if raw in ("b", "back", "/exit"):
                            continue
                        try:
                            idx = int(raw) - 1
                            if idx < 0:
                                raise IndexError
                            selected = int(options[idx]["value"])
                        except (ValueError, IndexError, KeyError, TypeError):
                            print(f" {C_WARN}⚠ ตัวเลือก Think Budget ไม่ถูกต้อง{R}\n")
                            continue
                        _apply_runtime_selection(config.get_model(), selected)
                    elif sub == "3":
                        mode_label = "External" if settings.get("shared_server") else "Standalone"
                        print(
                            f" {C_META}{mode_label} Model Server Port ปัจจุบัน :{config.get_server_port()} "
                            f"· ใส่ 1024–65535 หรือ b เพื่อย้อนกลับ{R}"
                        )
                        raw = (prompt_user() or "").strip().lower()
                        if raw in ("b", "back", "/exit"):
                            continue
                        try:
                            selected_port = int(raw)
                            if not 1024 <= selected_port <= 65535:
                                raise ValueError
                        except ValueError:
                            print(f" {C_WARN}⚠ Port ต้องเป็นตัวเลข 1024–65535{R}\n")
                            continue
                        _apply_runtime_selection(
                            config.get_model(), config.get_thinking_budget(), selected_port,
                        )
                    else:
                        print(f" {C_WARN}⚠ เลือก 1, 2, 3 หรือ b{R}\n")
                _print_runtime_header()
                if restart_requested:
                    print(" Bye — runtime model selection saved.\n")
                    if thread_id != _MEMORY_THREAD:
                        _purge_thread(_db_conn, thread_id)
                    break
            elif choice == "5":
                while True:
                    edit_state = _load_edit_access_state()
                    print_edit_access_menu(edit_state)
                    sub = (prompt_user() or "").strip().lower()
                    if sub in ("b", "back", "/exit"):
                        break
                    try:
                        if sub == "1":
                            raw = (prompt_user() or "").strip()
                            _apply_cli_edit_access_action("add", raw)
                            print(" Approved Edit Folder บันทึกแล้ว\n")
                        elif sub == "2":
                            folders = list(edit_state.get("folders") or [])
                            if not folders:
                                print(f" {C_META}ยังไม่มี Approved Edit Folder{R}\n")
                                continue
                            raw = (prompt_user() or "").strip()
                            idx = int(raw) - 1
                            if idx < 0 or idx >= len(folders):
                                raise ValueError("หมายเลขโฟลเดอร์ไม่ถูกต้อง")
                            _apply_cli_edit_access_action("remove", folders[idx])
                            print(" ลบ Approved Edit Folder แล้ว\n")
                        elif sub == "3":
                            raw = (prompt_user() or "").strip()
                            _apply_cli_edit_access_action("focus", raw)
                            print(" Active Workspace บันทึกแล้ว (เป็นสิทธิ์ชั่วคราว ไม่เพิ่ม approval)\n")
                        elif sub == "4":
                            _apply_cli_edit_access_action("clear_focus")
                            print(" ล้าง Active Workspace แล้ว\n")
                        else:
                            print(f" {C_WARN}⚠ เลือก 1, 2, 3, 4 หรือ b{R}\n")
                    except (OSError, TypeError, ValueError) as exc:
                        print(f" {C_WARN}⚠ แก้ไข edit access ไม่สำเร็จ: {exc}{R}\n")
            elif choice.lower() in ("q", "quit", "exit", "ออก", "บาย"):
                print("\n Bye.\n")
                if thread_id != _MEMORY_THREAD:
                    _purge_thread(_db_conn, thread_id)
                break
            continue

        if q.strip().lower() == "/build_kb":
            _run_build_kb()
            continue

        if q.strip().lower() == "/compact":
            with Spinner("🗜️  กำลังบีบอัด context…"):
                result = force_compact(app, cfg)
            if "error" in result:
                print(f" ⚠️  {result['error']}\n")
            else:
                _max = _ctx_stats.get("max_chars", 1) or 1
                pct_before = result["before"] / _max * 100
                pct_after  = result["after"]  / _max * 100
                print_compact_notice(result["cut"])
                print(f" บีบอัด context: {result['before']:,} → {result['after']:,} chars  ({pct_before:.0f}% → {pct_after:.0f}%)\n")
                _ctx_stats["chars"] = result["after"]
                update_ctx_info(result["after"], CONTEXT_MAX_CHARS)
                _seed_msgs = list(app.get_state(cfg).values.get("messages", []))
                _threading.Thread(target=rewarm_after_compact, args=(app, _db_conn, _seed_msgs), daemon=True).start()
            continue

        if q.strip().lower() == "/clear":
            if thread_id != _MEMORY_THREAD:
                _purge_thread(_db_conn, thread_id)
            thread_id = str(uuid.uuid4())
            cfg = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": thread_id}}
            _ctx_stats["chars"] = 0
            update_ctx_info(0, CONTEXT_MAX_CHARS)
            print(" เคลียร์ context แล้ว — เริ่ม session ใหม่\n")
            continue

        if q.startswith("/"):
            _sname = q[1:].strip().lower()
            if _sname in ("exit", "quit") or _sname == _active_skill:
                if _active_skill:
                    print(f" ปิด {_active_skill} mode\n")
                _active_skill, _active_skill_content = _set_skill("")
            else:
                _active_skill, _active_skill_content = _set_skill(_sname)
            continue

        actual_q = (
            f"[SKILL: {_active_skill}]\n{_active_skill_content}\n---\n{q}"
            if _active_skill else q
        )
        if not _ensure_server_alive():
            continue
        _run_turn(app, actual_q, cfg, thread_id=thread_id, saver=saver, db_conn=_db_conn,
                  logger=logger, system_prompt=get_system_prompt())

        # Reload completer in case agent created a new skill this turn (/build)
        _reg = _load_skill_registry()
        _hard_cmds = [c["name"] for c in _reg.get("builtin_cmds", [])]
        setup_skill_completer([s["name"] for s in _reg.get("skills", [])] + _hard_cmds)

        update_ctx_info(_ctx_stats["chars"], CONTEXT_MAX_CHARS)
        compact_n = _ctx_stats["compact_msg"]
        _ctx_stats["compact_msg"] = None
        if compact_n:
            print_compact_notice(compact_n)




if __name__ == "__main__":
    main()
