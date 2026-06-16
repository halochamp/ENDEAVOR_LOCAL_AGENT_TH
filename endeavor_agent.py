# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""endeavor_agent.py — ENDEAVOR_AGENT_V2 interactive CLI

รัน:  conda activate mlx && python endeavor_agent.py
สลับ model: export V2_MODEL=... MLX_BASE_URL=...  (ดู config.py)
"""
from __future__ import annotations
import json as _json
import os
import uuid

from langchain_core.callbacks import BaseCallbackHandler

from graph import build_graph, force_compact, rewarm_after_compact, summarize_history
from react import get_system_prompt, ctx_stats as _ctx_stats
from config import RECURSION_LIMIT, MODEL, MLX_BASE_URL, CONTEXT_MAX_CHARS
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
from tools._progress import set_callback as set_progress_callback, set_phase_callback, set_plan_callback
from tools.web_cache import web_count_reset as _reset_web_counter
from tools.scratchpad import _PAD as _scratch_pad
from agent_log import AgentLogger
from ui_cli import (
    Spinner, print_header, print_divider, print_user_prompt,
    print_tool_step, print_plan, print_synthesizing,
    print_agent_response, print_web_refs,
    extract_refs_from_search_result, prompt_user,
    print_mode_menu, print_special_commands, print_skill_help,
    print_startup_hint,
    setup_skill_completer, update_ctx_info, print_compact_notice,
    _SPINNER_LABELS,
    PHASE_THINKING, PHASE_EXECUTING, PHASE_SYNTH,
)


_WEB_TOOLS = {"web_search", "browse_url", "browser_use", "recall_web",
              "fetch_sitemap", "batch_browse", "scrape_table",
              "research_orchestrator"}


def _get_skill_tools(skill: str, online: bool) -> list:
    tools = SKILL_TOOLS.get(skill, [])
    if not online:
        return [t for t in tools if t.name not in _WEB_TOOLS]
    return tools


def _ensure_server_alive(timeout: int = 30, interval: int = 3) -> bool:
    """Check mid-session that MLX server is reachable; wait up to `timeout` seconds to reconnect."""
    if _server_up():
        return True
    print(f"\n⚠️  LLM server หายไป — รอ reconnect (สูงสุด {timeout}s)...")
    elapsed = 0
    while elapsed < timeout:
        import time as _time
        _time.sleep(interval)
        elapsed += interval
        if _server_up():
            print("✅ Server กลับมาแล้ว\n")
            return True
        print(f"   ยังไม่ได้ ({elapsed}s)...")
    print(f"❌ Server ไม่กลับมาใน {timeout}s — ข้าม turn นี้ไป\n")
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
    if name == "scratch_write":
        key = args.get("key", "")
        val = str(args.get("value", ""))
        return f"[{key}] {val[:40]}" if key else val[:55]
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


_PLAN_SKIP = {"scratch_write", "scratch_read", "scratch_clear", "create_plan"}


class _Turn:
    """Mutable UI state shared between _run_turn and _UICallback."""

    def __init__(self) -> None:
        self.spinner: "Spinner | None" = None
        self.active = False
        self.web_refs: list = []
        self.final = ""
        self.plan_steps: list[str] = []   # from create_plan output
        self.plan_step_idx: int = 0       # non-scratch tool calls since create_plan

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
            clear_scratch=_scratch_pad.clear, reset_web_counter=_reset_web_counter,
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


def main() -> None:
    if not _server_up():
        print(f"[!] เชื่อม mlx_lm.server ไม่ได้ที่ {MLX_BASE_URL}")
        print(f"    เริ่ม server ก่อน: mlx_lm.server --model {MODEL} --port <port>")
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

    import threading as _threading
    # Pre-warm mlx_lm.server prefix cache with [system_prompt] — runs in background so the
    # user's real first turn hits a cached system segment instead of a cold ~16k prefill
    # (V2-PF01: confirmed cross-session — fresh session's first turn measured 404 tokens
    # when an identical system prompt was already cached). Disposable thread_id, purged after.
    _WARM_THREAD_ID = "__cache_warm__"
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
            _purge_thread(_db_conn, _WARM_THREAD_ID)
    _threading.Thread(target=_warm_llm_cache, daemon=True).start()

    print_header(MODEL, len(active_tools), online=online)
    print_startup_hint()
    _reg = _load_skill_registry()
    _hard_cmds = [c["name"] for c in _reg.get("builtin_cmds", [])]
    setup_skill_completer([s["name"] for s in _reg.get("skills", [])] + _hard_cmds)

    while True:
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
            choice = prompt_user()
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
            elif choice.lower() in ("q", "quit", "exit", "ออก", "บาย"):
                print("\n Bye.\n")
                if thread_id != _MEMORY_THREAD:
                    _purge_thread(_db_conn, thread_id)
                break
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
