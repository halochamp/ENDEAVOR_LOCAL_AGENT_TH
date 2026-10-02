"""react.py — ReAct agent (create_react_agent) — main agent ที่คุมทุกอย่าง

ใน V2 ใหม่: main agent ตัวเดียวคุมทั้งระบบ
- เห็น query ง่าย → ตอบเลย หรือเรียก tool เดี่ยว
- เห็น query ซับซ้อน → เรียก create_plan ก่อน → ทำ steps ด้วย tool อื่น → รวมคำตอบ
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import inspect
import datetime
import json as _json
import logging
import os
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from config import CONTEXT_MAX_CHARS

from llm import build_llm
from runtime_common import ToolLoopDetected
from tools import ALL_TOOLS
from system_prompt import SYSTEM

log = logging.getLogger(__name__)


# Generic consecutive-call circuit breaker. The second identical call returns
# a corrective hint without executing the expensive/side-effectful tool; a
# third identical call stops the turn deterministically.
_REPEAT_TOOL_HINT_TEXT = (
    "[tool_loop_hint] เรียก {name} ด้วย query/arguments เดิมซ้ำ 2 รอบติด; "
    "รอบนี้จึงไม่รัน tool ซ้ำ. ต้องเปลี่ยน query/arguments หรือเปลี่ยนเครื่องมือก่อนทำต่อ. "
    "ห้ามเรียกคำขอเดิมติดกันอีก; ครั้งที่ 3 จะหยุด turn เพื่อป้องกัน loop."
)


def _normalise_web_query(value):
    """Canonicalize web_search query text so spacing/case cannot bypass the guard."""
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, list):
        return [_normalise_web_query(item) for item in value]
    return value


def _tool_call_fingerprint(tool_call: dict) -> str:
    """Stable loop identity independent of the provider-generated call id."""
    name = str(tool_call.get("name") or "")
    args = tool_call.get("args") or {}
    identity = args
    if name == "web_search" and isinstance(args, dict) and "query" in args:
        identity = {"query": _normalise_web_query(args.get("query"))}
    return _json.dumps(
        {"name": name, "identity": identity},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _prior_tool_calls_in_turn(messages: list, call_id: str) -> list[dict] | None:
    """Return tool-call attempts before ``call_id`` in the current user turn."""
    turn_start = -1
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            turn_start = idx
            break

    prior: list[dict] = []
    for msg in messages[turn_start + 1:]:
        if not isinstance(msg, AIMessage):
            continue
        for candidate in (getattr(msg, "tool_calls", None) or []):
            if str(candidate.get("id") or "") == call_id:
                return prior
            prior.append(candidate)
    return None


def _guard_repeated_tool_call(request, execute):
    """Hint on the second identical consecutive call; stop on the third."""
    call = request.tool_call or {}
    call_id = str(call.get("id") or "")
    name = str(call.get("name") or "tool")
    if not call_id:
        return execute(request)

    state = request.state
    if isinstance(state, dict):
        messages = list(state.get("messages") or [])
    else:
        messages = list(getattr(state, "messages", None) or [])
    if not messages:
        return execute(request)

    prior = _prior_tool_calls_in_turn(messages, call_id)
    if prior is None:
        return execute(request)

    fingerprint = _tool_call_fingerprint(call)
    same_before = 0
    for candidate in reversed(prior):
        if _tool_call_fingerprint(candidate) != fingerprint:
            break
        same_before += 1

    if same_before == 0:
        return execute(request)
    if same_before == 1:
        log.warning(
            "[tool-loop-guard] duplicate call hint for %s: %s",
            name, fingerprint[:500],
        )
        return ToolMessage(
            content=_REPEAT_TOOL_HINT_TEXT.format(name=name),
            tool_call_id=call_id,
            name=name,
        )

    log.error(
        "[tool-loop-guard] stopping turn after third consecutive %s call: %s",
        name, fingerprint[:500],
    )
    raise ToolLoopDetected(name)


def _tool_message_block(request, message: str):
    call = request.tool_call or {}
    return ToolMessage(
        content=message,
        tool_call_id=str(call.get("id") or ""),
        name=str(call.get("name") or "tool"),
    )


def _filesystem_calls_are_parallel(messages: list, call_id: str, intent_kind: str) -> bool:
    """ToolNode may execute one AIMessage's calls concurrently; serialize writes by refusing batches."""
    if not call_id:
        return False
    for message in reversed(messages):
        calls = getattr(message, "tool_calls", None) or []
        if not any(str(item.get("id") or "") == call_id for item in calls):
            continue
        writers = []
        for item in calls:
            name = str(item.get("name") or "")
            args = item.get("args") or {}
            args = args if isinstance(args, dict) else {}
            writes = name in {"edit", "write_file"}
            writes = writes or (intent_kind in {"exact", "tree"} and name in {"bash", "python_exec"})
            writes = writes or (name == "bash_bg" and str(args.get("action", "start")) == "start")
            writes = writes or (name == "tool_loop" and (
                bool(args.get("output_file")) or str(args.get("action", "")) == "bash_each"
            ))
            if writes:
                writers.append(item)
        return len(writers) > 1
    return False


def _write_roots_for_mutation(intent, authorized_roots: list[str]) -> tuple[str, ...]:
    """Narrow shared authorization to the root(s) implicated by this mutation."""
    from tools._safety import _is_within

    if intent.kind == "tree":
        root = os.path.realpath(os.path.abspath(intent.tree_root))
        return (root,) if any(_is_within(root, allowed) for allowed in authorized_roots) else ()
    if intent.kind != "exact":
        return ()

    selected: list[str] = []
    for target in intent.targets:
        candidates = [root for root in authorized_roots if _is_within(target, root)]
        if not candidates:
            return ()
        root = max(candidates, key=len)
        if root not in selected:
            selected.append(root)
    return tuple(selected)


def _guard_filesystem_tool_call(request, execute):
    """Enforce shared write access and current-turn reads at the ToolNode boundary."""
    from pathlib import Path
    from tools import edit_access
    from tools._mutation_guard import (
        classify_mutation_intent,
        has_current_read,
        shell_command_creates_symlink,
        shell_command_looks_mutating,
        tool_mutates_bypass,
    )
    from tools._safety import (
        _authorized_write_roots,
        validate_write_target,
    )

    call = request.tool_call or {}
    name = str(call.get("name") or "")
    args = call.get("args") or {}
    args = args if isinstance(args, dict) else {}
    call_id = str(call.get("id") or "")
    state = request.state
    messages = list(state.get("messages") or []) if isinstance(state, dict) else list(getattr(state, "messages", None) or [])
    intent = classify_mutation_intent(messages)

    if _filesystem_calls_are_parallel(messages, call_id, intent.kind):
        return _tool_message_block(
            request,
            "[BLOCKED] submit one filesystem mutation tool call at a time so each change can be checked safely",
        )

    if intent.kind == "none" and name == "bash" and shell_command_looks_mutating(args.get("command", "")):
        return _tool_message_block(
            request,
            "[BLOCKED] a requested file change must be explicit so the host can apply shared workspace guards",
        )

    blocked_executor = name in {"edit", "write_file", "bash", "python_exec"}
    blocked_executor = blocked_executor or (
        name == "bash_bg" and str(args.get("action", "start")) == "start"
    )
    blocked_executor = blocked_executor or (
        name == "tool_loop" and (
            bool(args.get("output_file")) or str(args.get("action", "")) == "bash_each"
        )
    )
    if intent.kind == "blocked_explicit" and blocked_executor:
        return _tool_message_block(request, intent.reason or "[BLOCKED] unsafe explicit mutation target")

    if name in {"edit", "write_file"}:
        raw_path = args.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return _tool_message_block(request, "[BLOCKED] a file path is required")
        target, error = validate_write_target(raw_path)
        if error:
            return _tool_message_block(request, error)
        target_abs = str(Path(target).absolute())
        if intent.kind == "exact" and target_abs not in intent.targets:
            return _tool_message_block(request, "[BLOCKED] file target does not match the user's requested file")
        exists = Path(target).exists() or Path(target).is_symlink()
        if name == "edit" and exists and args.get("mode", "") == "create":
            return _tool_message_block(request, "[BLOCKED] create mode cannot overwrite an existing target")
        replaces_existing = exists and (
            name == "edit" or bool(args.get("overwrite"))
        )
        if replaces_existing:
            if Path(target).is_symlink() or not Path(target).is_file():
                return _tool_message_block(request, "[BLOCKED] existing mutation target must be a regular non-symlink file")
            if not call_id or not has_current_read(messages, call_id, target_abs):
                return _tool_message_block(
                    request,
                    "[BLOCKED] read the existing target with read_file earlier in this turn before modifying it",
                )
        return execute(request)

    if intent.kind in {"exact", "tree"} and tool_mutates_bypass(name, args):
        return _tool_message_block(
            request,
            "[BLOCKED] use edit or guarded bash for this requested file change; nested/background/Python execution cannot receive mutation authority",
        )

    if name == "tool_loop" and args.get("output_file"):
        from tools._safety import resolve_path, validate_write_target
        fname = os.path.basename(str(args.get("output_file") or "")) or "output.md"
        if "." not in fname:
            fname += ".md"
        target, error = validate_write_target(fname)
        if error:
            return _tool_message_block(request, error)
        if intent.kind == "exact" and os.path.abspath(target) not in intent.targets:
            return _tool_message_block(request, "[BLOCKED] loop output does not match the user's requested file")
        if os.path.exists(target):
            if not call_id or not has_current_read(messages, call_id, os.path.abspath(target)):
                return _tool_message_block(
                    request,
                    "[BLOCKED] read the existing loop output target with read_file earlier in this turn",
                )

    if name == "tool_loop" and intent.kind == "none" and str(args.get("action", "")) == "bash_each":
        items = args.get("items") or []
        if isinstance(items, list) and any(shell_command_looks_mutating(item) for item in items):
            return _tool_message_block(request, "[BLOCKED] background file writes require an explicit mutation request")

    if name == "bash_bg" and intent.kind == "none" and str(args.get("action", "start")) == "start":
        if shell_command_looks_mutating(args.get("command", "")):
            return _tool_message_block(request, "[BLOCKED] background file writes require an explicit mutation request")

    if name == "bash" and intent.kind in {"exact", "tree"}:
        if shell_command_creates_symlink(args.get("command", "")):
            return _tool_message_block(
                request,
                "[BLOCKED] symlink creation is not allowed during a guarded Bash mutation",
            )
        if intent.kind == "exact":
            for target in intent.targets:
                path, error = validate_write_target(target)
                if error:
                    return _tool_message_block(request, error)
                p = Path(path)
                if p.exists() or p.is_symlink():
                    if p.is_symlink() or not p.is_file():
                        return _tool_message_block(request, "[BLOCKED] existing requested target must be a regular non-symlink file")
                    if not call_id or not has_current_read(messages, call_id, str(p.absolute())):
                        return _tool_message_block(
                            request,
                            f"[BLOCKED] read the existing requested target with read_file before Bash mutation: {p.name}",
                        )
        roots = _write_roots_for_mutation(intent, _authorized_write_roots())
        if not roots:
            return _tool_message_block(request, "[BLOCKED] no authorized filesystem write root matches this mutation")
        scope = {
            "kind": intent.kind,
            "targets": tuple(intent.targets),
            "tree_root": intent.tree_root,
            "write_roots": tuple(roots),
        }
        def invoke_in_scope(req):
            with edit_access.bash_mutation_scope(scope):
                result = execute(req)
                if inspect.isawaitable(result):
                    async def await_with_scope():
                        with edit_access.bash_mutation_scope(scope):
                            return await result
                    return await_with_scope()
                return result
        return invoke_in_scope(request)

    return execute(request)


def _guard_tool_call(request, execute):
    """Compose filesystem policy with the existing duplicate-call circuit breaker."""
    return _guard_repeated_tool_call(
        request,
        lambda req: _guard_filesystem_tool_call(req, execute),
    )


# Main agent system prompt — Thai output, tool guidance, complex-routing, synthesize rules
# System prompt is the tracked plaintext Python module at the repository root.

_BUILT_PROMPT: str = ""
_VISION_PUBLICATION_SUSPENDED = ContextVar(
    "endeavor_vision_publication_suspended", default=False
)


@contextmanager
def suspend_vision_publication():
    """Keep a utility/cache-warm invocation from consuming live tool pixels."""
    token = _VISION_PUBLICATION_SUSPENDED.set(True)
    try:
        yield
    finally:
        _VISION_PUBLICATION_SUSPENDED.reset(token)

# Keep this small, deterministic overlay in ordinary source so the old
# OCR/text-only contract is explicitly superseded.
_DIRECT_VISION_OVERLAY = (
    "\n\n[PUBLIC DIRECT-VISION CONTRACT]\n"
    "- Route explicit image-understanding requests to read_image and explicit screen/application actions to computer.\n"
    "- read_image is progressive direct vision: a source-only call exposes the whole original image to the main VLM first, without an automatic OCR/classifier/table/QR pass.  Only after that same source is visible may you request detail=text, detail=chart, detail=slide, find, or region/zoom assistance.  Do not pass a semantic question or prompt argument; the conversation supplies semantics.\n"
    "- If the configured backend rejects image input, read_image automatically switches to full OCR and returns a clearly marked [TEXT-ONLY IMAGE FALLBACK].  Use that OCR in the same turn without asking the user to retry, and never invent visual facts that OCR does not contain.\n"
    "- computer is an independent direct-vision/action tool.  It owns its current screenshot and observation lifecycle, while read_image owns its own image lifecycle.  Their queues, guards, and snapshots are never shared; the main model receives whichever pixels each tool publishes.\n"
    "- computer requires a vision-capable model.  If it returns [unsupported] because the current model is text-only, tell the user clearly and do not substitute OCR or retry desktop actions.\n"
)

# Shared per-turn context stats — updated by graph.py before each react_node invocation
ctx_stats: dict = {
    "chars": 0,
    "max_chars": CONTEXT_MAX_CHARS,
    "cooldown": False,      # True after compact — wait for pct < 70% before next compact
    "compact_msg": None,    # set to int (n_msgs) when compact fires; endeavor_agent.py reads + clears
    "compact_before": 0,    # chars before compact (for UI display)
}


def get_system_prompt() -> str:
    """Return the system prompt injected in the last build_react_agent call — for logging."""
    return _BUILT_PROMPT


def _make_ctx_note() -> str:
    chars = ctx_stats["chars"]
    max_c = ctx_stats["max_chars"]
    pct = chars / max_c * 100 if max_c > 0 else 0
    if pct >= 90:
        bucket = "NEAR LIMIT: be concise, avoid long code blocks"
    elif pct >= 70:
        bucket = "high"
    elif pct >= 50:
        bucket = "moderate"
    else:
        bucket = "ok"
    today = datetime.date.today().strftime("%Y-%m-%d")
    return f"\n\n[Today: {today}] [Context window: {bucket}]"


def prepare_turn_vision_messages(messages: list) -> list:
    """Attach currently published tool pixels to the last human message.

    Both tools publish only transient data URLs; promoting their pending queues
    here keeps the pixels visible to every remaining ReAct step and to a
    same-turn synthesis retry without persisting image blocks in graph state.
    """
    if _VISION_PUBLICATION_SUSPENDED.get():
        return list(messages)

    read_images: list[str] = []
    computer_images: list[str] = []
    try:
        from tools.read_image import active_turn_images
        read_images = list(active_turn_images())
    except Exception:
        log.debug("read_image image publication unavailable", exc_info=True)
    try:
        from tools.computer_use import active_computer_turn_images
        computer_images = list(active_computer_turn_images())
    except Exception:
        log.debug("computer image publication unavailable", exc_info=True)
    published = [url for url in read_images + computer_images if url]
    if not published:
        return list(messages)

    msgs = list(messages)
    human_index = next(
        (i for i in range(len(msgs) - 1, -1, -1) if _is_human_message(msgs[i])),
        None,
    )
    if human_index is None:
        log.warning("published image data had no HumanMessage recipient")
        return msgs

    from langchain_core.messages import HumanMessage
    message = msgs[human_index]
    existing_urls: set[str | None] = set()
    if isinstance(message.content, list):
        for block in message.content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            image_url = block.get("image_url")
            if isinstance(image_url, dict):
                existing_urls.add(image_url.get("url"))
    image_blocks = [
        {"type": "image_url", "image_url": {"url": url}}
        for url in published
        if url not in existing_urls
    ]
    if not image_blocks:
        return msgs
    if isinstance(message.content, list):
        content = list(message.content) + image_blocks
    else:
        content = [{"type": "text", "text": message.content}] + image_blocks
    msgs[human_index] = HumanMessage(content=content, id=getattr(message, "id", None))
    return msgs


def _is_human_message(message) -> bool:
    from langchain_core.messages import HumanMessage
    return isinstance(message, HumanMessage)


def build_react_agent(checkpointer=None, memory: str = "", tools=None, **llm_overrides):
    """สร้าง ReAct agent หลัก. memory = เนื้อหาจาก memory.md (โหลดตอน startup)
    tools: override ALL_TOOLS เช่น กรณี offline mode"""
    from langchain_core.messages import SystemMessage as _SM

    llm = build_llm(**llm_overrides)
    active_tools = tools if tools is not None else ALL_TOOLS
    active_names = {t.name for t in active_tools}

    # If web tools are disabled → notify agent clearly that there is no internet
    offline_note = ""
    if "web_search" not in active_names:
        offline_note = (
            "\n\n!! OFFLINE MODE: no internet access"
            "\n- Never use bash to fetch URLs or curl any external endpoint"
            "\n- If user asks for real-time data (prices, news, web content) → reply directly: 'ไม่มีอินเทอร์เน็ตตอนนี้ ไม่สามารถดึงข้อมูลได้'"
            "\n- Never fabricate or guess data that requires internet access"
        )

    global _BUILT_PROMPT
    system = SYSTEM + _DIRECT_VISION_OVERLAY
    base_prompt = (f"## Your memory about this user:\n{memory}\n\n---\n\n" + system + offline_note) if memory else (system + offline_note)
    _BUILT_PROMPT = base_prompt

    # Dynamic prompt callable — injects published pixels and ctx_note into the
    # last HumanMessage (not system), so image tool output is visible in the
    # same outer turn while the system message stays byte-identical.
    def dynamic_prompt(state: dict) -> list:
        from langchain_core.messages import HumanMessage
        msgs = prepare_turn_vision_messages(list(state.get("messages", [])))
        ctx_note = _make_ctx_note()
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                m = msgs[i]
                if isinstance(m.content, list):
                    new_content = list(m.content) + [{"type": "text", "text": ctx_note}]
                else:
                    new_content = m.content + ctx_note
                msgs[i] = HumanMessage(content=new_content, id=getattr(m, "id", None))
                break
        return [_SM(content=base_prompt)] + msgs

    tool_node = ToolNode(active_tools, wrap_tool_call=_guard_tool_call)
    return create_react_agent(llm, tool_node, prompt=dynamic_prompt, checkpointer=checkpointer)
