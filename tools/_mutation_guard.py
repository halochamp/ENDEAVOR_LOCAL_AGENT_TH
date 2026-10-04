"""Small normal-turn filesystem-intent and read-before-mutate guards."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


_ACTION = re.compile(
    r"^\s*(?:(?:please|can you|could you|would you|i need you to|help me|please help me)\s+)*"
    r"(?:fix|repair|correct|patch|refactor|edit|modify|update|change|create|write|save|format|delete|remove)\b",
    re.IGNORECASE,
)
_THAI_ACTION = re.compile(r"^\s*(?:(?:กรุณา|ขอให้)\s*)?(?:ช่วย\s*)?(?:แก้ไข|แก้|ปรับปรุง|อัปเดต|สร้าง|เขียน|ลบ)")
_TREE_SCOPE = re.compile(
    r"\b(?:workspace|repo(?:sitory)?|codebase|project|website|site)\b|"
    r"(?:ทั้งโปรเจกต์|ทั้งโฟลเดอร์|ทั้ง workspace|ทั้ง repo|ทั้งเว็บไซต์)",
    re.IGNORECASE,
)
_EXPLANATORY_OBJECT = re.compile(
    r"^\s*(?:(?:please|can you|could you)\s+)*(?:fix|repair|correct|patch)\s+"
    r"(?:my|our)\s+(?:understanding|knowledge|mental model)\b",
    re.IGNORECASE,
)
_EXPLICIT_FILE = re.compile(
    r"(?<![\w@])(?:~|/)?[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)*\.[A-Za-z][A-Za-z0-9_-]{0,11}(?![\w])"
)
_EXPLICIT_SLASH_PATH = re.compile(r"(?<![\w@])(?:~|/)?[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)+(?![\w])")
_EXPLICIT_ABSOLUTE = re.compile(r"(?<![\w])(?:~|/)[^\s\"'<>|;]+")
_QUOTED = re.compile(r"(?:\"[^\"\n]*\"|'[^'\n]*'|“[^”\n]*”|‘[^’\n]*’)")
_URL = re.compile(r"\bhttps?://[^\s<>\"']+", re.IGNORECASE)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_BLOCKQUOTE = re.compile(r"(?m)^\s*>.*$")
_SHELL_MUTATOR = re.compile(
    r"(?:^|[;&|]\s*)(?:touch|mkdir|rmdir|rm|mv|cp|ln|install|tee|truncate)\b|"
    r"\b(?:sed|perl)\b[^\n]*(?:\s-i(?:\s|$)|--in-place)|"
    r"\b(?:git)\s+(?:apply|checkout|clean|reset|restore|rm|mv)\b|"
    r"\b(?:Path|os)\b[^\n]*\.(?:write_text|write_bytes|unlink|mkdir|rmdir|remove|rename|replace)\s*\(|"
    r"\b(?:os\.(?:remove|unlink|rename|replace|mkdir|rmdir)|shutil\.(?:copy|move|rmtree|copytree))\s*\(|"
    r"\bopen\s*\([^\n,]+,\s*[\"'](?:w|a|x|\+)|"
    r"(?<![0-9])>>?\s*(?!/dev/null\b|/tmp/|/private/tmp/)[^\s;&|]+"
)
_SYMLINK_CREATOR = re.compile(
    r"\b(?:ln|cp)\s+-[^\s;&|]*s[^\s;&|]*(?:\s|$)|"
    r"\b(?:os\.)?symlink\s*\(|\.symlink_to\s*\(",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MutationIntent:
    kind: str  # none | exact | tree | blocked_explicit
    targets: tuple[str, ...] = ()
    tree_root: str = ""
    reason: str = ""


def _text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", "")) for block in content
            if isinstance(block, dict) and block.get("type", "text") == "text"
        )
    return str(content or "")


def _unquoted_text(text: str) -> str:
    text = _CODE_FENCE.sub(" ", text)
    text = _BLOCKQUOTE.sub(" ", text)
    return _QUOTED.sub(" ", text)


def _action_text(text: str) -> str:
    plain = _unquoted_text(text)
    plain = re.sub(r"^\s*(?:(?:effective[- ]action|requested action)\s*:\s*|\[[^\]\n]{1,60}\]\s*)+", "", plain, flags=re.IGNORECASE)
    return plain


def _has_action(text: str) -> bool:
    plain = _action_text(text).strip()
    return bool(_ACTION.search(plain) or _THAI_ACTION.search(plain))


def _mentioned_targets(text: str) -> list[str]:
    # Keep inline-code filenames, but ignore quoted prose and code blocks.
    plain = _BLOCKQUOTE.sub(" ", _CODE_FENCE.sub(" ", text))
    plain = _unquoted_text(plain)
    plain = _URL.sub(" ", plain)
    matches = [m.group(0).rstrip(".,:!?)]}") for m in _EXPLICIT_ABSOLUTE.finditer(plain)]
    matches.extend(m.group(0).rstrip(".,:!?)]}") for m in _EXPLICIT_FILE.finditer(plain))
    matches.extend(m.group(0).rstrip(".,:!?)]}") for m in _EXPLICIT_SLASH_PATH.finditer(plain))
    result: list[str] = []
    for value in matches:
        if value and value not in result:
            result.append(value)
    return result


def _focus_or_workspace() -> str:
    from config import WORKSPACE
    from .edit_access import get_focus_folder
    return os.path.realpath(get_focus_folder() or WORKSPACE)


def classify_mutation_intent(messages: list[Any]) -> MutationIntent:
    """Classify the latest user request; unsafe explicit targets never become tree scope."""
    human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if human is None:
        return MutationIntent("none")
    original = _text(human)
    action_text = _action_text(original)
    if not (_ACTION.search(action_text) or _THAI_ACTION.search(action_text)):
        return MutationIntent("none")
    if _EXPLANATORY_OBJECT.search(action_text):
        return MutationIntent("none")

    mentioned = _mentioned_targets(original)
    if mentioned:
        from ._safety import validate_write_target
        resolved: list[str] = []
        for raw in mentioned:
            path, error = validate_write_target(raw)
            if error:
                return MutationIntent("blocked_explicit", reason=error)
            canonical = os.path.abspath(path)
            if canonical not in resolved:
                resolved.append(canonical)
        return MutationIntent("exact", tuple(resolved))

    if _TREE_SCOPE.search(action_text):
        from ._safety import validate_write_target
        root = _focus_or_workspace()
        path, error = validate_write_target(root)
        if error:
            return MutationIntent("blocked_explicit", reason=error)
        return MutationIntent("tree", tree_root=os.path.abspath(path))

    return MutationIntent("none")


def _request_messages(request) -> list[Any]:
    state = request.state
    if isinstance(state, dict):
        return list(state.get("messages") or [])
    return list(getattr(state, "messages", None) or [])


def _arg_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str) and item]
    return []


def _canonical_read_path(path: str) -> str | None:
    try:
        from ._safety import resolve_read_path
        return os.path.abspath(resolve_read_path(path))
    except (OSError, ValueError, PermissionError):
        return None


def successful_current_reads(messages: list[Any], current_call_id: str) -> set[str]:
    """Paths returned by a successful read_file call earlier in this user turn."""
    start = -1
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            start = idx
            break
    prior_calls: dict[str, tuple[str, Any]] = {}
    successful: set[str] = set()
    for message in messages[start + 1:]:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", None) or []:
                call_id = str(call.get("id") or "")
                if call_id == current_call_id:
                    return successful
                prior_calls[call_id] = (str(call.get("name") or ""), call.get("args") or {})
        elif isinstance(message, ToolMessage):
            name, args = prior_calls.get(str(message.tool_call_id), ("", {}))
            if name != "read_file" or message.name not in (None, "read_file"):
                continue
            result = _text(message)
            if not result.strip() or result.lstrip().startswith("[error]") or "[error] read_file failed" in result:
                continue
            paths = _arg_paths(args.get("path") if isinstance(args, dict) else None)
            # A batch read may have mixed errors; accept only a single exact file
            # result so one item's success cannot authorize another item's write.
            if len(paths) != 1:
                continue
            canonical = _canonical_read_path(paths[0])
            if canonical:
                successful.add(canonical)
    return successful


def has_current_read(messages: list[Any], call_id: str, target: str) -> bool:
    return os.path.abspath(target) in successful_current_reads(messages, call_id)


def tool_mutates_bypass(name: str, args: dict[str, Any]) -> bool:
    """Nested executor routes which cannot inherit a per-call Bash capability safely."""
    if name == "python_exec":
        return True
    if name == "bash_bg":
        return str(args.get("action", "start")) == "start"
    return False


def shell_command_looks_mutating(command: str) -> bool:
    """Conservative command-shape guard for obvious writes without user intent."""
    return bool(_SHELL_MUTATOR.search(str(command or "")))


def shell_command_creates_symlink(command: str) -> bool:
    """Recognize common explicit symlink-creation primitives for fail-closed handling."""
    return bool(_SYMLINK_CREATOR.search(str(command or "")))
