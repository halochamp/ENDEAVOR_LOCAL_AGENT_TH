# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Generic MCP client over Streamable HTTP or guarded local stdio.

Two registries are merged by name: ``config.MCP_SERVERS`` for optional
user/developer configuration and ``workspace/tool_mcp/servers.json`` for
agent self-service. Workspace entries win on collisions. HTTP connections use
the official MCP SDK. Local stdio servers are spawned without a shell and run
under the same macOS sandbox profile as this agent's ``bash`` tool.

This public fork intentionally ships with no bundled/default MCP servers and no
extra write scopes outside its normal workspace + /tmp sandbox.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shutil
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Callable, Coroutine, TypeVar

from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MCP_MAX_CHARS, MCP_SERVERS, MCP_TIMEOUT, WORKSPACE
from tools.bash import _build_sandbox_profile

_T = TypeVar("_T")
_SANDBOX_EXEC = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((os.path.realpath(path), os.path.realpath(root))) == os.path.realpath(root)
    except ValueError:
        return False


def _dynamic_path() -> Path:
    return Path(WORKSPACE) / "tool_mcp" / "servers.json"


def _load_dynamic_servers(*, strict: bool = False) -> dict:
    """Read the self-service registry; mutation callers may request strict parsing."""
    try:
        data = json.loads(_dynamic_path().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            if strict:
                raise ValueError("dynamic MCP registry must contain a JSON object")
            return {}
        return data
    except FileNotFoundError:
        return {}
    except Exception as exc:
        if strict:
            raise ValueError(f"dynamic MCP registry is invalid: {exc}") from exc
        return {}


def _all_servers() -> dict:
    return {**MCP_SERVERS, **_load_dynamic_servers()}


def _save_dynamic_servers(servers: dict) -> None:
    """Atomically save a private registry; caller holds ``_dynamic_registry_lock``."""
    path = _dynamic_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(servers, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _dynamic_registry_lock():
    """Serialize cross-thread/process registry read-modify-write cycles."""
    path = _dynamic_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    handle = open(lock_path, "a+")
    try:
        try:
            os.chmod(lock_path, 0o600)
        except OSError:
            pass
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _known_servers() -> str:
    return ", ".join(sorted(_all_servers())) or "(none configured)"


def _cap(text: str) -> str:
    if len(text) <= MCP_MAX_CHARS:
        return text
    return text[:MCP_MAX_CHARS] + f"\n...[truncated at {MCP_MAX_CHARS} chars]"


def _compact_tool_list(tools) -> str:
    """Bound list output while preserving every discovered tool name."""
    tools = list(tools)
    if not tools:
        return "(no tools exposed)"

    rows = [
        (str(item.name), " ".join((item.description or "(no description)").split()))
        for item in tools
    ]
    marker = "\n[descriptions compacted so every tool name remains visible]"
    names_only = "\n".join(name for name, _ in rows)
    if len(names_only) + len(marker) >= MCP_MAX_CHARS:
        return names_only + marker

    fixed = sum(len(name) + 2 for name, _ in rows) + max(0, len(rows) - 1) + len(marker)
    description_budget = max(0, MCP_MAX_CHARS - fixed)
    per_tool = description_budget // len(rows)
    clipped = False
    rendered: list[str] = []
    for name, description in rows:
        if per_tool and len(description) > per_tool:
            description = description[: max(0, per_tool - 1)].rstrip() + "…"
            clipped = True
        elif not per_tool and description:
            description = ""
            clipped = True
        rendered.append(f"{name}: {description}" if description else name)
    text = "\n".join(rendered)
    return text + marker if clipped else text


def _normalise_server_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise ValueError("MCP server configuration must be an object")

    url = str(cfg.get("url") or "").strip()
    command = str(cfg.get("command") or "").strip()
    transport = str(
        cfg.get("transport") or ("streamable-http" if url else "stdio" if command else "")
    ).strip()

    if transport in {"http", "https", "streamable-http"}:
        if not url.startswith(("http://", "https://")):
            raise ValueError("HTTP MCP server url must start with http:// or https://")
        headers = cfg.get("headers") or {}
        if not isinstance(headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in headers.items()
        ):
            raise ValueError("HTTP MCP headers must be a string-to-string object")
        return {"transport": "streamable-http", "url": url, "headers": headers}

    if transport == "stdio":
        if not command:
            raise ValueError("stdio MCP server requires command")
        expanded = os.path.expanduser(command)
        if not os.path.isabs(expanded):
            raise ValueError("stdio MCP command must be an absolute executable path")
        command_path = os.path.realpath(expanded)
        if not os.path.isfile(command_path) or not os.access(command_path, os.X_OK):
            raise ValueError("stdio MCP command does not exist or is not executable")

        args = cfg.get("args") or []
        if not isinstance(args, list) or not all(
            isinstance(item, str) and "\x00" not in item for item in args
        ):
            raise ValueError("stdio MCP args must be a JSON array of strings")

        cwd_value = str(cfg.get("cwd") or WORKSPACE)
        cwd = os.path.realpath(os.path.expanduser(cwd_value))
        if not os.path.isdir(cwd):
            raise ValueError("stdio MCP cwd does not exist or is not a directory")
        if not _inside(cwd, WORKSPACE):
            raise ValueError("stdio MCP cwd must be inside the approved workspace")
        return {"transport": "stdio", "command": command_path, "args": args, "cwd": cwd}

    raise ValueError("MCP transport must be streamable-http or stdio")


def _run_async(factory: Callable[[], Coroutine[object, object, _T]]) -> _T:
    """Run one MCP coroutine from sync code, including when a loop already exists."""

    async def _bounded() -> _T:
        return await asyncio.wait_for(factory(), timeout=MCP_TIMEOUT)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_bounded())

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="endeavor-mcp") as pool:
        future = pool.submit(lambda: asyncio.run(_bounded()))
        try:
            return future.result(timeout=MCP_TIMEOUT + 5)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"MCP operation exceeded {MCP_TIMEOUT}s") from exc


@asynccontextmanager
async def _open_session(cfg: dict):
    from mcp import ClientSession

    cfg = _normalise_server_config(cfg)
    if cfg["transport"] == "streamable-http":
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(cfg["url"], headers=cfg["headers"]) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
        return

    from mcp.client.stdio import StdioServerParameters, stdio_client

    profile_path: str | None = None
    try:
        profile = _build_sandbox_profile(WORKSPACE)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", delete=False) as handle:
            handle.write(profile)
            profile_path = handle.name
        params = StdioServerParameters(
            command=_SANDBOX_EXEC,
            args=["-f", profile_path, cfg["command"], *cfg["args"]],
            cwd=cfg["cwd"],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    finally:
        if profile_path:
            try:
                os.unlink(profile_path)
            except OSError:
                pass


async def _list_tools_async(cfg: dict, *, tool_name: str = "") -> str:
    async with _open_session(cfg) as session:
        result = await session.list_tools()
    if tool_name:
        for item in result.tools:
            if item.name == tool_name:
                return json.dumps(
                    {
                        "name": item.name,
                        "description": item.description or "",
                        "inputSchema": item.inputSchema,
                    },
                    ensure_ascii=False,
                )
        return f"[error] MCP tool '{tool_name}' not found"
    return _compact_tool_list(result.tools)


async def _call_tool_async(cfg: dict, tool_name: str, arguments: dict) -> str:
    async with _open_session(cfg) as session:
        result = await session.call_tool(tool_name, arguments)
    parts = [item.text for item in result.content if getattr(item, "type", None) == "text"]
    body = "\n".join(parts) if parts else "(tool returned no text content)"
    return f"[error] {tool_name} returned an error: {body}" if result.isError else body


@tool
def mcp_list_tools(server: str, tool_name: str = "") -> str:
    """List tools exposed by a configured MCP server, or inspect one exact schema.

    Pass only ``server`` for a compact name/description catalog. Pass
    ``tool_name`` as well to retrieve that tool's description and input schema.
    Servers may come from ``config.MCP_SERVERS`` or the self-service registry
    created with ``mcp_add_server``.
    """
    cfg = _all_servers().get(server)
    if cfg is None:
        return f"[error] no MCP server named '{server}' configured — known servers: {_known_servers()}"
    try:
        return _run_async(lambda: _list_tools_async(cfg, tool_name=tool_name))
    except Exception as exc:
        return f"[error] mcp_list_tools failed for '{server}': {exc}"


@tool
def mcp_call_tool(server: str, tool_name: str, arguments_json: str = "{}") -> str:
    """Call a tool exposed by a configured HTTP or local stdio MCP server.

    Inspect the server/tool first with ``mcp_list_tools`` so the exact tool name
    and input schema are known. ``arguments_json`` must encode one JSON object.
    The remote/local MCP server still owns its own authorization and safety
    policy; this client only transports the request and bounds returned text.
    """
    cfg = _all_servers().get(server)
    if cfg is None:
        return f"[error] no MCP server named '{server}' configured — known servers: {_known_servers()}"
    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return f"[error] invalid arguments_json: {exc}"
    if not isinstance(arguments, dict):
        return "[error] arguments_json must be a JSON object, e.g. '{\"key\": \"value\"}'"
    try:
        return _cap(_run_async(lambda: _call_tool_async(cfg, tool_name, arguments)))
    except Exception as exc:
        return f"[error] mcp_call_tool failed for '{server}.{tool_name}': {exc}"


@tool
def mcp_add_server(
    name: str,
    url: str = "",
    headers_json: str = "{}",
    command: str = "",
    args_json: str = "[]",
    cwd: str = "",
) -> str:
    """Register a Streamable HTTP or guarded local stdio MCP server.

    HTTP: provide ``url`` and optional ``headers_json``. Stdio: provide an
    absolute executable ``command``, optional JSON-array ``args_json``, and an
    optional ``cwd`` inside this agent's approved workspace. Exactly one of URL
    or command is required. Registrations persist in
    ``workspace/tool_mcp/servers.json`` with private file permissions.
    """
    name = (name or "").strip()
    if not name:
        return "[error] name is required"
    url = (url or "").strip()
    command = (command or "").strip()
    if bool(url) == bool(command):
        return "[error] provide exactly one transport: url for HTTP or command for stdio"

    try:
        headers = json.loads(headers_json or "{}")
    except json.JSONDecodeError as exc:
        return f"[error] invalid headers_json: {exc}"
    if not isinstance(headers, dict):
        return "[error] headers_json must be a JSON object"

    try:
        args = json.loads(args_json or "[]")
    except json.JSONDecodeError as exc:
        return f"[error] invalid args_json: {exc}"
    if not isinstance(args, list):
        return "[error] args_json must be a JSON array of strings"

    if url:
        if command or cwd or args:
            return "[error] command/args_json/cwd are only valid for stdio MCP servers"
        raw_cfg = {"transport": "streamable-http", "url": url, "headers": headers}
    else:
        if headers:
            return "[error] headers_json is only valid for HTTP MCP servers"
        raw_cfg = {
            "transport": "stdio",
            "command": command,
            "args": args,
            "cwd": cwd or WORKSPACE,
        }

    try:
        cfg = _normalise_server_config(raw_cfg)
        with _dynamic_registry_lock():
            servers = _load_dynamic_servers(strict=True)
            servers[name] = cfg
            _save_dynamic_servers(servers)
    except Exception as exc:
        return f"[error] mcp_add_server failed to save '{name}': {exc}"

    target = cfg["url"] if cfg["transport"] == "streamable-http" else f"stdio:{cfg['command']}"
    return f"registered MCP server '{name}' -> {target}. Try mcp_list_tools(server='{name}') to confirm it connects."


@tool
def mcp_remove_server(name: str) -> str:
    """Remove a server previously registered with ``mcp_add_server``."""
    name = (name or "").strip()
    if not name:
        return "[error] name is required"
    try:
        with _dynamic_registry_lock():
            servers = _load_dynamic_servers(strict=True)
            if name not in servers:
                if name in MCP_SERVERS:
                    return (
                        f"[error] '{name}' is provisioned in config.MCP_SERVERS — "
                        "only self-registered servers can be removed here"
                    )
                known = ", ".join(sorted(servers)) or "(none)"
                return f"[error] no self-registered MCP server named '{name}' — known: {known}"
            del servers[name]
            _save_dynamic_servers(servers)
    except Exception as exc:
        return f"[error] mcp_remove_server failed for '{name}': {exc}"
    return f"removed MCP server '{name}' from the workspace registry."
