"""CPU/network-free regression coverage for the generic MCP client.

Run from the repository with the project MLX interpreter:
    conda run -n mlx python tests/test_mcp_client.py

The stdio integration uses a tiny local fake MCP server and a test-only sandbox
wrapper; it never contacts an external MCP endpoint or model server.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import stat
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mcp = importlib.import_module("tools.mcp_client")

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail="") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}: {detail}")


with tempfile.TemporaryDirectory(prefix="th-mcp-unit-") as tmp:
    root = Path(tmp)
    workspace = root / "workspace"
    workspace.mkdir()

    old_workspace = mcp.WORKSPACE
    old_sandbox_exec = mcp._SANDBOX_EXEC
    old_max_chars = mcp.MCP_MAX_CHARS
    original_servers = dict(mcp.MCP_SERVERS)
    check("public release bundles no default MCP servers", original_servers == {}, original_servers)
    mcp.WORKSPACE = str(workspace)
    mcp.MCP_SERVERS.clear()

    try:
        print("=== concurrent private registry ===")
        barrier = threading.Barrier(6)
        results: list[str] = []
        lock = threading.Lock()

        def add_server(index: int) -> None:
            barrier.wait(timeout=5)
            result = mcp.mcp_add_server.invoke(
                {"name": f"server-{index}", "url": f"https://server-{index}.example/mcp"}
            )
            with lock:
                results.append(result)

        threads = [threading.Thread(target=add_server, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        registry = workspace / "tool_mcp" / "servers.json"
        saved = json.loads(registry.read_text(encoding="utf-8"))
        check("all concurrent add calls succeed", len(results) == 6 and not any(r.startswith("[error]") for r in results), results)
        check("all entries are durable", set(saved) == {f"server-{i}" for i in range(6)}, sorted(saved))
        check("registry is mode 0600", stat.S_IMODE(registry.stat().st_mode) == 0o600, oct(stat.S_IMODE(registry.stat().st_mode)))
        check("no registry worker remains alive", not any(thread.is_alive() for thread in threads))

        print("=== registration guards ===")
        relative = mcp.mcp_add_server.invoke({"name": "bad-relative", "command": "python3"})
        check("relative stdio executable rejected", relative.startswith("[error]") and "absolute" in relative, relative)

        outside = mcp.mcp_add_server.invoke(
            {"name": "bad-cwd", "command": sys.executable, "cwd": str(root)}
        )
        check("stdio cwd outside workspace rejected", outside.startswith("[error]") and "workspace" in outside, outside)

        mixed = mcp.mcp_add_server.invoke(
            {"name": "bad-mixed", "url": "https://example.test/mcp", "command": sys.executable}
        )
        check("mixed HTTP and stdio rejected", mixed.startswith("[error]") and "exactly one" in mixed, mixed)

        non_object = mcp.mcp_call_tool.invoke(
            {"server": "server-0", "tool_name": "noop", "arguments_json": "[]"}
        )
        check("call arguments must be JSON object", non_object.startswith("[error]") and "JSON object" in non_object, non_object)

        print("=== developer/self-service ownership ===")
        mcp.MCP_SERVERS["developer"] = {"url": "https://developer.example/mcp", "headers": {}}
        remove_developer = mcp.mcp_remove_server.invoke({"name": "developer"})
        check("developer server cannot be removed by self-service tool", remove_developer.startswith("[error]") and "provisioned" in remove_developer, remove_developer)

        override = mcp.mcp_add_server.invoke({"name": "developer", "url": "https://override.example/mcp"})
        check("workspace registration may shadow developer config", not override.startswith("[error]"), override)
        check("workspace entry wins on collision", mcp._all_servers()["developer"]["url"] == "https://override.example/mcp", mcp._all_servers()["developer"])
        removed_override = mcp.mcp_remove_server.invoke({"name": "developer"})
        check("removing shadow restores developer config", not removed_override.startswith("[error]") and mcp._all_servers()["developer"]["url"] == "https://developer.example/mcp", removed_override)

        print("=== event-loop compatibility ===")
        async def inside_running_loop():
            return mcp._run_async(lambda: asyncio.sleep(0, result="EVENT_LOOP_OK"))

        check("sync MCP runner works inside active event loop", asyncio.run(inside_running_loop()) == "EVENT_LOOP_OK")

        print("=== bounded list keeps names discoverable ===")
        class SyntheticTool:
            def __init__(self, name: str, description: str):
                self.name = name
                self.description = description

        mcp.MCP_MAX_CHARS = 120
        compact = mcp._compact_tool_list(
            [SyntheticTool(f"tool_{i}", "description " * 20) for i in range(8)]
        )
        check("all tool names survive list compaction", all(f"tool_{i}" in compact for i in range(8)), compact)
        check("compaction marker is explicit", "descriptions compacted" in compact, compact)

        print("=== local stdio integration ===")
        fake_root = workspace / "fake_stdio"
        fake_root.mkdir()
        server_py = fake_root / "fake_mcp.py"
        wrapper = fake_root / "sandbox_wrapper.sh"
        server_py.write_text(
            "from mcp.server.fastmcp import FastMCP\n"
            "server = FastMCP('fake')\n"
            "@server.tool()\n"
            "def ping(text: str = 'pong') -> str:\n"
            "    return text\n"
            "if __name__ == '__main__':\n"
            "    server.run()\n",
            encoding="utf-8",
        )
        wrapper.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"-f\" ]; then shift 2; fi\n"
            "exec \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        mcp._SANDBOX_EXEC = str(wrapper)

        registered = mcp.mcp_add_server.invoke(
            {
                "name": "stdio-fake",
                "command": sys.executable,
                "args_json": json.dumps([str(server_py)]),
                "cwd": str(fake_root),
            }
        )
        listed = mcp.mcp_list_tools.invoke({"server": "stdio-fake"})
        schema = mcp.mcp_list_tools.invoke({"server": "stdio-fake", "tool_name": "ping"})
        called = mcp.mcp_call_tool.invoke(
            {"server": "stdio-fake", "tool_name": "ping", "arguments_json": '{"text":"HELLO_STDIO"}'}
        )
        check("stdio server registration succeeds", not registered.startswith("[error]"), registered)
        check("list_tools reaches local child", "ping" in listed and not listed.startswith("[error]"), listed)
        check("targeted schema lookup returns inputSchema", '"inputSchema"' in schema and '"ping"' in schema, schema)
        check("call_tool round-trips text", called == "HELLO_STDIO", called)

        print("=== malformed registry mutation guard ===")
        broken = "{not valid json"
        registry.write_text(broken, encoding="utf-8")
        add_broken = mcp.mcp_add_server.invoke({"name": "new", "url": "https://new.example/mcp"})
        check("add refuses malformed registry", add_broken.startswith("[error]") and "invalid" in add_broken, add_broken)
        check("failed add preserves forensic content", registry.read_text(encoding="utf-8") == broken)
        remove_broken = mcp.mcp_remove_server.invoke({"name": "server-0"})
        check("remove refuses malformed registry", remove_broken.startswith("[error]") and "invalid" in remove_broken, remove_broken)
        check("failed remove preserves forensic content", registry.read_text(encoding="utf-8") == broken)
    finally:
        mcp.WORKSPACE = old_workspace
        mcp._SANDBOX_EXEC = old_sandbox_exec
        mcp.MCP_MAX_CHARS = old_max_chars
        mcp.MCP_SERVERS.clear()
        mcp.MCP_SERVERS.update(original_servers)

print(f"\nPASS {PASS}  FAIL {FAIL}")
if FAIL:
    raise SystemExit(1)
print("MCP_CLIENT_OK")
