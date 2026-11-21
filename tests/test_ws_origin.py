"""Agent server: which Origin may open the WebSocket / call the REST routes from a browser context.

The desktop app (Electron "file://" or the Tauri app's "tauri://localhost") and the server's own host are allowed; any
other site is not, which is what stops a page on the web reaching 127.0.0.1 through DNS rebinding.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("AGENT_SERVER_TOKEN", "test-ws-origin-token")

import agent_server as srv


class WebSocketOriginTests(unittest.TestCase):
    def test_desktop_app_origins_are_allowed(self) -> None:
        for origin in ("tauri://localhost", "http://tauri.localhost", "file://", "null", None):
            self.assertTrue(srv._origin_ok(origin), origin)

    def test_own_host_is_allowed(self) -> None:
        self.assertTrue(srv._origin_ok(f"http://localhost:{srv.SERVER_PORT}"))
        self.assertTrue(srv._origin_ok(f"http://127.0.0.1:{srv.SERVER_PORT}"))

    def test_other_sites_are_refused(self) -> None:
        for origin in ("https://evil.example", "http://evil.example:8765", "tauri://evil.example",
                       "http://localhost:9999", "https://tauri.localhost", "http://localhost.evil.example"):
            self.assertFalse(srv._origin_ok(origin), origin)


if __name__ == "__main__":
    unittest.main()
