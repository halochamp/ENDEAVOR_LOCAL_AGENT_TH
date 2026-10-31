"""The request body reaching mlx_vlm.server must carry the generation limit.

langchain-openai renames ``max_tokens`` to ``max_completion_tokens`` before sending a
chat-completions request, but mlx_vlm.server (0.7.x) only reads the top-level
``max_tokens`` field, so without an explicit copy every request silently fell back to
the server default of 2048 tokens (long answers and long thinking were cut mid-way).
"""
from __future__ import annotations

import json
import unittest

import httpx

import config
from llm import build_llm


def _client_capturing(bodies: list[dict]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
    return httpx.Client(transport=httpx.MockTransport(handler))


class RequestBodyTests(unittest.TestCase):
    def _send(self, **overrides) -> dict:
        bodies: list[dict] = []
        llm = build_llm(streaming=False, http_client=_client_capturing(bodies), **overrides)
        llm.invoke("hi")
        self.assertEqual(len(bodies), 1)
        return bodies[0]

    def test_default_limit_reaches_the_server_as_max_tokens(self) -> None:
        body = self._send()
        self.assertEqual(body["max_tokens"], config.MAX_TOKENS)

    def test_an_explicit_limit_overrides_the_default(self) -> None:
        body = self._send(max_tokens=777)
        self.assertEqual(body["max_tokens"], 777)

    def test_other_extra_body_fields_are_unchanged(self) -> None:
        body = self._send()
        self.assertIn("thinking_budget", body)
        self.assertIn("enable_thinking", body)

    def test_no_thinking_helpers_keep_the_limit_and_drop_the_budget(self) -> None:
        body = self._send(extra_body={"enable_thinking": False}, max_tokens=300)
        self.assertEqual(body["max_tokens"], 300)
        self.assertNotIn("thinking_budget", body)


if __name__ == "__main__":
    unittest.main()
