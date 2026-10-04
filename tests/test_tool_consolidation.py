from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import tools


EXPECTED_TOOLS = (
    "web_search",
    "bash",
    "bash_bg",
    "python_exec",
    "plot",
    "read_file",
    "edit",
    "create_plan",
    "browse_url",
    "browser_use",
    "recall_web",
    "remember",
    "read_image",
    "computer",
    "awake",
    "speak",
    "mcp_list_tools",
    "mcp_call_tool",
    "mcp_add_server",
    "mcp_remove_server",
)


class ToolConsolidationTests(unittest.TestCase):
    def test_normal_registry_is_exact_and_native_rag_is_not_configured(self):
        names = tuple(tool.name for tool in tools.ALL_TOOLS)
        self.assertEqual(EXPECTED_TOOLS, names)
        self.assertEqual(20, len(tools.ALL_TOOLS))
        self.assertEqual({"research"}, set(tools.SKILL_TOOLS))
        self.assertNotIn(tools.SKILL_TOOLS["research"][0], tools.ALL_TOOLS)
        self.assertEqual({}, config.MCP_SERVERS)

        removed_modules = (
            "tools.write_file",
            "tools.grep",
            "tools.workspace_ls",
            "tools.fetch_sitemap",
            "tools.batch_browse",
            "tools.scrape_table",
            "tools.tool_loop",
            "tools.rag_tool",
        )
        for module_name in removed_modules:
            with self.subTest(module=module_name):
                self.assertIsNone(importlib.util.find_spec(module_name))

    def test_browse_url_schema_unifies_modes(self):
        browse_url = importlib.import_module("tools.browse_url").browse_url
        self.assertEqual(
            {"url", "user_query", "urls", "mode", "filter_keyword", "table_index"},
            set(browse_url.args),
        )

    def test_mcp_activity_uses_the_public_server_argument(self):
        agent_server = importlib.import_module("agent_server")
        self.assertEqual(
            "library/search",
            agent_server._tool_detail("mcp_call_tool", {"server": "library", "tool_name": "search"}),
        )

    def test_browse_url_dispatches_batch_sitemap_and_table_requests(self):
        browse_url = importlib.import_module("tools.browse_url").browse_url
        batch = importlib.import_module("tools._batch_browse")
        sitemap = importlib.import_module("tools._fetch_sitemap")
        table = importlib.import_module("tools._scrape_table")

        with patch.object(batch, "browse_urls", return_value="batch-result") as call:
            result = browse_url.invoke({"urls": ["one.test", "two.test"], "user_query": "topic"})
        self.assertEqual("batch-result", result)
        call.assert_called_once_with(["one.test", "two.test"], "topic")

        with patch.object(sitemap, "read_sitemap", return_value="sitemap-result") as call:
            result = browse_url.invoke({
                "url": "example.test",
                "mode": "sitemap",
                "filter_keyword": "fund 2026",
            })
        self.assertEqual("sitemap-result", result)
        call.assert_called_once_with("https://example.test", "fund 2026")

        with patch.object(table, "extract_table", return_value="table-result") as call:
            result = browse_url.invoke({"url": "https://example.test/data", "table_index": -1})
        self.assertEqual("table-result", result)
        call.assert_called_once_with("https://example.test/data", -1)

    def test_browse_url_accepts_model_serialized_batch_urls(self):
        browse_url = importlib.import_module("tools.browse_url").browse_url
        batch = importlib.import_module("tools._batch_browse")
        urls = [
            "https://example.com",
            "https://www.iana.org/help/example-domains",
        ]
        serialized_urls = '["https://example.com", "https://www.iana.org/help/example-domains"]'

        url_schema = browse_url.args_schema.model_json_schema()["properties"]["urls"]
        accepted_types = {part.get("type") for part in url_schema["anyOf"]}
        self.assertEqual({"array", "string", "null"}, accepted_types)

        with patch.object(batch, "browse_urls", return_value="batch-result") as call:
            result = browse_url.invoke({"urls": serialized_urls, "user_query": "compare them"})
        self.assertEqual("batch-result", result)
        call.assert_called_once_with(urls, "compare them")

        with patch.object(batch, "browse_urls") as call:
            sitemap_result = browse_url.invoke({
                "url": "https://example.com",
                "urls": serialized_urls,
                "mode": "sitemap",
            })
            table_result = browse_url.invoke({
                "urls": serialized_urls,
                "table_index": 0,
            })
            invalid_result = browse_url.invoke({"urls": "not a JSON list"})

        self.assertIn("sitemap mode accepts", sitemap_result)
        self.assertIn("table_index is only valid", table_result)
        self.assertIn("JSON-encoded list", invalid_result)
        call.assert_not_called()

    def test_browse_url_read_preserves_cache_for_recall_and_adds_table_evidence(self):
        browse_module = importlib.import_module("tools.browse_url")
        recall_web = importlib.import_module("tools.recall_web").recall_web
        web_cache = importlib.import_module("tools.web_cache")
        url = f"https://unit.invalid/{uuid.uuid4().hex}"
        raw = """# Quarterly report

| Year | Revenue |
| --- | ---: |
| 2026 | 42 |
"""
        try:
            with patch.object(browse_module, "_wc_check_and_inc", return_value=None), \
                 patch.object(browse_module, "_fetch_body", return_value=raw) as fetch, \
                 patch.object(browse_module, "summarize", return_value="Quarterly summary"):
                result = browse_module.browse_url.invoke({"url": url, "user_query": "revenue"})

            self.assertIn(f"[web:{url}] Quarterly summary", result)
            self.assertIn("[table evidence]", result)
            self.assertIn("| 2026 | 42 |", result)
            fetch.assert_called_once_with(url, "revenue")
            self.assertEqual(raw, web_cache.get(url))
            self.assertEqual(raw, recall_web.invoke({"url": url}))
        finally:
            with web_cache._LOCK:
                web_cache._purge_url_locked(url)

    def test_cached_bounded_batch_deduplicates_without_spending_web_budget(self):
        batch = importlib.import_module("tools._batch_browse")
        web_cache = importlib.import_module("tools.web_cache")
        query = f"topic-{uuid.uuid4().hex}"
        urls = [
            f"https://unit.invalid/{uuid.uuid4().hex}",
            f"https://unit.invalid/{uuid.uuid4().hex}",
        ]
        try:
            for index, url in enumerate(urls):
                raw = f"cached body {index}"
                web_cache.put(url, raw)
                web_cache.put_summary(url, query, f"summary {index}", raw=raw)

            with patch.object(batch, "_wc_check", side_effect=AssertionError("cache hit spent budget")), \
                 patch.object(batch, "_http_only", side_effect=AssertionError("cache hit fetched")):
                result = batch.browse_urls([urls[0], urls[1], urls[0]], query)

            self.assertIn("[browse_url batch] 2 URLs (0 fetched, 2 cached)", result)
            self.assertIn(f"[web:{urls[0]}] summary 0", result)
            self.assertIn(f"[web:{urls[1]}] summary 1", result)
        finally:
            with web_cache._LOCK:
                for url in urls:
                    web_cache._purge_url_locked(url)

    def test_sitemap_parser_preserves_entities_and_newest_first_order(self):
        sitemap = importlib.import_module("tools._fetch_sitemap")
        entries = sitemap._parse_entries("""<urlset>
          <url><loc>https://example.test/a?x=1&amp;y=2</loc><lastmod>2025-01-01</lastmod></url>
          <url><loc>https://example.test/b</loc><lastmod>2026-02-03</lastmod></url>
        </urlset>""")
        self.assertEqual("https://example.test/a?x=1&y=2", entries[0][0])
        self.assertEqual(
            ["https://example.test/b", "https://example.test/a?x=1&y=2"],
            [url for url, _ in sitemap._sort_by_lastmod(entries)],
        )

    def test_markdown_numeric_tables_are_bounded_evidence(self):
        browse_module = importlib.import_module("tools.browse_url")
        evidence = browse_module._markdown_table_evidence(
            "| Year | Revenue |\n| --- | --- |\n| 2026 | 42 |"
        )
        self.assertIn("[table evidence]", evidence)
        self.assertIn("| 2026 | 42 |", evidence)
        self.assertEqual("", browse_module._markdown_table_evidence("| Name | Note |\n| --- | --- |\n| A | text |"))

    def test_rendered_table_extractor_returns_csv_without_workspace_write(self):
        import pandas as pd

        table_module = importlib.import_module("tools._scrape_table")
        frame = pd.DataFrame([{"Year": "2026", "Revenue": "42"}])
        page = types.SimpleNamespace(
            goto=lambda *args, **kwargs: None,
            wait_for_load_state=lambda *args, **kwargs: None,
            content=lambda: "<html><body>fixture</body></html>",
        )
        browser = types.SimpleNamespace(new_page=lambda: page, close=lambda: None)
        playwright = types.SimpleNamespace(chromium=types.SimpleNamespace(launch=lambda **kwargs: browser))

        class PlaywrightContext:
            def __enter__(self):
                return playwright

            def __exit__(self, *_args):
                return False

        playwright_package = types.ModuleType("playwright")
        playwright_sync = types.ModuleType("playwright.sync_api")
        playwright_sync.sync_playwright = PlaywrightContext
        with patch.dict(sys.modules, {
            "playwright": playwright_package,
            "playwright.sync_api": playwright_sync,
        }), patch.object(pd, "read_html", return_value=[frame]):
            result = table_module.extract_table("https://example.test/data", 0)

        self.assertIn("[browse_url table]", result)
        self.assertIn("Year,Revenue", result)
        self.assertIn("2026,42", result)


if __name__ == "__main__":
    unittest.main()
