"""CPU-only regressions: web_search / browse_url / recall_web are LLM-free and
run several queries or URLs concurrently inside one tool call."""
from __future__ import annotations

import importlib
import sys
import threading
import time
import unittest
import uuid
from unittest.mock import patch

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _no_llm(*_args, **_kwargs):
    raise AssertionError("web tools must not call an LLM summarizer")


class WebSearchBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = importlib.import_module("tools.web_search")
        self.web_cache = importlib.import_module("tools.web_cache")
        self.summarizer = importlib.import_module("tools._summarize")
        self.web_cache.web_count_reset()

    def _search(self, query, fake):
        with patch.object(self.ws, "_web_search_one_with_sources", side_effect=fake), \
             patch.object(self.summarizer, "summarize", side_effect=_no_llm), \
             patch.object(self.summarizer, "summarize_batch", side_effect=_no_llm):
            return self.ws.web_search.func(query)

    def test_module_has_no_llm_summarizer_import(self) -> None:
        self.assertFalse(hasattr(self.ws, "summarize"))
        self.assertFalse(hasattr(self.ws, "summarize_batch"))

    def test_single_query_returns_evidence_and_source_artifact(self) -> None:
        def fake(query, _uq, _recency, _slots):
            return f"[web:https://a.test/x] evidence for {query}", ["https://a.test/x"]

        content, artifact = self._search("alpha", fake)
        self.assertIn("evidence for alpha", content)
        self.assertEqual("web_search_results_v1", artifact["provenance_type"])
        self.assertEqual(["https://a.test/x"], artifact["external_source_urls"])

    def test_query_list_runs_concurrently_in_one_call_and_keeps_order(self) -> None:
        active = peak = 0
        lock = threading.Lock()

        def fake(query, _uq, _recency, _slots):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.3)
            with lock:
                active -= 1
            return f"[web:https://{query}.test/] evidence {query}", [f"https://{query}.test/"]

        started = time.monotonic()
        content, artifact = self._search(["one", "two", "three"], fake)
        elapsed = time.monotonic() - started

        self.assertGreaterEqual(peak, 2, "queries must overlap")
        self.assertLess(elapsed, 0.8, "3 x 0.3s sequential would take 0.9s")
        positions = [content.index(f"[query:{q}]") for q in ("one", "two", "three")]
        self.assertEqual(sorted(positions), positions)
        self.assertEqual(
            {"https://one.test/", "https://two.test/", "https://three.test/"},
            set(artifact["external_source_urls"]),
        )

    def test_batch_costs_one_web_budget_call(self) -> None:
        def fake(query, _uq, _recency, _slots):
            return f"evidence {query}", []

        self.web_cache.web_count_reset()
        before = self.web_cache.web_count_remaining()
        self._search(["a", "b", "c"], fake)
        self.assertEqual(before - 1, self.web_cache.web_count_remaining())

    def test_more_than_four_queries_are_capped_with_a_note(self) -> None:
        seen: list[str] = []

        def fake(query, _uq, _recency, _slots):
            seen.append(query)
            return f"evidence {query}", []

        content, _artifact = self._search(["q1", "q2", "q3", "q4", "q5"], fake)
        self.assertEqual({"q1", "q2", "q3", "q4"}, set(seen))
        self.assertIn("[query_limit]", content)

    def test_one_failing_query_does_not_hide_the_others(self) -> None:
        def fake(query, _uq, _recency, _slots):
            if query == "bad":
                raise RuntimeError("boom")
            return f"evidence {query}", []

        content, _artifact = self._search(["good", "bad"], fake)
        self.assertIn("evidence good", content)
        self.assertIn("[error] web search unavailable: boom", content)

    def test_json_list_string_is_accepted_like_a_list(self) -> None:
        queries, limited = self.ws._normalise_query_batch('["a b", "c d"]')
        self.assertEqual(["a b", "c d"], queries)
        self.assertFalse(limited)


class BrowseBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = importlib.import_module("tools._batch_browse")
        self.web_cache = importlib.import_module("tools.web_cache")
        self.summarizer = importlib.import_module("tools._summarize")
        self.web_cache.web_count_reset()
        self.urls = [f"https://unit.invalid/{uuid.uuid4().hex}" for _ in range(4)]

    def tearDown(self) -> None:
        with self.web_cache._LOCK:
            for url in self.urls:
                self.web_cache._purge_url_locked(url)

    def _browse(self, urls, fake_fetch, query="topic"):
        with patch.object(self.batch, "_http_only", side_effect=fake_fetch), \
             patch.object(self.summarizer, "summarize", side_effect=_no_llm), \
             patch.object(self.summarizer, "summarize_batch", side_effect=_no_llm):
            return self.batch.browse_urls(urls, query)

    @staticmethod
    def _body(url: str) -> str:
        return f"Article for {url}\n\n" + ("The topic has several verifiable facts. " * 12)

    def test_urls_fetch_in_parallel_without_an_llm_and_keep_request_order(self) -> None:
        active = peak = 0
        lock = threading.Lock()

        def fake(url, _query):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.25)
            with lock:
                active -= 1
            return url, self._body(url)

        result = self._browse(self.urls[:3], fake)
        self.assertGreaterEqual(peak, 2, "fetches must overlap")
        self.assertLessEqual(peak, self.batch._MAX_WORKERS)
        order = [result.index(f"[web:{u}]") for u in self.urls[:3]]
        self.assertEqual(sorted(order), order)
        self.assertIn("(3 fetched, 0 cached)", result)

    def test_fetched_pages_are_cached_raw_so_a_second_call_is_free(self) -> None:
        def fake(url, _query):
            return url, self._body(url)

        self._browse(self.urls[:2], fake)
        remaining = self.web_cache.web_count_remaining()
        with patch.object(self.batch, "_http_only", side_effect=AssertionError("cache hit fetched")):
            again = self.batch.browse_urls(self.urls[:2], "topic")
        self.assertEqual(remaining, self.web_cache.web_count_remaining())
        self.assertIn("(0 fetched, 2 cached)", again)

    def test_budget_is_reserved_up_front_and_extra_urls_are_refused(self) -> None:
        fetched: list[str] = []

        def fake(url, _query):
            fetched.append(url)
            return url, self._body(url)

        with patch.object(self.batch, "_wc_reserve_up_to", return_value=1):
            result = self._browse(self.urls[:3], fake)
        self.assertEqual([self.urls[0]], fetched)
        self.assertIn(f"[web:{self.urls[0]}]", result)
        self.assertEqual(2, result.count("[web_limit]"))

    def test_a_failed_fetch_is_isolated(self) -> None:
        def fake(url, _query):
            if url == self.urls[1]:
                return url, "[error] fetch failed"
            return url, self._body(url)

        result = self._browse(self.urls[:3], fake)
        self.assertIn(f"[web:{self.urls[0]}]", result)
        self.assertIn("[error] fetch failed", result)
        self.assertIn(f"[web:{self.urls[2]}]", result)


class BrowseSingleReadTests(unittest.TestCase):
    def test_single_read_selects_query_relevant_evidence_without_an_llm(self) -> None:
        browse = importlib.import_module("tools.browse_url")
        web_cache = importlib.import_module("tools.web_cache")
        summarizer = importlib.import_module("tools._summarize")
        url = f"https://unit.invalid/{uuid.uuid4().hex}"
        raw = "Intro paragraph about nothing in particular. " * 20 + "\n\nThe launch date is 12 March 2027.\n\n" + "Footer text. " * 20
        web_cache.web_count_reset()
        try:
            with patch.object(browse, "_fetch_body", return_value=raw), \
                 patch.object(summarizer, "summarize", side_effect=_no_llm), \
                 patch.object(summarizer, "summarize_batch", side_effect=_no_llm):
                result = browse.browse_url.invoke({"url": url, "user_query": "launch date"})
            self.assertIn(f"[web:{url}]", result)
            self.assertIn("12 March 2027", result)
            self.assertIsNone(web_cache.get_summary(url, "launch date"))
        finally:
            with web_cache._LOCK:
                web_cache._purge_url_locked(url)


if __name__ == "__main__":
    unittest.main()
