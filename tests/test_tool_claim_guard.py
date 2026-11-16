"""Code-enforced tool use: the chart directive and the tool-claim follow-up (LLM-free: the agent is a stub)."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("AGENT_SERVER_TOKEN", "test-tool-claim-token")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
import tool_claim_guard as guard

BOUND = {"plot", "read_image", "web_search", "bash"}


def ai(text: str = "", *calls: str, finish: str = "stop") -> AIMessage:
    return AIMessage(content=text, tool_calls=[{"name": name, "args": {}, "id": f"id-{name}-{i}"} for i, name in enumerate(calls)],
                     response_metadata={"finish_reason": finish})


class RuleSelection(unittest.TestCase):
    def tools(self, request: str, bound=BOUND, **kw) -> list[str]:
        return [r["tool"] for r in guard.rules_for(request, bound, **kw)]

    def test_a_chart_request_selects_plot_and_a_negated_one_does_not(self) -> None:
        self.assertEqual(self.tools("วาดกราฟยอดขายให้หน่อย"), ["plot"])
        self.assertEqual(self.tools("please create chart of sales"), ["plot"])      # the regex is the measured one: "create a chart" is not matched
        self.assertEqual(self.tools("ไม่ต้องวาดกราฟนะ แค่อธิบายตัวเลข"), [])
        self.assertEqual(self.tools("don't plot anything, just explain"), [])
        self.assertEqual(self.tools("ราคาหุ้นวันนี้เป็นเท่าไร"), [])

    def test_a_rule_applies_only_when_its_tool_is_bound_in_this_run(self) -> None:
        self.assertEqual(self.tools("วาดกราฟให้หน่อย", bound={"web_search"}), [])

    def test_find_an_image_on_the_web_and_describe_it_selects_read_image_unless_the_user_attached_one(self) -> None:
        request = "ค้นหารูปโลโก้ apple บนเว็บแล้วอธิบายให้หน่อย"
        self.assertEqual(self.tools(request), ["read_image"])
        self.assertEqual(self.tools(request, has_attachments=True), [])
        self.assertEqual(self.tools("ค้นหาข้อมูล apple แล้วอธิบายภาพรวมธุรกิจ"), [])      # ภาพรวม is not a picture

    def test_a_chart_picture_to_look_at_does_not_push_the_agent_toward_plot(self) -> None:
        self.assertEqual(self.tools("search the web for a chart image of bitcoin and describe what the image looks like"), ["read_image"])


class ClaimDetection(unittest.TestCase):
    rules = guard.rules_for("วาดกราฟให้หน่อย", BOUND)

    def test_a_chart_claim_without_a_plot_call_is_unbacked(self) -> None:
        for text in ("สร้างกราฟเรียบร้อยแล้วครับ", "Here is the chart, saved as sales.png", "กราฟด้านล่างแสดงแนวโน้ม"):
            self.assertIsNotNone(guard.unbacked_claim(self.rules, [ai("", "web_search")], text), text)

    def test_a_plot_call_backs_the_claim_and_a_mere_mention_is_not_a_claim(self) -> None:
        self.assertIsNone(guard.unbacked_claim(self.rules, [ai("", "plot")], "สร้างกราฟเรียบร้อยแล้ว chart.png"))
        self.assertIsNone(guard.unbacked_claim(self.rules, [ai("", "web_search")], "ผมต้องการข้อมูลเพิ่มเติมก่อนจะทำกราฟ ขอตัวเลขได้ไหม"))

    def test_a_chart_drawn_with_the_sandboxed_runtimes_is_backed_so_a_true_answer_is_not_withdrawn(self) -> None:
        for tool in ("python_exec", "bash", "bash_bg"):
            self.assertIsNone(guard.unbacked_claim(self.rules, [ai("", tool)], "สร้างกราฟเรียบร้อยแล้ว บันทึกเป็น sales.png"), tool)

    def test_reading_an_image_or_file_means_the_chart_is_an_input_not_a_claim(self) -> None:
        self.assertIsNone(guard.unbacked_claim(self.rules, [ai("", "read_image")], "กราฟนี้แสดงแนวโน้มขาขึ้นแล้ว"))

    def test_an_empty_final_counts_only_after_other_tools_ran(self) -> None:
        self.assertIsNotNone(guard.unbacked_claim(self.rules, [ai("", "web_search")], "", empty_is_claim=True))
        self.assertIsNone(guard.unbacked_claim(self.rules, [], "", empty_is_claim=False))

    def test_any_description_of_a_web_image_nobody_opened_is_unbacked(self) -> None:
        rules = guard.rules_for("ค้นหารูปโลโก้ apple บนเว็บแล้วอธิบายให้หน่อย", BOUND)
        self.assertIsNotNone(guard.unbacked_claim(rules, [ai("", "web_search")], "โลโก้เป็นแอปเปิลสีเงินมีรอยกัด"))
        self.assertIsNone(guard.unbacked_claim(rules, [ai("", "web_search", "read_image")], "โลโก้เป็นแอปเปิลสีเงินมีรอยกัด"))

    def test_messages_for_the_agent_and_the_user_name_the_tool(self) -> None:
        rule = self.rules[0]
        self.assertIn("`plot`", guard.continue_message(rule))
        self.assertIn("no chart was created", guard.unbacked_note(rule))
        self.assertIn("Pass read_image an image URL", guard.continue_message(guard.rules_for("หารูปโลโก้ apple บนเว็บแล้วอธิบาย", BOUND)[0]))


class Directive(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(graph, "_BOUND_TOOL_NAMES", set(BOUND))
        patcher.start()
        self.addCleanup(patcher.stop)

    def patched(self, text: str):
        msgs, seeded = graph._force_plan_or_directive([HumanMessage(content=text)])
        return msgs[-1].content, seeded

    def test_a_chart_request_gets_the_directive_in_its_own_turn_and_nothing_is_persisted(self) -> None:
        content, seeded = self.patched("วาดกราฟยอดขายให้หน่อย")
        self.assertTrue(content.startswith("[CHART DIRECTIVE"))
        self.assertTrue(content.endswith("วาดกราฟยอดขายให้หน่อย"))
        self.assertEqual(seeded, [])

    def test_the_chart_rung_comes_before_the_search_rung(self) -> None:
        content, seeded = self.patched("ค้นหาราคา BTC แล้ววาดกราฟ")
        self.assertTrue(content.startswith("[CHART DIRECTIVE"))
        self.assertEqual(seeded, [])

    def test_no_directive_without_the_plot_tool_for_a_plain_question_or_a_second_time(self) -> None:
        with mock.patch.object(graph, "_BOUND_TOOL_NAMES", {"web_search"}):
            self.assertEqual(self.patched("วาดกราฟให้หน่อย")[0], "วาดกราฟให้หน่อย")
        self.assertEqual(self.patched("วันนี้อากาศเป็นอย่างไร")[0], "วันนี้อากาศเป็นอย่างไร")
        once = self.patched("วาดกราฟให้หน่อย")[0]
        self.assertEqual(graph._force_plan_or_directive([HumanMessage(content=once)])[0][-1].content, once)


class FollowUp(unittest.TestCase):
    """graph._enforce_tool_claims against a stub agent."""

    def setUp(self) -> None:
        patcher = mock.patch.object(graph, "_BOUND_TOOL_NAMES", set(BOUND))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.trimmed = [HumanMessage(content="วาดกราฟยอดขายให้หน่อย")]
        self.config = {"callbacks": []}

    def stub_agent(self, replies: list[list]):
        calls: list[list] = []

        class Agent:
            def invoke(inner, payload, config=None):
                calls.append(list(payload["messages"]))
                return {"messages": list(payload["messages"]) + replies.pop(0)}
        patcher = mock.patch.object(graph, "_REACT", Agent())
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def first_pass(self, new_msgs: list) -> dict:
        return {"messages": self.trimmed + new_msgs}

    def run_guard(self, new_msgs: list, query: str = "วาดกราฟยอดขายให้หน่อย", config=None):
        return graph._enforce_tool_claims(self.first_pass(new_msgs), self.trimmed, [], new_msgs, query, config or self.config)

    def test_a_false_chart_claim_is_withdrawn_and_the_agent_resumed_once(self) -> None:
        draft = ai("สร้างกราฟเรียบร้อยแล้ว sales.png")
        calls = self.stub_agent([[ai("", "plot"), ToolMessage(content="saved /w/sales.png", tool_call_id="id-plot-0", name="plot"), ai("กราฟอยู่ที่ /w/sales.png")]])
        out, new_msgs, still = self.run_guard([draft])
        self.assertIsNone(still)
        self.assertEqual(len(calls), 1)
        self.assertNotIn(draft, calls[0])                                   # the false draft is not shown to the agent again
        self.assertIn("`plot`", calls[0][-1].content)                       # the resume names the tool
        self.assertTrue(guard.has_named_tool_call(new_msgs, {"plot"}))
        self.assertEqual(new_msgs[-1].content, "กราฟอยู่ที่ /w/sales.png")
        self.assertFalse(any(isinstance(m, HumanMessage) and "TOOL NOT CALLED" in str(m.content) for m in new_msgs))   # not kept in history

    def test_the_front_end_is_told_to_drop_the_streamed_draft(self) -> None:
        class Handler:
            withdrawn = 0

            def on_superseded_answer(self):
                Handler.withdrawn += 1
        self.stub_agent([[ai("", "plot")]])
        self.run_guard([ai("สร้างกราฟเรียบร้อยแล้ว")], config={"callbacks": [Handler()]})
        self.assertEqual(Handler.withdrawn, 1)

    def test_a_second_skip_is_reported_so_the_caller_can_say_so(self) -> None:
        calls = self.stub_agent([[ai("กราฟเสร็จแล้วครับ ดูไฟล์ chart.png")]])
        _, _, still = self.run_guard([ai("สร้างกราฟเรียบร้อยแล้ว")])
        self.assertEqual(still["tool"], "plot")
        self.assertEqual(len(calls), 1)                                     # resumed once, never twice

    def test_nothing_happens_when_the_tool_ran_when_no_rule_applies_or_the_answer_is_cut_off(self) -> None:
        calls = self.stub_agent([])
        backed = [ai("", "plot"), ToolMessage(content="saved", tool_call_id="id-plot-0", name="plot"), ai("เสร็จแล้ว chart.png")]
        self.assertEqual(self.run_guard(backed)[2], None)
        self.assertEqual(self.run_guard([ai("สร้างกราฟเรียบร้อยแล้ว")], query="สวัสดีครับ")[2], None)
        self.assertEqual(self.run_guard([ai("สร้างกราฟเรียบร้อยแล้ว", finish="length")])[2], None)
        self.assertEqual(calls, [])

    def test_an_empty_final_after_other_tools_resumes_but_an_empty_turn_with_no_tools_does_not(self) -> None:
        calls = self.stub_agent([[ai("", "plot")]])
        self.run_guard([ai("", "web_search"), ToolMessage(content="data", tool_call_id="id-web_search-0", name="web_search"), ai("")])
        self.assertEqual(len(calls), 1)
        calls2 = self.stub_agent([])
        self.run_guard([ai("")])
        self.assertEqual(calls2, [])


class StreamedDraft(unittest.TestCase):
    def test_the_websocket_callback_emits_discard_stream(self) -> None:
        import asyncio
        import agent_server as srv
        events: list[dict] = []

        class Loop:
            def call_soon_threadsafe(self, fn, *args):
                fn(*args)

        class Queue:
            def put_nowait(self, event):
                events.append(event)
        handler = srv._WSCallback(Queue(), Loop(), mock.Mock(), "turn-1")
        handler.on_superseded_answer()
        self.assertEqual(events, [{"type": "discard_stream"}])


if __name__ == "__main__":
    unittest.main()
