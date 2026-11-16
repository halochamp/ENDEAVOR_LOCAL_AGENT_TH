# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Code-enforced tool use for one turn: make the model call the tool a request needs, and follow up in code when it did not.

Prompt rules alone are not enough on a local model (measured on the owner's other agent: a chart request ended with no ``plot``
call and a "chart created" answer in about 1 of 3 runs).  Two small, deterministic pieces, both table-driven:

* **Directive** (before the model runs): a request that asks for a chart gets ``CHART_DIRECTIVE`` prepended to the user turn
  (an ephemeral patch, never persisted), so calling ``plot`` is stated in the turn itself.
* **Claim guard** (after the model ran): each row of ``TOOL_CLAIM_RULES`` maps what the *user's own request* asks for to the
  tool that does it.  When that tool never ran this turn and the final answer claims the result, graph.py withdraws the draft and
  resumes the agent once (``CONTINUE_PROMPT``); if it still did not run, ``UNBACKED_NOTE`` is appended so the answer cannot
  pass for done.  ``intent`` reads only the request (negated verbs dropped); ``claim`` is deliberately tight, so a mere mention
  of a chart never forces a call.  Rows apply only when their tool is bound in the running graph.

Everything here is a pure function of text and messages: no model, no network.
"""
from __future__ import annotations

import re

NEGATION_RE = re.compile(
    r"(?:ไม่ต้อง|อย่า|ห้าม|ไม่เอา|ไม่ขอ)\s*(?:ให้\s*)?(?:ทำการ\s*)?(?:วาด|สร้าง|ทำ|ส่ง|พล็อต)\S*"
    r"|\b(?:do\s+not|don['’]t|never|without|no\s+need\s+to)\s+(?:plot|draw|chart|graph|create|make|send)\b\S*",
    re.IGNORECASE,
)

# ── chart ────────────────────────────────────────────────────────────────────
CHART_RE = re.compile(
    r"(วาด|สร้าง|ทำ|draw|create|generate)\s*(กราฟ|chart|graph)"
    r"|plot\s+(?:graph|chart|กราฟ|ราคา|price|data|sales|trend|value)"
    r"|กราฟ.{0,10}(ให้|หน่อย|เลย|ด้วย)",  # "กราฟให้หน่อย", "กราฟเลย"
    re.IGNORECASE,
)

CHART_DIRECTIVE = (
    "[CHART DIRECTIVE — deterministic intercept]\n"
    "ผู้ใช้ขอ chart/กราฟใน turn นี้ → ต้องเรียก plot tool — ห้ามตอบเป็น text แทนกราฟ "
    "(ตัวเลขอยู่ใน context ≠ user มีกราฟแล้ว — user ต้องการ visual ไม่ใช่ prose)\n"
    "ข้อมูลตัวเลข: มีใน context → ใช้ค่านั้นใน plot code เลย / "
    "user ขอตัวเลข real-time ชัดเจน (\"ราคาล่าสุด\", \"ข้อมูลจริงวันนี้\") และไม่มีใน context → web_search ก่อน แล้วค่อย plot / "
    "นอกนั้น → ใช้ค่า placeholder ที่สมเหตุสมผลใน plot code ทันที ห้าม web_search โดย default "
    "(เช่น \"วาดกราฟยอดขาย BTC ETH SOL\" — ยอดขายเป็นแค่ label ไม่ใช่คำขอข้อมูลจริง)\n"
    "เรียก plot ด้วย Python code (pandas/matplotlib) → ได้ .png → จบ"
)

_CLAIM_CHART_WORD = r"(?:กราฟ|แผนภูมิ|แผนภาพ|ฮิสโตแกรม|พล็อต|\b(?:histogram|chart|graph|plot)\b)"
_CLAIM_DONE_WORD = (
    r"(?:แล้ว|เรียบร้อย|สำเร็จ|เสร็จ|ด้านบน|ด้านล่าง|ข้างต้น|ข้างบน|"
    r"\b(?:ready|done|created|generated|saved|drawn|plotted|below|above|attached|opened)\b)"
)
PLOT_CLAIM_RE = re.compile(
    r"\.(?:png|jpe?g|svg)\b"  # names a chart file
    r"|^(?=.*" + _CLAIM_CHART_WORD + r")(?=.*" + _CLAIM_DONE_WORD + r").*$",  # chart + "done" on one line
    re.IGNORECASE | re.MULTILINE,
)

# ── "find an image on the web and describe it" ───────────────────────────────
# All three parts are required so a plain "ค้นหาข้อมูล… อธิบายภาพรวม" is not an image request (ภาพรวม/รูปแบบ excluded).
WEB_IMAGE_DESCRIBE_RE = re.compile(
    r"^(?=.*(?:ค้นหา|ค้น|หา(?:รูป|ภาพ)|บนเว็บ|ในเว็บ|ออนไลน์|อินเทอร์เน็ต|"
    r"\b(?:search|google|find|look\s+up|online|on\s+the\s+web)\b))"
    r"(?=.*(?:รูป(?!แบบ)|ภาพ(?!รวม|ยนตร์)|โลโก้|ไอคอน|\b(?:image|picture|photo|logo|icon)s?\b))"
    r"(?=.*(?:อธิบาย|บรรยาย|มีอะไร|หน้าตา|ดูรูป|ดูภาพ|"
    r"\b(?:describe|what(?:'s|\s+is)?\s+in|looks?\s+like)\b))",
    re.IGNORECASE | re.DOTALL,
)
ANY_TEXT_RE = re.compile(r"\S")  # any non-empty answer to that request describes an image nobody looked at

# `unless_tools`: a turn that ran one of these is talking about an INPUT artifact (explaining a chart image also matches the chart
# intent), not claiming output.  `skip_with_attachments`: the user attached the picture, so the model already has it.
TOOL_CLAIM_RULES = (
    # python_exec / bash / bash_bg can draw a chart with matplotlib and save a .png themselves: a turn that ran one of them has done
    # real work toward the chart, so a true "saved chart.png" answer must not be withdrawn (found by the pre-commit audit: it would
    # have forced a second, duplicate chart).
    {"tool": "plot", "intent": CHART_RE, "claim": PLOT_CLAIM_RE, "outcome": "no chart was created",
     "unless_tools": frozenset({"read_image", "read_file", "python_exec", "bash", "bash_bg"})},
    {"tool": "read_image", "intent": WEB_IMAGE_DESCRIBE_RE, "claim": ANY_TEXT_RE,
     "outcome": "no image was looked at", "skip_with_attachments": True,
     "unless_tools": frozenset({"read_file", "computer"}),
     "hint": "Pass read_image an image URL (.png/.jpg/.svg/.webp) taken from pages already fetched this turn; "
             "if none is available, run one image search. If no image can be opened, say so plainly "
             "instead of describing it."},
)

CONTINUE_PROMPT = (
    "[TOOL NOT CALLED]\n"
    "This request needs `{tool}` but no `{tool}` call ran this turn, so {outcome} yet. "
    "Call `{tool}` now; do not describe results you have not produced. "
    "After it returns, confirm only what its result shows."
)
UNBACKED_NOTE = (
    "Note: `{tool}` never ran this turn, so {outcome}. "
    "Disregard any statement above that says otherwise; ask again to retry."
)


def has_named_tool_call(messages: list, names: set[str] | frozenset[str]) -> bool:
    """True when any message in ``messages`` carries a tool call with one of ``names``."""
    return any(call.get("name") in names for message in messages for call in (getattr(message, "tool_calls", None) or []))


def rules_for(request: str, bound_tools: set[str] | frozenset[str], *, has_attachments: bool = False) -> list[dict]:
    """Rows whose intent the user's own request states (negated verbs ignored) and whose tool is bound in this run."""
    clean = NEGATION_RE.sub(" ", str(request or ""))
    rules = [
        rule for rule in TOOL_CLAIM_RULES
        if rule["tool"] in bound_tools and rule["intent"].search(clean)
        and not (has_attachments and rule.get("skip_with_attachments"))
    ]
    # "find a chart picture on the web and explain it": the chart is an INPUT to look at, so a chart claim must not resume toward plot.
    if any(rule["tool"] == "read_image" for rule in rules):
        rules = [rule for rule in rules if rule["tool"] != "plot"]
    return rules


def unbacked_claim(rules: list[dict], evidence: list, text: str, *, empty_is_claim: bool = False) -> dict | None:
    """First rule whose tool never ran in ``evidence`` while ``text`` claims its result.

    An empty final counts as a claim only with ``empty_is_claim`` (other tools ran: the no-tool empty case has its own retry)."""
    for rule in rules:
        if has_named_tool_call(evidence, {rule["tool"], *rule["unless_tools"]}):
            continue
        if (text or "").strip():
            if rule["claim"].search(text):
                return rule
        elif empty_is_claim:
            return rule
    return None


def continue_message(rule: dict) -> str:
    """The follow-up the agent receives when its answer was withdrawn."""
    return "\n".join(filter(None, (CONTINUE_PROMPT.format(tool=rule["tool"], outcome=rule["outcome"]), rule.get("hint"))))


def unbacked_note(rule: dict) -> str:
    return UNBACKED_NOTE.format(tool=rule["tool"], outcome=rule["outcome"])


def chart_directive_applies(text: str) -> bool:
    """The user turn asks for a chart and no directive was patched in already."""
    value = str(text or "")
    return not value.startswith("[CHART DIRECTIVE") and bool(CHART_RE.search(value))
