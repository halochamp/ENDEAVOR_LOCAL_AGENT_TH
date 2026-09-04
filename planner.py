"""planner.py — is_trivial (diagnostic-only) + classify simple/complex + plan

- is_trivial: greeting/thanks/recall detector — NOT wired into any production path.
  Used only by the unit test suite as a sanity-check for the trivial/recall regexes.
- planner: 1 LLM call → {"mode":"simple"} | {"mode":"complex","plan":[str,...]}
- STOP-chain (knowhow C2), plan = list[str] เท่านั้น (locked contract)
- JSON retry 1 รอบ (robustness #2); fallback → simple (safe: simple→react)
"""
from __future__ import annotations
import re
import json
import logging
import datetime
from llm import build_llm

log = logging.getLogger(__name__)

# diagnostic-only detector — conservative (false-positive ของ complex แย่กว่า miss; miss แค่เสีย planner call เปล่า)
# แยก Thai (ไม่มี \b — knowhow: \b ไม่ทำงานกับ Thai combining chars) ออกจาก English (\b กัน "hi"⊂"highway")
_TRIVIAL_RE = re.compile(
    r'^\s*(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|ขอบคุณ|ขอบใจ|โอเค|บาย)'
    r'|^\s*(hello|hi|hey|thanks?|bye|ok(ay)?)\b',
    re.IGNORECASE,
)
_RECALL_RE = re.compile(
    r'เมื่อกี้|ก่อนหน้า(นี้)?|ที่ถามไป|ถามอะไร(ไป|มา)|คุยเรื่องอะไร|(ครั้ง|รอบ|คราว)ที่แล้ว|^\s*ที่แล้ว',
    re.IGNORECASE,
)


def is_trivial(query: str) -> bool:
    """greeting/thanks/recall detector — DIAGNOSTIC ONLY, not called from any production
    path (no caller in graph.py/react.py/tools/). Used by the unit test suite to sanity-check
    the trivial/recall regexes stay correct."""
    q = query.strip()
    return bool(_TRIVIAL_RE.match(q) or _RECALL_RE.search(q))


def _today() -> str:
    """วันที่ปัจจุบัน (YYYY-MM-DD) — inject เข้า system prompt ทุกครั้งที่เรียก plan()
    ป้องกัน planner สับสนคำอ้างอิงเวลาสัมพัทธ์ ("ปีนี้"/"ล่าสุด"/"ตอนนี้") กับวันที่จริงของ training data"""
    return datetime.date.today().strftime("%Y-%m-%d")


_PLANNER_SYSTEM = """You classify a user request into simple or complex, then produce a step-by-step OUTCOME plan. Output ONLY JSON, nothing else.

Decision chain — stop at the first match:
S1. SIMPLE — answerable in one pass: a single fact/price lookup, one file operation, a concept/math/opinion question, one-subject summary, casual chat.
    -> {"mode":"simple"}
S2. COMPLEX — needs 2–6 distinct steps whose results must be combined: researching multiple angles or sources, multi-file analysis, "do A then B then C", comparing alternatives, auditing/testing + synthesis, collecting data from many sources before summarizing.
    -> {"mode":"complex","plan":["step1","step2",...]}

Rules for plan steps:
- Each step describes the OUTCOME (what you will KNOW after this step) — NOT the method, tool, or search action.
- Each step covers ONE specific angle or aspect. Do not combine multiple concerns in one step.
- Use outcome verbs: สรุป / อธิบาย / วิเคราะห์ / เปรียบเทียบ / รวบรวม / ประเมิน / ตรวจสอบ / ระบุ
- Use "ตรวจสอบ/lookup" ONLY for steps that genuinely need current live data (prices, today's providers, real-time stats).
- NEVER use "ค้นหา" or "หา" as the leading verb — these cause the executor to web_search every step unnecessarily.
- NEVER write filler steps like "รวบรวมข้อมูลทั่วไป" or "สรุปทุกอย่าง" — each step must answer a specific sub-question.
- Minimum 2 steps. Maximum 6 steps. Use more steps only when each adds a genuinely distinct angle that contributes to the final answer.
- Output ONLY the JSON object.

Today's date: __TODAY__ — use it to resolve relative-time references ("ปีนี้"/"ล่าสุด"/"ตอนนี้").

GOOD examples (specific outcome, one angle per step, covers all needed aspects):
{"mode":"simple"}
{"mode":"complex","plan":["สรุปสถาปัตยกรรมและ API หลักของระบบ","ตรวจสอบ provider และค่าธรรมเนียมปัจจุบันในตลาด","อธิบายวิธีพัฒนาและข้อควรระวังด้านความปลอดภัย"]}
{"mode":"complex","plan":["อ่านไฟล์แรกเพื่อดูโครงสร้างและ interface","อ่านไฟล์ที่สองแล้วระบุจุดต่างในตรรกะ","สรุปผลกระทบของความต่างต่อ downstream system"]}
{"mode":"complex","plan":["วิเคราะห์สาเหตุหลักของเหตุการณ์","ประเมินผลกระทบที่ตามมาในระยะสั้นและระยะยาว","เปรียบเทียบกับกรณีในอดีตที่คล้ายกัน","สรุปบทเรียนและข้อเสนอแนะ"]}
{"mode":"complex","plan":["รวบรวมข้อมูลสถิติและแนวโน้มล่าสุดจากหลายแหล่ง","วิเคราะห์ปัจจัยหลักที่ขับเคลื่อนแนวโน้มนั้น","ระบุความเสี่ยงและโอกาสสำหรับผู้เล่นในตลาด","สรุปข้อเสนอแนะเชิงกลยุทธ์"]}
{"mode":"complex","plan":["ตรวจสอบ code ส่วนที่เกี่ยวข้องเพื่อระบุจุดเสี่ยงเบื้องต้น","วิเคราะห์ช่องโหว่ที่พบและประเมินระดับความรุนแรง","ระบุผลกระทบและ attack surface ที่เป็นไปได้","สรุป risk overview พร้อม action items ที่จัดลำดับความเร่งด่วน"]}
{"mode":"complex","plan":["รวบรวมและอ่านเนื้อหาจากแหล่งที่ระบุ","วิเคราะห์ key insights และระบุมุมมองหลักจากแต่ละแหล่ง","เปรียบเทียบความเหมือนและความต่างของมุมมองระหว่างแหล่ง","สรุปภาพรวมพร้อมอ้างอิงแหล่งที่มา"]}
{"mode":"complex","plan":["อ่านและทำความเข้าใจโครงสร้างข้อมูล","วิเคราะห์ pattern แนวโน้ม และ outlier ในข้อมูล","ระบุ insight สำคัญที่ตอบคำถามหลัก","สรุปผลวิเคราะห์พร้อมข้อจำกัดและข้อควรระวัง"]}

BAD steps — never use these:
- "ค้นหาข้อมูลเกี่ยวกับ X" — method not outcome
- "หาข้อมูลจากอินเทอร์เน็ต" — method not outcome
- "รวบรวมข้อมูลทั่วไป" — filler, no specific sub-question
- "สรุปทุกอย่างที่หาได้" — too vague"""


def _parse(text: str) -> dict | None:
    """ดึง JSON + validate locked contract (mode simple/complex, plan=list[str]≥2).
    Returns None on ANY schema violation (bad JSON, unknown mode, plan not a list,
    <2 valid steps) so plan()'s retry loop fires — never silently coerces a
    schema violation to simple; only the final fallback in plan() does that."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, c in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except Exception:
                    return None
                mode = obj.get("mode")
                if mode == "simple":
                    return {"mode": "simple"}
                if mode == "complex":
                    raw_plan = obj.get("plan", [])
                    if not isinstance(raw_plan, list):
                        return None  # schema ผิด (เช่น plan เป็น string) → retry, not simple
                    steps = [str(s).strip() for s in raw_plan if str(s).strip()]
                    if len(steps) >= 2:
                        if len(steps) > 6:
                            # keep head + final step: the last step is the synthesis
                            # step — never drop it via a blunt head-only truncation
                            steps = steps[:5] + steps[-1:]
                        return {"mode": "complex", "plan": steps}
                    return None  # complex แต่ <2 step → retry, not simple
                return None
    return None


_planner_llm = None  # module-level cache: avoid rebuilding ChatOpenAI + httpx pool every plan() call


def _get_planner_llm():
    global _planner_llm
    if _planner_llm is None:
        # planner = JSON classification task (simple/complex + step list)
        # ไม่ต้อง reasoning chain → ปิด thinking ลด latency ~10× (verified 2026-05-29)
        _planner_llm = build_llm(
            temperature=0.0,
            max_tokens=768,
            extra_body={"enable_thinking": False},
        )
    return _planner_llm


def plan(query: str) -> dict:
    """คืน {"mode":"simple"} หรือ {"mode":"complex","plan":[...]}"""
    llm = _get_planner_llm()
    system = _PLANNER_SYSTEM.replace("__TODAY__", _today())
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    last = ""
    for attempt in range(2):
        if attempt > 0:
            messages += [
                {"role": "assistant", "content": last},
                {"role": "user", "content": "Invalid. Output ONLY the JSON object, nothing else."},
            ]
        resp = llm.invoke(messages, config={"callbacks": []})
        last = resp.content or ""
        parsed = _parse(last)
        if parsed:
            log.info(f"[planner] mode={parsed['mode']} steps={len(parsed.get('plan', []))}")
            return parsed
        log.warning(f"[planner] JSON parse failed attempt={attempt}: {last[:80]!r}")
    log.warning("[planner] fallback -> simple")
    return {"mode": "simple"}
