"""planner.py — trivial-skip + classify simple/complex + plan

- trivial-skip: greeting/thanks/recall → ข้าม planner (knowhow A5: deterministic before LLM)
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

# trivial-skip — conservative (false-positive ของ complex แย่กว่า miss; miss แค่เสีย planner call เปล่า)
# แยก Thai (ไม่มี \b — knowhow: \b ไม่ทำงานกับ Thai combining chars) ออกจาก English (\b กัน "hi"⊂"highway")
_TRIVIAL_RE = re.compile(
    r'^\s*(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|ขอบคุณ|ขอบใจ|โอเค|บาย)'
    r'|^\s*(hello|hi|hey|thanks?|bye|okay?)\b',
    re.IGNORECASE,
)
_RECALL_RE = re.compile(
    r'เมื่อกี้|ก่อนหน้า(นี้)?|ที่ถามไป|ถามอะไร(ไป|มา)|คุยเรื่องอะไร|ที่แล้ว',
    re.IGNORECASE,
)


def is_trivial(query: str) -> bool:
    """greeting/thanks/recall → ข้าม planner ไป react ตรงๆ (optimization)"""
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
    """ดึง JSON + validate locked contract (mode simple/complex, plan=list[str]≥2)"""
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
                    steps = [str(s).strip() for s in obj.get("plan", []) if str(s).strip()]
                    if len(steps) >= 2:
                        return {"mode": "complex", "plan": steps[:6]}
                    return {"mode": "simple"}  # complex แต่ <2 step → simple
                return None
    return None


def plan(query: str) -> dict:
    """คืน {"mode":"simple"} หรือ {"mode":"complex","plan":[...]}"""
    # planner = JSON classification task (simple/complex + step list)
    # ไม่ต้อง reasoning chain → ปิด thinking ลด latency ~10× (verified 2026-05-29)
    llm = build_llm(
        temperature=0.0,
        max_tokens=768,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
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
        resp = llm.invoke(messages)
        last = resp.content or ""
        parsed = _parse(last)
        if parsed:
            log.info(f"[planner] mode={parsed['mode']} steps={len(parsed.get('plan', []))}")
            return parsed
        log.warning(f"[planner] JSON parse failed attempt={attempt}: {last[:80]!r}")
    log.warning("[planner] fallback -> simple")
    return {"mode": "simple"}
