"""In-memory scratchpad for multi-source research tasks.

Survives within a session; cleared between tasks via scratch_clear().
"""
from langchain_core.tools import tool

_PAD: dict[str, str] = {}


@tool
def scratch_write(key: str, value: str) -> str:
    """บันทึก note ชั่วคราวลง scratchpad ระหว่างทำ multi-step research
    ใช้หลังได้ผลจาก tool แต่ละ step เพื่อสะสมข้อมูลก่อน synthesize

    KEY = DIMENSION ที่กำลัง track ไม่ใช่ชื่อ source:
      ✅ "expense", "return", "risk", "pricing", "policy"   (dimension — ช่วย synthesize)
      ❌ "web_data", "search1", "result_1"                  (source — ไม่มีประโยชน์ตอน synthesize)

    ACCUMULATE — หลาย source ใส่ key เดิม คั่นด้วย " | ":
      scratch_write("expense", "KB: 0.5%")                      ← source แรก
      scratch_write("expense", "KB: 0.5% | web: 0.03%")         ← เพิ่ม source สอง
      scratch_write("expense", "KB: 0.5% | web: 0.03% — 10× diff") ← เพิ่ม insight

    _dim CONVENTION — หลัง scratch_clear() ต้องเขียนก่อนเสมอ:
      scratch_write("_dim", "expense, return, risk")   ← roadmap สำหรับ synthesis

    value: ≤200 chars — ตัวเลขหลัก / ข้อขัดแย้ง / ช่องโหว่ เท่านั้น
      ✅ "PTT: P/E 8.2x yield 3.1% — cheap vs sector"
      ✅ "KB: RMF ลดหย่อน 15% | web 2025: 30% — ใช้ web (ใหม่กว่า)"
      ✅ "ESG rating: N/A — ไม่พบทุก source"
      ❌ "PTT stock is trading at 34.0 baht per share with P/E ratio of 8.2x..." (ยาวเกิน / copy ต้นฉบับ)
      ❌ "web search returned information about PTT" (ไม่มีข้อมูลจริง)
    """
    try:
        _PAD[key.strip()] = value.strip()
        return f"[scratch] saved '{key.strip()}' ({len(value)} chars)"
    except Exception as e:
        return f"[error] {e}"


@tool
def scratch_read() -> str:
    """อ่าน notes ทั้งหมดใน scratchpad — เรียกก่อน synthesize คำตอบสุดท้ายเสมอ

    ANSWER GATE — บังคับทุกครั้ง ห้ามข้าม:
      ❌ WRONG: scratch_write(dim, note) → [เขียน final answer ทันที]
      ✅ RIGHT:  scratch_write(dim, note) → scratch_read() → synthesize จาก output เท่านั้น

    Synthesize จาก output ของ scratch_read() ONLY:
      ห้ามใช้ข้อมูลที่จำจาก context หรือเห็นก่อนหน้า — ใช้เฉพาะสิ่งที่ปรากฏใน read output
      เหตุผล: scratchpad เป็น source of truth ที่ได้ distill มาแล้ว — context อาจมีข้อมูลเก่าหรือสับสน

    fail→detect→retry→correct:
      FAIL:    [เขียน final answer โดยไม่ call scratch_read()]
      DETECT:  "ฉันข้าม scratch_read() ไป — ผิด SC4"
      RETRY:   call scratch_read() ทันที
      CORRECT: synthesize ใหม่จาก read output เท่านั้น
    """
    try:
        if not _PAD:
            return "[scratch] empty"
        lines = [f"[{k}] {v}" for k, v in _PAD.items()]
        return "\n".join(lines)
    except Exception as e:
        return f"[error] {e}"


@tool
def scratch_clear() -> str:
    """ล้าง scratchpad ทั้งหมด — เรียกเมื่อเริ่ม multi-step task ใหม่

    ตามด้วย scratch_write("_dim", ...) ทันทีเสมอ:
      scratch_clear()
      scratch_write("_dim", "expense, return, risk")   ← กำหนด dimensions ก่อนเริ่ม collect

    !! ห้ามเรียกกลางงาน — ล้างข้อมูลทุกอย่างที่สะสมมา
    !! scratchpad เป็น in-memory per-session — notes จาก turn ก่อนหน้าหายแล้ว ไม่ต้อง clear ข้ามหน้า
    """
    try:
        _PAD.clear()
        return "[scratch] cleared"
    except Exception as e:
        return f"[error] {e}"
