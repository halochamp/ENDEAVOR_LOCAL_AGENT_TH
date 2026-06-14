"""create_plan tool — main agent เรียกเมื่อเจอคำถามซับซ้อนหลายขั้นตอน

Option B (advisor): tool wraps existing `planner.plan()` LLM ที่มี dedicated prompt
(temp=0, JSON retry, ≥2-step validation, locked plan=list[str] contract) — ดีกว่า
ให้ main agent generate steps เองที่ temp=0.3 + tool-laden context

contract: plan() คืน {"mode":"simple"} หรือ {"mode":"complex","plan":[str,...]}
- ถ้า simple → tool บอก agent ตรงๆ ว่าไม่ต้องวางแผน ตอบเลย
- ถ้า complex → format steps เป็น Thai numbered list ให้ agent ทำตามลำดับ
"""
from __future__ import annotations
from langchain_core.tools import tool

from planner import plan as _plan

# Shared return strings — single source of truth for both the create_plan tool and
# the graph.py deterministic enforcement path (no dual formatting).
SIMPLE_MSG = "คำถามนี้ตอบได้ในขั้นเดียว ไม่ต้องวางแผนหลายขั้น — ตอบจากความรู้/เครื่องมือเดี่ยวได้เลย"
EMPTY_PLAN_MSG = "[error] แผนว่าง — ตอบจากความรู้เดิมหรือเครื่องมือเดี่ยวแทน"


def format_plan(steps: list[str]) -> str:
    """Render a validated step list into the Thai numbered plan + EXECUTE-NOW block.

    Single source of truth so the create_plan tool and the graph.py forced-plan
    enforcement emit byte-identical plan text (the model's downstream behaviour is
    driven by this text, so the two paths must not diverge).
    """
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    return (
        f"แผนการทำงาน {len(steps)} ขั้นตอน:\n{numbered}\n\n"
        f"EXECUTE NOW — call the right tool for each step immediately:\n"
        f"  search/find/ค้นหา → web_search → scratch_write(dim, note≤200chars) ทันที\n"
        f"  browse URL → browse_url → scratch_write(dim, note≤200chars) ทันที\n"
        f"  browse multiple URLs in parallel → batch_browse → scratch_write(dim, key findings≤200chars) ทันที\n"
        f"  iterative multi-source loop (search+read many items) → tool_loop → scratch_write(dim, note≤200chars) ทันที\n"
        f"  read file → read_file\n"
        f"  analyze numbers → python_exec → scratch_write(dim, key result≤200chars) ถ้าผลสำคัญต่อ synthesis\n"
        f"  save result → write_file\n"
        f"  synthesize (after web/rag/analysis) → scratch_read() then answer\n"
        f"  synthesize (file/code only) → answer from context (no tool)\n\n"
        f"FORBIDDEN during plan execution:\n"
        f"  bash — not allowed at all. bash cannot search the web and has no research capability.\n"
        f"  create_plan again — this plan is final.\n\n"
        f"Start step 1 now. No announcements, no echo, call the tool directly."
    )


@tool
def create_plan(query: str) -> str:
    """วางแผนงานหลายขั้นตอนสำหรับคำถามที่ซับซ้อน — เรียกก่อนทำงานจริงเมื่อคำถามต้องการหลายขั้น
    (ค้นหลายมุม, ทำหลายไฟล์, A แล้ว B แล้ว C, เปรียบเทียบหลายสิ่ง).
    ห้ามเรียกสำหรับคำถามง่าย/ทักทาย/ความรู้ทั่วไป/คณิตศาสตร์.

    Args:
        query: คำถามเต็มของผู้ใช้

    Returns:
        ขั้นตอน 2-6 ข้อเป็นภาษาไทย — ทำตามลำดับโดยใช้เครื่องมืออื่น
    """
    try:
        result = _plan(query)
    except Exception as e:
        return f"[error] วางแผนไม่สำเร็จ: {e}"

    if result.get("mode") == "simple":
        return SIMPLE_MSG

    steps = result.get("plan", [])
    if not steps:
        return EMPTY_PLAN_MSG

    return format_plan(steps)
