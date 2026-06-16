# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

import os
from langchain_core.tools import tool

_MEMORY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "memory.md"))
_MEMORY_MAX_CHARS = 5_000  # ~50-100 facts; keeps system prompt bounded


def _trim_memory(path: str, max_chars: int) -> None:
    """Drop oldest lines until file fits within max_chars."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        while lines and sum(len(l) for l in lines) > max_chars:
            lines.pop(0)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        pass


@tool
def remember(fact: str) -> str:
    """บันทึกข้อมูลสำคัญเกี่ยวกับผู้ใช้ลง memory.md ถาวร
    ใช้เมื่อผู้ใช้พูดว่า 'จำไว้ว่า...', 'จำด้วยว่า...', 'บันทึกว่า...', 'remember that...'
    fact คือข้อความที่จะจำ ควรกระชับชัดเจน
    """
    try:
        with open(_MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"- {fact.strip()}\n")
        _trim_memory(_MEMORY_FILE, _MEMORY_MAX_CHARS)
        return f"บันทึกแล้ว: {fact.strip()}"
    except Exception as e:
        return f"[error] {e}"
