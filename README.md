# ENDEAVOR AGENT TH

Local AI Agent ที่รันบนเครื่องของคุณเอง — ไม่มี API key, ไม่มีค่า token, ไม่มีข้อมูลหลุดออกไปไหน

---

## เริ่มใช้งานเร็ว (Quick Start)

```bash
# ติดตั้งครั้งเดียว (สร้าง conda env "mlx" + ติดตั้งทุกอย่าง + copy .env)
bash install_library/install.sh

# terminal 1: เปิด MLX server
conda activate mlx
mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --port 8080

# terminal 2: รัน agent — เลือกแบบที่ต้องการ
conda activate mlx

# แบบ CLI
python endeavor_agent.py

# หรือแบบ Streamlit UI (เปิด browser)
python agent_server.py &
streamlit run streamlit_app.py
```

ถ้า run ไม่ได้เพราะ port ถูกใช้อยู่ (เช่น เปิด server ค้างจากรอบก่อน) ให้ kill ตัวเก่าก่อน:

```bash
# kill MLX server (port 8080)
lsof -ti:8080 | xargs kill -9

# kill agent_server.py (port 8765)
lsof -ti:8765 | xargs kill -9
```

---

## มันทำอะไรได้บ้าง?

พิมพ์ภาษาไทยแล้วมันจัดการเอง

```
คุณ: วิเคราะห์หุ้น PTT กับ CPALL ว่าตัวไหนน่าสนใจกว่า

agent: [ค้นหาข้อมูลทั้งสองบริษัท → เปรียบเทียบ P/E, yield, ราคา → สรุปผล]
```

```
คุณ: อ่านไฟล์ sales.csv แล้ววาดกราฟยอดขายรายเดือน

agent: [อ่านไฟล์ → คำนวณ → render กราฟ → เปิดให้ดูเลย]
```

```
คุณ: หาข้อมูล AI agent framework ที่น่าสนใจ 10 ตัวแล้วสรุปให้

agent: [ค้นหาหลายมุม → อ่าน 10 แหล่ง → สรุปเป็นตาราง]
```

ไม่ต้องบอกว่าให้ใช้ tool อะไร — มันตัดสินใจเอง

---

## Architecture

```
user input
    ↓
ReAct Agent (LangGraph)
    ↓ วิเคราะห์ query
    ↓ เลือก tool
    ↓ รัน tool → ได้ผล
    ↓ คิดต่อ → tool ถัดไป หรือ ตอบ
final answer
```

ใช้ **LangGraph `create_react_agent`** — single loop ที่ agent คิด→ทำ→คิด→ทำ จนได้คำตอบ
โมเดลหลัก: **Qwen3.6-35B** ผ่าน `mlx_lm.server` (Apple Silicon)

---

## Tools ที่มีให้

| หมวด | Tools |
|---|---|
| **Web** | `web_search`, `browse_url`, `batch_browse`, `fetch_sitemap`, `scrape_table` |
| **File** | `read_file`, `write_file`, `edit`, `grep`, `workspace_ls` |
| **Code** | `python_exec`, `bash`, `plot` |
| **Research** | `tool_loop`, `research_orchestrator`, `create_plan` |
| **Vision** | `read_image` |
| **Memory** | `remember`, `recall_web`, `scratch_write/read/clear` |
| **Browser** | `browser_use` |

---

## Requirements

- macOS Apple Silicon (M1/M2/M3/M4/M5)
- RAM ≥ 48 GB (สำหรับ 35B model)
- Python 3.11 via conda (`mlx` env)
- `mlx-lm` installed

> รันโมเดลเล็กกว่าก็ได้ — แก้ `config.py` หรือ set env var `V2_MODEL`

---

## Setup

```bash
# 1. clone
git clone https://github.com/halochamp/ENDEAVOR_AGENT_TH
cd ENDEAVOR_AGENT_TH

# 2. ติดตั้ง dependencies ทั้งหมด (สร้าง conda env "mlx" + copy .env ให้อัตโนมัติ)
bash install_library/install.sh

# 3. (optional) แก้ค่า config — ค่า default ใช้งานได้เลย ไม่ต้องแก้ถ้าไม่มีความจำเป็น
#    nano .env

# 4. เปิด MLX server (terminal แยก)
conda activate mlx
mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --port 8080

# 5. รัน agent — เลือกแบบที่ต้องการ
conda activate mlx

# แบบ CLI:
python endeavor_agent.py

# แบบ Streamlit UI (เปิด browser):
python agent_server.py &
streamlit run streamlit_app.py
```

> ติดตั้งเอง (ไม่ใช้สคริปต์): `pip install -r install_library/requirements.txt && playwright install chromium`

---

## ต่อยอดได้ยังไง?

### เพิ่ม Tool ใหม่

สร้างไฟล์ใน `tools/` แค่นั้นเลย:

```python
# tools/my_tool.py
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """อธิบาย tool นี้ให้ agent เข้าใจ"""
    return "ผลลัพธ์"
```

จากนั้น import เพิ่มใน `tools/__init__.py` → agent ใช้ได้เลย

### เพิ่ม Skill Mode

สร้างไฟล์ `skills/<name>.md` พร้อม:

```markdown
## Role
## Workflow
## Tools allowed
## Output format
```

พิมพ์ `/<name>` ใน agent → เข้า skill mode ทันที

### ใช้โมเดลอื่น

```bash
export V2_MODEL="mlx-community/Qwen3-8B-4bit"
export MLX_BASE_URL="http://localhost:8081/v1"
python endeavor_agent.py
```

---

## ต่อ Web UI / Telegram ของตัวเอง (`agent_server.py`)

ปกติ `python endeavor_agent.py` คือ CLI ล้วน — ไม่เปิด port อะไร ไม่เกี่ยวกับหัวข้อนี้เลย

ถ้าอยากต่อ **web UI หรือ Telegram bot ของตัวเอง** มี `agent_server.py` ให้ — เป็น FastAPI server แยก process (รันคนละไฟล์ คนละคำสั่ง):

```bash
python agent_server.py   # เปิด WebSocket + REST บน http://127.0.0.1:8765
```

| Endpoint | ใช้ทำอะไร |
|---|---|
| `ws://localhost:8765/ws` | real-time chat (สำหรับ web UI) |
| `POST /chat` | sync request/response (สำหรับ Telegram bot ฯลฯ) |
| `GET /status`, `/files`, `/file` | health check / อ่านไฟล์ workspace |

**Auth (สำคัญ):** ทุก request ต้องมี token —
- ครั้งแรกที่รัน `agent_server.py` มันจะ **gen token ให้อัตโนมัติ** แล้วเก็บไว้ที่ `.agent_token` (chmod 0600, ไม่ขึ้น git)
- เอาค่าจาก `.agent_token` ไปใส่:
  - REST: header `X-Auth-Token: <token>`
  - WebSocket: query param `ws://localhost:8765/ws?token=<token>`
- ไม่มี token → 401 / connection ปิดทันที (กัน drive-by browser เรียก agent ของคุณโดยไม่รู้ตัว)
- dev only: `AGENT_AUTH_DISABLED=1 python agent_server.py` ปิด auth ชั่วคราว (อย่ารันค้างคู่กับเบราว์เซอร์ทั่วไป)

---

## Commands ใน CLI

```
menu          เปิด mode menu
/history      โหลด conversation history เดิม
/compact      บีบอัด context
/clear        เริ่ม session ใหม่
/exit         ออกจาก skill mode
exit / ออก   ปิดโปรแกรม
```

---

## License

MIT License + Commons Clause

ใช้ส่วนตัวและเพื่อการเรียนรู้ได้อย่างอิสระ
ห้ามนำไปใช้เชิงพาณิชย์โดยไม่ได้รับอนุญาตจากผู้พัฒนา

---

*สร้างโดย [HaloChamp](https://github.com/halochamp)*
