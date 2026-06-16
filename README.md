# ENDEAVOR_LOCAL_AGENT_TH

**Local AI Agent ที่ออกแบบให้รองรับภาษาไทยโดยเฉพาะ ที่รันบนเครื่องของคุณเอง 100%**
ไม่มี API key, ไม่มีค่า token รายเดือน, ไม่มีข้อมูลหลุดออกไปนอกเครื่อง — ขับเคลื่อนด้วย **Qwen3.6-35B** ผ่าน **MLX** บน Apple Silicon และ orchestrate ด้วย **LangGraph ReAct Agent**

---

## บทนำ

มนุษย์ค้นพบไฟ แล้วทุกอย่างก็เปลี่ยนไป — ไฟให้ความอบอุ่น ปรุงอาหาร และปกป้องเราจากอันตราย AI ก็เช่นกัน มันคือ 'ไฟ' ของยุคนี้ เพียงแต่ไฟเกิดขึ้นเองตามธรรมชาติ ส่วน AI คือสิ่งที่เราต้องหยิบพลังของมันมาจัดสรรและควบคุมด้วยตัวเอง

**รถถังเช่า vs ปืนพกส่วนตัว**

Cloud AI เปรียบเหมือนรถถังที่เราเช่ามาใช้: ทรงพลัง แม่นยำ และยิงได้ไกล แต่เรากำลังยืมจมูกคนอื่นหายใจ วันดีคืนดีเขาอาจจะเปลี่ยนเงื่อนไข ขึ้นราคา หรือปิดระบบไม่ให้เราเช่าเมื่อไหร่ก็ได้ ที่สำคัญ ทุกครั้งที่เราใช้งาน ข้อมูลทั้งหมดต้องถูกส่งกลับไปให้เจ้าของระบบรับรู้เสมอ

Local AI เปรียบเหมือนปืนพกส่วนตัว: เล็กกว่า คล่องตัวกว่า แม้ไม่ได้ทรงพลังเท่ารถถัง แต่สิ่งที่ได้กลับมาคือ ความมั่นคงและอิสรภาพ

**ความจริงของ Local AI**

เราอาจไม่ได้เป็นเจ้าของมัน 100% ในแง่ของลิขสิทธิ์หรือการสร้างขึ้นมาเองตั้งแต่ศูนย์ แต่คุณค่าที่แท้จริงคือ เราสามารถควบคุมและใช้งานมันได้ 100% ในวันที่เราจำเป็นต้องใช้ ต่อให้โลกภายนอกป่วน อินเทอร์เน็ตล่ม หรือระบบคลาวด์ปิดตัว มันจะยังคงทำงานอยู่บนเครื่องของคุณอย่างปลอดภัย โดยไม่มีใครมาแอบดูข้อมูลหรือเรียกเก็บเงินรายเดือนจากคุณ

**สิ่งที่เรากำลังทำ**

โปรเจกต์นี้ไม่ได้พยายามสร้างรถถังไปแข่งกับใคร เพราะเราทำแบบนั้นไม่ได้ สิ่งที่เราทำคือการส่งมอบ "ปืนพกที่ดีพอ" ให้กับคอมพิวเตอร์ส่วนตัวของคุณ

มันเป็นเครื่องมือที่ทำงานอยู่บนเครื่องของคุณโดยตรง แต่ฉลาดเพียงพอที่จะช่วยค้นเว็บ อ่านไฟล์ วิเคราะห์ข้อมูล วาดกราฟ และสื่อสารภาษาไทยได้อย่างทรงประสิทธิภาพ โดยที่คุณเป็นผู้ควบคุมทุกอย่างเองทั้งหมดอย่างแท้จริง และสามารถพัฒนาต่อยอดได้ด้วยตัวคุณเอง

---

## สารบัญ

- [บทนำ](#บทนำ)
- [เริ่มใช้งานเร็ว (Quick Start)](#เริ่มใช้งานเร็ว-quick-start)
- [ENDEAVOR Agent ทำอะไรได้บ้าง?](#endeavor-agent-ทำอะไรได้บ้าง)
- [UI ที่มีให้ (2 แบบ)](#ui-ที่มีให้-2-แบบ)
- [หลักการทำงานของ Agent](#หลักการทำงานของ-agent)
- [เทคโนโลยีที่ใช้](#เทคโนโลยีที่ใช้)
- [Security](#security)
- [Tools ที่มีให้ (22 tools)](#tools-ที่มีให้-22-tools)
- [Skill Modes](#skill-modes)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration (.env)](#configuration-env)
- [ต่อ Web UI / Telegram ของตัวเอง](#ต่อ-web-ui--telegram-ของตัวเอง-agent_serverpy)
- [Commands ใน CLI](#commands-ใน-cli)
- [ต่อยอดได้ยังไง?](#ต่อยอดได้ยังไง)
- [License](#license)
- [ผู้พัฒนา](#ผู้พัฒนา)

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

# แบบ CLI (วิธีที่ 1: activate เอง)
python endeavor_agent.py

# แบบ CLI (วิธีที่ 2: ไม่ต้อง activate — run.sh จัดการให้)
bash run.sh

# หรือแบบ Web UI (VS Code-style, เปิด browser ที่ http://localhost:8765/ui)
python agent_server.py
```

ถ้า run ไม่ได้เพราะ port ถูกใช้อยู่ (เช่น เปิด server ค้างจากรอบก่อน) ให้ kill ตัวเก่าก่อน:

```bash
# kill MLX server (port 8080)
lsof -ti:8080 | xargs kill -9

# kill agent_server.py (port 8765)
lsof -ti:8765 | xargs kill -9
```

> เพิ่งเคย clone ครั้งแรก หรืออยากดูทุกขั้นตอนแบบละเอียด (รวม `git clone`, config, ติดตั้งแบบไม่ใช้สคริปต์) → ดู [Setup](#setup)

---

## ENDEAVOR Agent ทำอะไรได้บ้าง?

พิมพ์ภาษาไทยแล้วมันจัดการเอง — agent วิเคราะห์ query เอง เลือก tool เอง วน loop จนกว่าจะได้คำตอบ

```
คุณ: วิเคราะห์หุ้น PTT กับ CPALL ว่าตัวไหนน่าสนใจกว่า

agent: [ค้นหาข้อมูลทั้งสองบริษัท → เปรียบเทียบ P/E, yield, ราคา → สรุปผล]
```

```
คุณ: อ่านไฟล์ sales.csv แล้ววาดกราฟยอดขายรายเดือน

agent: [อ่านไฟล์ → คำนวณด้วย pandas → render กราฟ matplotlib → เปิดให้ดูเลย]
```

```
คุณ: หาข้อมูล AI agent framework ที่น่าสนใจ 10 ตัวแล้วสรุปให้

agent: [วางแผน → ค้นหาหลายมุม → อ่าน 10 แหล่ง → สรุปเป็นตาราง]
```

ไม่ต้องบอกว่าให้ใช้ tool อะไร — มันตัดสินใจเอง

> **ติดปัญหา หรือไม่รู้จะเริ่มยังไง?** แนะนำให้เปิดไฟล์ `install_library/AI_SETUP.md`
> ด้วย AI assistant ของคุณ (Claude Code, Cursor, ฯลฯ) แล้วถามได้เลย — ไฟล์นี้เขียนไว้
> ให้ AI ช่วย setup และตอบคำถามการใช้งานทั่วไปแทนคุณโดยเฉพาะ

---

## UI ที่มีให้ (2 แบบ)

เลือกใช้ได้ทั้ง 2 แบบ — ทำงานเหมือนกัน ต่างกันที่หน้าตา ทุกแบบ**บอก user เสมอว่า agent กำลังทำอะไรอยู่** ไม่ใช่แค่รอคำตอบเฉย ๆ

### 1. CLI (`endeavor_agent.py`)

- รันใน terminal — เบา เร็ว เหมาะกับงานที่ต้องทำซ้ำ ๆ หรือเปิดทิ้งไว้นาน ๆ
- ระหว่าง agent ทำงาน จะเห็น **spinner 2 บรรทัด** ตลอดเวลา:
  - บรรทัดบน: phase หลัก เช่น "กำลังคิด...", "กำลังเรียก tool..."
  - บรรทัดล่าง (`⎿`): sub-status รายละเอียด tool ที่กำลังรัน เช่น `web_search("...")`, `read_file("data.csv")`
- มี context status bar แสดงการใช้ context (token) แบบ real-time

### 2. Web UI — VS Code-style (`agent_server.py` + `chat.html`)

- รัน `python agent_server.py` แล้วเปิด `http://localhost:8765/ui` — single-file HTML, dark theme คล้าย VS Code
- Layout 3 ส่วน: icon sidebar (Workspace / Activity / History / Commands) | side panel | chat
- **Workspace panel**: เรียกดูไฟล์ในโฟลเดอร์ทำงานเป็น tree — กดดูเนื้อหา (text/code แสดงเป็น code block, รูปภาพแสดง preview ในตัว), เปิดด้วยโปรแกรม default ของเครื่อง (✏️), หรือลบไฟล์ (🗑 กด 2 ครั้งเพื่อยืนยัน)
- **Activity panel**: log การเรียก tool ทั้งหมดของ session แบบเรียงตามลำดับเวลา
- **History panel**: ดูประวัติคำถาม-คำตอบที่เคยคุยไว้
- **Commands panel**: เมนู `/clear`, `/compact`, `/history`, skill ต่าง ๆ (`/research`, `/pdf_to_text`) — กดเปิด/ปิดได้โดยไม่ต้องพิมพ์
- แชทแบบ streaming token-by-token, รองรับ markdown, command autocomplete พิมพ์ `/` แล้วเลือกจาก dropdown
- เมื่อเปิด skill mode (เช่น `/pdf_to_text`) ป้ายชื่อผู้ตอบในแชทจะเปลี่ยนเป็น `Agent | pdf_to_text` ให้เห็นชัดว่ากำลังอยู่ใน mode ไหน

ทั้ง 2 แบบเชื่อมต่อ MLX server ตัวเดียวกัน (`localhost:8080`) — เลือกใช้ตัวไหนก็ได้ ไม่ต้องรันพร้อมกัน

---

## หลักการทำงานของ Agent

### ReAct Loop (Reason → Act → Observe)

```
user input
    │
    ▼
┌─────────────────────────────────────────────┐
│  ReAct Agent  (LangGraph create_react_agent) │
│                                               │
│   1. วิเคราะห์ query + ประวัติการสนทนา        │
│   2. งานซับซ้อน? → เรียก create_plan ก่อน     │
│   3. เลือก tool ที่เหมาะสมจาก 22 tools         │
│   4. รัน tool → ได้ผลลัพธ์ (Observation)       │
│   5. คิดต่อ: ทำต่อ tool ถัดไป หรือ ตอบเลย      │
│      ↑_____________________________│         │
│         วน loop จนกว่าจะพอ                     │
└─────────────────────────────────────────────┘
    │
    ▼
final answer (ภาษาไทย)
```

### Single-Node Graph Design

ทั้งระบบ orchestrate ด้วย **LangGraph** ที่มี node เดียว (`graph.py`):

```
START → react (agent คุมเองทั้งหมด) → END
```

ไม่มี `planner_node` / `execute_node` / `synthesize_node` แยกเป็น node ต่าง ๆ
— main agent เห็น **full message history + tool results ทั้งหมด** ภายใน loop เดียว
แล้วตัดสินใจเองว่าจะ "วางแผน → ทำตามแผน → สรุปคำตอบ" หรือ "ตอบตรง ๆ" ลด overhead จากการส่ง state ข้าม node และลดจุดที่ context หลุด

### องค์ประกอบหลัก

| ส่วน | ทำหน้าที่ |
|---|---|
| **`react.py`** (compiled `.so`) | สร้าง ReAct agent + system prompt + คำนวณ context stats |
| **`planner.py`** (compiled `.so`) | ตรวจว่า query เป็น trivial หรือซับซ้อน → ควรเรียก `create_plan` ก่อนไหม |
| **`graph.py`** | ผูก agent เข้ากับ LangGraph state machine, จัดการ retry เมื่อ synthesis ล้มเหลว, deterministic intercept สำหรับ search/research intent |
| **`llm.py`** | สร้าง `ChatOpenAI` client ชี้ไปที่ `mlx_lm.server` (OpenAI-compatible API) |
| **`runtime_common.py`** | infra ร่วมระหว่าง CLI กับ Web UI — memory store, liveness check, skill detection (single source of truth ตาม Dual-Path Prohibition) |

### Context & Memory Management

- **Context trimming**: ตัดข้อความเก่าทิ้งเมื่อ conversation ยาวเกิน `CONTEXT_MAX_CHARS` (default 200K chars ≈ 50K tokens) — กัน context overflow บน session ยาว
- **Web cache**: ผลลัพธ์จาก `web_search` / `browse_url` / `browser_use` ถูกเก็บ raw ไว้ใน process memory แยกจาก message history — agent เห็นแค่ summary สั้น ๆ แล้วเรียก `recall_web` ถ้าต้องการเนื้อหาเต็ม
- **Persistent memory**: `remember` tool เขียนข้อเท็จจริงสำคัญลง `logs/memory.md` ข้าม session
- **Conversation history**: เก็บผ่าน LangGraph `SqliteSaver` (`logs/history.db`) — `/history` ใน CLI โหลดกลับมาได้

### Generation Tuning

| พารามิเตอร์ | ค่า default | ผลลัพธ์ |
|---|---|---|
| `TEMPERATURE` | 0.1 | คำตอบ deterministic, เหมาะกับ tool calling |
| `THINKING_BUDGET` | 2048 tokens | จำกัดเวลาที่โมเดล "คิด" ก่อนตอบ — กัน query ง่าย ๆ คิดนานเกินไป |
| `REPETITION_PENALTY` | 1.05 | กัน thinking loop ซ้ำ ๆ โดยไม่กระทบ JSON ของ tool call |
| `RECURSION_LIMIT` | 60 | จำนวน step สูงสุดต่อ 1 query (รองรับ research 4 ขั้นตอน × ~12 tool calls) |

---

## เทคโนโลยีที่ใช้

| Layer | เทคโนโลยี | หน้าที่ |
|---|---|---|
| **LLM Runtime** | [MLX](https://github.com/ml-explore/mlx) | รัน Qwen3.6-35B แบบ quantized (4-bit) บน Apple Silicon GPU ผ่าน Metal |
| **Model** | `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` (MoE) | reasoning + tool calling — สลับเป็นโมเดลเล็กกว่าได้ผ่าน `.env` |
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph) `create_react_agent` | ReAct loop, state graph, checkpointing |
| **LLM Client** | LangChain Core + `langchain-openai` | คุยกับ `mlx_lm.server` ผ่าน OpenAI-compatible API |
| **Backend Server** | [FastAPI](https://fastapi.tiangolo.com/) + `uvicorn` | WebSocket (real-time streaming) + REST endpoints |
| **Web UI** | `chat.html` (vanilla JS) + marked.js | VS Code-style chat UI พร้อม activity log, file viewer, streaming response |
| **CLI** | `prompt_toolkit` + `rich` | interactive terminal, autocomplete, markdown rendering |
| **Web Search** | `ddgs` (DuckDuckGo) | ค้นหาข้อมูล real-time ไม่ต้องใช้ API key |
| **Web Scraping** | `trafilatura`, `requests`, Jina Reader | ดึงเนื้อหาเว็บ → clean text |
| **Browser Automation** | `playwright` + `browser-use` | ควบคุม browser จริงสำหรับเว็บ JS-heavy / SPA |
| **Data Processing** | `pandas`, `numpy` | วิเคราะห์ข้อมูล CSV/Excel |
| **Visualization** | `matplotlib` + Pillow | สร้างกราฟ พร้อมรองรับฟอนต์ไทย (Noto Sans Thai / Thonburi) |
| **Document Parsing** | `markitdown[pdf,docx,xlsx,xls]` | แปลง PDF/Word/Excel → markdown ให้ agent อ่านได้ |
| **OCR** | Apple Vision Framework (`pyobjc`) | อ่านข้อความจากภาพ — แม่นยำสูงสำหรับภาษาไทย, native macOS |
| **Persistence** | SQLite (`langgraph.checkpoint.sqlite`) | เก็บ conversation history แบบ persistent |
| **Config** | `python-dotenv` | โหลด `.env` อัตโนมัติ — ปรับ config ได้โดยไม่แก้ code |
| **Sandboxing** | macOS `sandbox-exec` | จำกัด `bash` tool ให้เขียนไฟล์ได้เฉพาะใน workspace |

---

## Security

Agent ตัวนี้ออกแบบมาให้ "เขียนได้แค่ใน sandox แต่ดูเห็นได้กว้าง" — เน้นให้ agent ทำงานอัตโนมัติได้เต็มที่ โดยไม่เสี่ยงทำลายหรือหลุดข้อมูลของเครื่อง

### ข้อมูลอยู่ในเครื่องคุณเท่านั้น

โมเดล LLM รันบนเครื่องผ่าน `mlx_lm.server` (MLX, Apple Silicon GPU) — บทสนทนา, ไฟล์, และ context ทั้งหมดที่ส่งเข้า/ออกจากโมเดล **ไม่ถูกส่งออกไปนอกเครื่อง** ไม่มี API call ไป cloud LLM provider ใด ๆ (OpenAI, Anthropic ฯลฯ) ในการทำงานปกติ

ข้อยกเว้นเดียวคือเมื่อ agent **เลือกเรียกใช้** เครื่องมือที่ต้องคุยกับอินเทอร์เน็ตตามคำสั่งของคุณเอง เช่น `web_search`, `browse_url`, `browser_use` — กรณีนี้เฉพาะ "คำค้น/URL" ที่จำเป็นเท่านั้นจะถูกส่งไปยังบริการนั้น ๆ (เช่น DuckDuckGo, เว็บปลายทาง) ไม่ใช่บทสนทนาทั้งหมด

### หลักการ

- **เขียนไฟล์ได้เฉพาะใน `workspace/`** — เครื่องมือ `write_file`, `edit`, และ `bash`/`python_exec` (เมื่อ spawn process เขียนไฟล์) ถูกจำกัดให้เขียนได้แค่ภายใต้ `workspace/` เท่านั้น พยายามเขียนไฟล์นอก workspace จะถูก block ทันที (เว้นแต่ตั้ง `V2_ALLOW_OUTSIDE=1` ซึ่งเป็น dev-only flag)
- **อ่านไฟล์ได้กว้างกว่า แต่ไม่ใช่ทุกที่** — `read_file`/`read_image`/`grep` อ่านไฟล์นอก `workspace/` ได้ (เช่นให้ agent ช่วยอ่านโค้ดในโปรเจคอื่น หรือเอกสารบนเครื่อง) แต่ path ที่เข้าข่าย "ระบบ/credentials" จะถูก block เสมอ ไม่ว่าจะตั้ง flag ใดก็ตาม
- **Path ที่ block ทั้งอ่านและเขียนเสมอ** — `/etc`, `/usr`, `/bin`, `/sbin`, `/lib`, `/System`, `/Library`, `/Applications`, `~/.ssh`, `~/.aws`, `~/.gnupg`
- **กัน symlink/`../` traversal** — ทุก path ผ่าน `os.path.realpath()` ก่อนเช็ค ป้องกัน trick เช่น สร้าง symlink ใน workspace ชี้ออกไป `~/.ssh` หรือใช้ `../../etc/passwd`

### การควบคุม `bash` / `python_exec` (process-level sandbox)

Tool ที่ spawn process จริง (`bash`, `python_exec`) ถูกครอบด้วย **macOS `sandbox-exec`** (Seatbelt) เพิ่มอีกชั้น แยกจาก path-guard ข้างบน:

- `(deny file-write*)` ครอบ `/etc`, `/usr`, `/bin`, `/sbin`, `/System`, `/Library`, `/Applications`, และโฟลเดอร์ผู้ใช้ที่สำคัญ — `~/Desktop`, `~/Documents`, `~/Downloads`, `~/Movies`, `~/Music`, `~/Pictures`, `~/Library`, `~/.ssh`, `~/.aws`, `~/.config`, `~/.gnupg`
- `(deny file-read*)` ครอบ `~/.ssh`, `~/.aws`, `~/.gnupg` — แม้ใน sandbox process ก็อ่าน credentials เหล่านี้ไม่ได้
- `(allow file-write*)` เปิดเฉพาะ `workspace/` และ `/private/tmp` (last-match wins ใน sandbox profile → override deny ด้านบน)
- ทุก process มี **timeout** (`bash` default 30s, `python_exec` ปรับตาม workload) — กัน infinite loop ค้าง resource

### สรุป

| สิ่งที่ทำได้ | สิ่งที่ทำไม่ได้ |
|---|---|
| อ่านไฟล์/โค้ด/เอกสารทั่วเครื่อง (นอก system paths) | อ่าน/เขียน `~/.ssh`, `~/.aws`, `~/.gnupg`, `/etc`, `/System` ฯลฯ |
| เขียน/แก้/รันโค้ดใน `workspace/` | เขียนไฟล์นอก `workspace/` (รวม Desktop, Documents, Downloads ผ่าน `bash`) |
| รัน shell command / python script ผ่าน sandbox | escape sandbox ด้วย symlink หรือ `../` traversal |

ผลคือ agent ทำงานอัตโนมัติ (รัน loop, เรียก tool ต่อเนื่อง) ได้เต็มที่โดยไม่ต้องกังวลว่าจะไปลบ/แก้ไฟล์สำคัญของเครื่อง หรือหลุดอ่าน credentials โดยไม่ตั้งใจ

---

## Tools ที่มีให้ (22 tools)

Agent เลือก tool เองตาม docstring ของแต่ละ tool — ไม่ต้องสั่งตรง ๆ

### 🌐 Web & Research

| Tool | คำอธิบาย |
|---|---|
| `web_search` | ค้นหาข้อมูล real-time (ราคา, ข่าววันนี้, เหตุการณ์ล่าสุด) ผ่าน DuckDuckGo — auto-fetch เนื้อหาจากผลลัพธ์อันดับต้น ๆ |
| `browse_url` | อ่าน URL ที่กำหนดผ่าน Jina Reader → คืน summary ภาษาไทย, cache เนื้อหาเต็มไว้เรียกย้อนกลับได้ |
| `batch_browse` | อ่านหลาย URL พร้อมกัน (parallel fetch) แล้วคืน summary รวมในครั้งเดียว — ลดจำนวน tool call |
| `browser_use` | ควบคุม browser เหมือนคนจริง — คลิก, scroll, กรอกฟอร์ม, นำทางหลายหน้า (สำหรับเว็บที่ scrape ตรงไม่ได้) |
| `fetch_sitemap` | ดึงรายการ URL ทั้งหมดจาก `sitemap.xml` ของเว็บไซต์ — ใช้เมื่อต้องสำรวจ domain ทั้งหมด |
| `scrape_table` | ดึงตารางจากเว็บ JS-rendered (React/Vue/SPA) ด้วย Playwright → คืนเป็น CSV |
| `recall_web` | ดึงเนื้อหาเต็มของ URL ที่เคย fetch ไปแล้ว (จาก web cache) — ใช้ตอนต้องการรายละเอียดเพิ่มจากที่ summarize ไว้ |

### 📁 File & Code

| Tool | คำอธิบาย |
|---|---|
| `read_file` | อ่านไฟล์ text/code รวมถึง PDF, Word, Excel — แปลงเป็น markdown อัตโนมัติ |
| `write_file` | สร้างไฟล์ใหม่ใน workspace (สำหรับไฟล์ที่ยังไม่มี) |
| `edit` | แก้ไขไฟล์ที่มีอยู่แบบ find & replace (รองรับ replace ทั้งหมด) |
| `grep` | ค้นหา regex pattern ข้ามไฟล์ในโฟลเดอร์ — คืนผลแบบ `file:line: content` |
| `workspace_ls` | แสดงรายการไฟล์ทั้งหมดใน workspace แบบ recursive tree |

### 💻 Code Execution

| Tool | คำอธิบาย |
|---|---|
| `python_exec` | รัน Python code ใน interpreter เดียวกับ agent — `pandas`, `numpy`, `matplotlib` พร้อมใช้ทันที |
| `bash` | รันคำสั่ง bash บนเครื่อง (cwd = workspace) สำหรับงาน system-level — รันใน macOS sandbox จำกัดการเขียนไฟล์นอก workspace |
| `plot` | สร้างกราฟด้วย matplotlib จาก Python code — รองรับฟอนต์ไทยเต็มรูปแบบ, บันทึกและเปิดไฟล์ให้อัตโนมัติ |

### 🖼️ Vision

| Tool | คำอธิบาย |
|---|---|
| `read_image` | อ่านข้อความจากภาพด้วย Apple Vision OCR — แม่นยำสำหรับภาษาไทยและอังกฤษ รองรับทั้งไฟล์ local และ URL |

### 🧠 Memory & Scratchpad

| Tool | คำอธิบาย |
|---|---|
| `remember` | บันทึกข้อมูลสำคัญเกี่ยวกับผู้ใช้ลง `memory.md` แบบถาวร — จำได้ข้าม session |
| `scratch_write` | บันทึก note ชั่วคราวระหว่างทำงาน multi-step |
| `scratch_read` | อ่าน notes ทั้งหมดในscratchpad — เรียกก่อน synthesize คำตอบสุดท้ายเสมอ |
| `scratch_clear` | ล้าง scratchpad ทั้งหมดเมื่อเริ่มงานใหม่ |

### 🗺️ Planning & Loops

| Tool | คำอธิบาย |
|---|---|
| `create_plan` | วางแผนงานหลายขั้นตอนสำหรับ query ที่ซับซ้อน — เรียกก่อนเริ่มทำงานจริงเมื่อ query ต้องใช้หลาย step |
| `tool_loop` | วน loop ประมวลผล items จำนวนมากด้วย Python โดยตรง — ไม่หลุด loop ไม่ว่า items จะมากแค่ไหน (เหมาะกับงาน batch) |

### 🔬 Research Skill (เปิดด้วย `/research`)

| Tool | คำอธิบาย |
|---|---|
| `research_orchestrator` | ค้นหาข่าว/ข้อมูลจาก N แหล่ง แล้วเขียนรายงานสรุป — วน batch โดย Python loop อัตโนมัติ พร้อม checkpoint สำหรับ resume งานที่ทำค้างไว้ |

---

## Skill Modes

Skill mode คือ system prompt + tool set เฉพาะทาง เปิด/ปิดได้ด้วยคำสั่งใน CLI

| Skill | Trigger | ใช้ทำอะไร |
|---|---|---|
| `research` | `/research` | ค้นหาข่าวล่าสุดจากหลายแหล่ง → เขียนรายงาน รองรับ resume งานค้าง |
| `pdf_to_text` | `/pdf_to_text` | แปลง PDF (รวมที่เป็นภาพ/scan) → text ผ่าน OCR + LLM extraction |

เปิด skill ซ้ำ = toggle ปิด, หรือใช้ `/exit` ออกจาก skill mode ปัจจุบัน

---

## Requirements

- macOS Apple Silicon (M1/M2/M3/M4/M5)
- RAM ≥ 48 GB (สำหรับ 35B model)
- Python 3.11 via conda (`mlx` env)
- `mlx-lm` installed

> **หมายเหตุเรื่องโมเดล:** harness นี้ออกแบบและ tune มาเพื่อ **Qwen3.6-35B-A3B (MoE)** ซึ่งเป็น production model ที่ใช้อยู่ — ค่า default ทั้งหมด (prompt, thinking budget, repetition penalty, tool-calling behavior) คาดหวังความสามารถระดับนี้
>
> หากเครื่องไม่พอ (RAM < 48GB) และต้องการรันโมเดลเล็กกว่า แนะนำ **Qwen3-14B** เป็นขั้นต่ำที่ยังพอใช้งาน tool calling ได้สมเหตุสมผล — โมเดลที่เล็กกว่านี้ (เช่น 7B/8B ลงไป) มีโอกาสสูงที่จะ tool-call ผิด, หลุด format, หรือตอบไม่ตรงคำถามบ่อยขึ้น ต้อง tune prompt/parameter เพิ่มเอง
>
> สลับโมเดลทำได้ผ่าน `config.py` หรือ env var `V2_MODEL` + `MLX_BASE_URL` (ดูหัวข้อ [Configuration](#configuration-env))

---

## Setup

```bash
# 1. clone
git clone https://github.com/halochamp/ENDEAVOR_LOCAL_AGENT_TH
cd ENDEAVOR_LOCAL_AGENT_TH

# 2. ติดตั้ง dependencies ทั้งหมด (สร้าง conda env "mlx" + copy .env ให้อัตโนมัติ)
bash install_library/install.sh

# 3. (optional) แก้ค่า config — ค่า default ใช้งานได้เลย ไม่ต้องแก้ถ้าไม่มีความจำเป็น
#    nano .env

# 4. เปิด MLX server (terminal แยก)
conda activate mlx
mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --port 8080

# 5. รัน agent — เลือกแบบที่ต้องการ
conda activate mlx

# แบบ CLI (วิธีที่ 1: activate เอง)
python endeavor_agent.py

# แบบ CLI (วิธีที่ 2: ไม่ต้อง activate — run.sh จัดการให้)
bash run.sh

# แบบ Web UI (เปิด browser ที่ http://localhost:8765/ui):
python agent_server.py
```

> ติดตั้งเอง (ไม่ใช้สคริปต์): `pip install -r install_library/requirements.txt && playwright install chromium`

---

## Configuration (.env)

ทุกค่าตั้งค่าผ่าน environment variable — `config.py` โหลด `.env` อัตโนมัติด้วย `python-dotenv` ค่า default ใช้งานได้เลยโดยไม่ต้องแก้อะไร

```bash
cp .env.example .env   # ทำให้อัตโนมัติโดย install.sh แล้ว
```

หมวดหมู่หลักใน `.env.example`:

| หมวด | ตัวแปรตัวอย่าง | ใช้ทำอะไร |
|---|---|---|
| LLM Backend | `MLX_BASE_URL`, `V2_MODEL`, `MLX_API_KEY` | เปลี่ยน server/โมเดล |
| Generation | `V2_TEMPERATURE`, `V2_THINKING_BUDGET`, `V2_REPETITION_PENALTY`, `V2_RECURSION_LIMIT` | tuning การตอบ |
| Context Window | `V2_CONTEXT_MAX_CHARS` | ขยาย/ลด session length |
| Agent Server | `AGENT_SERVER_PORT`, `AGENT_SERVER_TOKEN`, `AGENT_AUTH_DISABLED` | ตั้งค่า web server |
| Workspace & Logs | `V2_WORKSPACE`, `V2_LOG_DIR`, `V2_LOG_MAX_ENTRIES` | path สำหรับไฟล์งานและ log |
| Web Tool Limits | `V2_WEB_SEARCH_MAX_RESULTS`, `V2_BROWSE_URL_MAX_CHARS`, ฯลฯ | จำกัดขนาดผลลัพธ์จาก web tools |
| Web Cache | `V2_WEB_CACHE_MAX_ENTRIES`, `V2_WEB_CACHE_MAX_BYTES` | จัดการ cache เนื้อหาเว็บ |
| Summarization | `V2_SUMMARY_MAX_CHARS`, `V2_SUMMARY_SKIP_LLM_BELOW` | ควบคุมการสรุปผลลัพธ์ก่อนเข้า context |

ดูรายละเอียดทั้งหมดพร้อม comment อธิบายในไฟล์ `.env.example`

---

## ต่อ Web UI / Telegram ของตัวเอง (`agent_server.py`)

ปกติ `python endeavor_agent.py` คือ CLI ล้วน — ไม่เปิด port อะไร ไม่เกี่ยวกับหัวข้อนี้เลย

ถ้าอยากต่อ **web UI หรือ Telegram bot ของตัวเอง** มี `agent_server.py` ให้ — เป็น FastAPI server แยก process (รันคนละไฟล์ คนละคำสั่ง):

```bash
python agent_server.py   # เปิด WebSocket + REST บน http://127.0.0.1:8765
```

| Endpoint | ใช้ทำอะไร |
|---|---|
| `GET /ui` | เปิด Web UI (VS Code-style) — `chat.html` |
| `GET /ui-token` | ดึง auth token สำหรับหน้า `/ui` (ใช้เฉพาะ same-origin) |
| `ws://localhost:8765/ws` | real-time chat พร้อม token streaming (สำหรับ web UI) |
| `POST /chat` | sync request/response (สำหรับ Telegram bot ฯลฯ) |
| `GET /status`, `/files`, `/file` | health check / อ่านไฟล์ workspace |

**Auth (สำคัญ):** ทุก request ต้องมี token —
- ครั้งแรกที่รัน `agent_server.py` มันจะ **gen token ให้อัตโนมัติ** แล้วเก็บไว้ที่ `.agent_token` (chmod 0600, ไม่ขึ้น git)
- เอาค่าจาก `.agent_token` ไปใส่:
  - REST: header `X-Auth-Token: <token>`
  - WebSocket: `Sec-WebSocket-Protocol` header หรือ query param `ws://localhost:8765/ws?token=<token>`
- ไม่มี token → 401 / connection ปิดทันที (กัน drive-by browser เรียก agent ของคุณโดยไม่รู้ตัว)
- dev only: `AGENT_AUTH_DISABLED=1 python agent_server.py` ปิด auth ชั่วคราว (อย่ารันค้างคู่กับเบราว์เซอร์ทั่วไป)

---

## Commands ใน CLI

```
menu          เปิด mode menu
/research      เข้า skill mode research (toggle ปิด/เปิด)
/pdf_to_text   เข้า skill mode แปลง PDF → text
/history      โหลด conversation history เดิม
/compact      บีบอัด context
/clear        เริ่ม session ใหม่
/exit         ออกจาก skill mode
exit / ออก   ปิดโปรแกรม
```

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

⚠️ default ของ harness ทุกค่า (prompt, thinking budget, repetition penalty) tune สำหรับ **Qwen3.6-35B-A3B (MoE)** — เปลี่ยนโมเดลแล้วพฤติกรรม tool-calling อาจต่างไปและต้อง tune เพิ่มเอง ถ้า RAM ไม่พอสำหรับ 35B แนะนำ **Qwen3-14B** เป็นขั้นต่ำ:

```bash
export V2_MODEL="mlx-community/Qwen3-14B-4bit"
export MLX_BASE_URL="http://localhost:8081/v1"
python endeavor_agent.py
```

---

## License

MIT License + Commons Clause

ใช้ส่วนตัวและเพื่อการเรียนรู้ได้อย่างอิสระ
ห้ามนำไปใช้เชิงพาณิชย์โดยไม่ได้รับอนุญาตจากผู้พัฒนา

---

## ผู้พัฒนา

**HaloChamp**

- Website: [poomwat.com](https://www.poomwat.com)
- GitHub: [github.com/halochamp](https://github.com/halochamp)
- Email: [champoomwat@gmail.com](mailto:champoomwat@gmail.com)

มีคำถาม, แจ้งบั๊ก, หรืออยากต่อยอด — ติดต่อได้ตามช่องทางด้านบน

---

*สร้างโดย [HaloChamp](https://github.com/halochamp)*
