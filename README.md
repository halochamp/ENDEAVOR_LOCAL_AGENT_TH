# ENDEAVOR_LOCAL_AGENT_TH

**Local AI Agent ที่ออกแบบให้รองรับภาษาไทยโดยเฉพาะ ที่รันบนเครื่องของคุณเอง 100%**
ไม่มี API key, ไม่มีค่า token รายเดือน, ไม่มีข้อมูลหลุดออกไปนอกเครื่อง — มี **Qwen3.5-2B-OptiQ-4bit (VLM)** เป็น lightweight test model, รองรับ **Qwen3.5-9B-4bit (VLM)** สำหรับ Mac RAM 16GB, **Qwen3-14B** เป็นค่าเริ่มต้นสำหรับเครื่อง 24GB+, และ **Qwen3.6-35B-A3B (MoE)** สำหรับคุณภาพสูง ผ่าน **MLX** บน Apple Silicon และ orchestrate ด้วย **LangGraph ReAct Agent**

เป้าหมายของโปรเจกต์คือทำให้ Local AI Agent ที่ใช้งานได้จริงเข้าถึงคนทั่วไปได้มากขึ้น: 2B มีไว้สำหรับทดลอง/diagnostic บนเครื่องทรัพยากรจำกัด, เครื่อง 16GB สามารถเลือก Qwen3.5 9B VLM, fresh install ยังเริ่มด้วย Qwen3-14B เป็นค่า default สำหรับเครื่องที่มี headroom มากกว่า และเครื่องแรงสามารถเลือก 35B MoE เพื่อคุณภาพ reasoning/tool calling สูงสุด

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
- [Tools ที่มีให้ (28 tools)](#tools-ที่มีให้-28-tools)
- [Skill Modes](#skill-modes)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration (.env)](#configuration-env)
- [ต่อ Custom Client / Telegram](#ต่อ-custom-client--telegram-agent_serverpy)
- [Commands ใน CLI](#commands-ใน-cli)
- [ต่อยอดได้ยังไง?](#ต่อยอดได้ยังไง)
- [License](#license)
- [ผู้พัฒนา](#ผู้พัฒนา)

---

## เริ่มใช้งานเร็ว (Quick Start)

**ทางลัด — ไม่อยากยุ่งกับ terminal เลย:** ดับเบิลคลิก `agent_start.command` ที่ root ของโปรเจกต์ — เปิดครั้งแรกจะติดตั้งให้อัตโนมัติทั้งหมด (conda env + AGENT_UI dependencies) แล้วเปิด desktop app ให้เลย โดยไม่ต้องเปิด MLX server เองแยกต่างหาก. `agent_server.py` + `model_runtime.py` ดูแล model server, port และ watchdog ของ Agent TH เอง; Electron เป็น UI host ไม่ใช่ process owner

มีปัญหา/อยากเริ่มใหม่สะอาดๆ → ดับเบิลคลิก `agent_stop.command` — จะหยุดเฉพาะ model server ที่ตรวจยืนยันว่า Agent TH เป็นเจ้าของและหยุด Agent/UI backend. ถ้ากำลังใช้ **Shared MAX test server** สคริปต์จะปล่อย server ของ MAX VLM ไว้ไม่แตะต้อง

หรือทำเองทีละขั้นผ่าน terminal:

```bash
# ขั้นที่ 1: ติดตั้งครั้งเดียว (สร้าง conda env "mlx" + ติดตั้งทุกอย่าง + copy .env)
bash install_library/install.sh

# ขั้นที่ 2: รัน agent — model server จะถูกดูแลให้อัตโนมัติ

# แบบ CLI: activate เอง
conda activate mlx && python endeavor_agent.py

# แบบ CLI: run.sh (ไม่ต้อง activate — สคริปต์จัดการให้)
bash run.sh

# แบบ Electron Desktop: AGENT_UI เปิด agent_server; Python backend ดูแล MLX lifecycle
cd AGENT_UI && npm install && npm start
```

ค่าเริ่มต้นคือ **Standalone** ที่ port `8085`. ถ้า `:8085` ถูก Agent MAX VLM ครอบครองด้วย patched launcher ที่ตรวจยืนยันได้ Agent TH จะเข้า **Shared MAX test server** แบบ read-only อัตโนมัติ: ใช้ model ที่ MAX โหลดอยู่ได้ แต่ไม่ Start/Stop/Reset/Watchdog หรือเปลี่ยน model ของ MAX. Listener อื่นที่ไม่ใช่ MAX จะไม่ถูก adopt หรือ kill

> การติดตั้ง **ไม่บังคับดาวน์โหลดทุกโมเดล**: fresh install ใช้ `Qwen3-14B` เป็น default และดาวน์โหลดเฉพาะโมเดลที่ถูกเปิดใช้งานจริง. ผู้ใช้สามารถเลือก `Qwen3.5-2B-OptiQ-4bit` สำหรับการทดสอบเบา ๆ, `Qwen3.5-9B-4bit` สำหรับเครื่อง 16GB หรือ `Qwen3.6-35B` สำหรับคุณภาพสูงได้ภายหลัง; 35B บน RAM <24GB จะมีคำเตือนก่อน แต่ผู้ใช้ยังยืนยันทำต่อได้

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

โปรเจกต์รองรับ **CLI + Electron Desktop** โดยทั้งสองฝั่งใช้ runtime config กลางเดียวกันสำหรับ Model, Think Budget, **Server Mode** และ **Model Server Port** (`workspace/runtime_settings.json`, override ได้ด้วย `V2_RUNTIME_SETTINGS_PATH`) จึงไม่ต้องตั้งค่าซ้ำคนละ UI. ค่าเริ่มต้นคือ `Standalone :8085`

### 1. CLI (`endeavor_agent.py`)

- รันใน terminal — เบา เร็ว เหมาะกับงานที่ต้องทำซ้ำ ๆ หรือเปิดทิ้งไว้นาน ๆ
- ระหว่าง agent ทำงาน จะเห็น **spinner 2 บรรทัด** ตลอดเวลา พร้อม context status bar แบบ real-time
- พิมพ์ `menu` → **Model / Think Budget / Port** เพื่อเลือก model, ระดับ Low 256 / Medium 512 / High 1024 / xhigh 1536 / Max 2048 และ Model Server Port จาก config กลางเดียวกับ Electron
- CLI ใช้ owner state เดียวกับ Electron; ใน `Standalone` Agent TH จะ start/reconcile/restart model server ของตัวเองได้โดยตรงและเคารพ intentional Stop
- ถ้า runtime อยู่ใน `Shared MAX` Model จะถูกล็อกตาม model ที่ MAX VLM โหลดอยู่; Think Budget ของ TH ยังเปลี่ยนได้ แต่ CLI จะไม่เปลี่ยนหรือหยุด process ของ MAX

### 2. AGENT_UI — Electron Desktop App

- โฟลเดอร์ `AGENT_UI/` — desktop app แยกหน้าต่างจริง พร้อม **Workspace**, activity, history, streaming chat, file attachment, **PDF to Text** และ Settings
- พิมพ์ `@` ในช่องข้อความเพื่อ tag ไฟล์จาก Workspace ได้โดยตรง เช่น `@world.doc สรุปไฟล์ให้หน่อย`; autocomplete รองรับไฟล์ในโฟลเดอร์ย่อยและชื่อซ้ำจะใช้ relative path เพื่อไม่เดาผิด. Backend ตรวจ canonical path ซ้ำก่อนส่ง grounding ให้ Agent จึงกัน `../`/symlink escape ได้
- tab **PDF to Text** แปลง PDF โดยตรงนอก chat graph: PDF ที่มี text layer ใช้ข้อความเดิม, หน้า scan ใช้ OCR + deterministic Thai formatting cleanup. ตัวเลือก `LLM rewrite` ปิดเป็นค่าเริ่มต้น; เมื่อเปิดจะใช้ local model/port ปัจจุบันแบบ **no-think** (`enable_thinking=false`, `thinking_budget=0`) เพื่อแก้ OCR ภาษาไทยแบบอนุรักษ์นิยม พร้อม validation/fallback กลับ OCR เดิมถ้าข้อความหรือเลขเปลี่ยนเกินขอบเขต
- **ไม่ได้ bundle Electron ไว้ในรีโป** ต้อง `npm install` เองครั้งแรก:
  ```bash
  cd AGENT_UI
  npm install     # โหลด Electron + dependencies (~150MB, ครั้งเดียว)
  npm start
  ```
- Electron เปิด authenticated `agent_server.py` เท่านั้น; **Python backend/model_runtime เป็น owner ของ MLX lifecycle** จึงทำงาน standalone ได้แม้ไม่มี Electron process manager อื่น
- แถบข้อความเหนือช่องพิมพ์แสดง `CPU · GPU · RAM · TOK · NET` แบบ real-time แทนการแสดง phase ซ้ำกับใน chat. Telemetry มาจาก Agent TH เอง ไม่พึ่ง Server Monitor/private repo และไม่ hard-code path/สเปกของเครื่องผู้พัฒนา: CPU/RAM/Network ใช้ `psutil`, GPU ตรวจ capability ของเครื่องที่รันจริง (`ioreg` บน macOS หรือ `nvidia-smi` ถ้ามี) และถ้าอ่านไม่ได้จะแสดง `GPU —` โดยไม่กระทบ Agent. `TOK` คือ token/sec แบบ rolling average 5 วินาทีจาก streaming generation callback จริงของ LLM ที่กำลังทำงาน จึงไม่ fix ชื่อ model และไม่ประมาณจากจำนวนตัวอักษร; หลัง generation จบ ค่ายังสะท้อน token ที่อยู่ในหน้าต่าง 5 วินาทีล่าสุดแล้วค่อยลดเป็น `0.0 t/s` เมื่อ window ว่าง. CPU/GPU/RAM/NET sample ทุก 5 วินาทีตามปกติ แต่ bar publish ทุก 1 วินาทีเพื่อให้ `TOK` ตอบสนองระหว่าง generation โดยไม่เพิ่ม system/GPU polling เป็น 1 Hz. Backend ส่ง event นี้เฉพาะ WebSocket ที่ประกาศ `transport=desktop` เพื่อไม่เพิ่ม unsolicited telemetry ให้ custom client ทั่วไป
- Settings มี `Server Mode`, Model, Think Budget, Model Server Port, Watchdog และปุ่ม Start / Stop / Reset. Port default คือ `8085` และ standalone port เลือกได้ `1024–65535`
- `Standalone`: TH ใช้ owner launcher ของตัวเอง, refuse foreign listener, pin `request.model` ให้ตรง owner model และ watchdog จะกู้เฉพาะ server ที่หายขณะที่ desired state=`running`
- `Shared MAX test server`: ตรวจ process signature ของ Agent MAX VLM + `/health` ก่อน attach; Model ถูกล็อกตาม MAX และ Start/Stop/Reset/Watchdog ถูกปิดทั้งหมด. Generic external listener จะไม่ถูก adopt
- ถ้าเปิด TH ขณะที่ MAX VLM ครอบครอง port ที่ TH ตั้งไว้ (default `:8085`) และตรวจยืนยันว่าเป็น MAX จริง TH จะ auto-attach read-only; ถ้า MAX ย้าย port ผู้ใช้ตั้ง Shared MAX port ตามได้
- ปิด Electron = ปิดเฉพาะ `agent_server.py`; model server standalone ยังคงอยู่ตาม durable owner intent. ใช้ Stop ใน Settings หรือ `agent_stop.command` เมื่อต้องการปิด TH model server
- Qwen3-14B เป็น text-only ใน configuration ที่ทดสอบกับโปรเจกต์นี้: `read_image` ใช้ full OCR fallback อัตโนมัติ ส่วน `Qwen3.5-2B-OptiQ-4bit`, `Qwen3.5-9B-4bit` และ Qwen3.6-35B เป็นตัวเลือก vision-capable สำหรับ direct image understanding / `computer`; 2B มีไว้สำหรับ testing/diagnostic มากกว่าคุณภาพงาน production

`agent_server.py` ยังคงเป็น authenticated WebSocket/REST **backend ของ Electron และ custom clients** แต่โปรเจกต์ไม่ bundle browser HTML UI แยกอีกต่อไป

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
│   3. เลือก tool ที่เหมาะสมจาก 28 tools         │
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

ทุก tool call ผ่าน `ToolNode` guard กลาง: ถ้าเรียก tool เดิมด้วย arguments เดิมติดกันครั้งที่สอง
ระบบจะคืน hint โดยไม่รันซ้ำ และถ้ายังเรียกซ้ำครั้งที่สามจะหยุด turn อย่าง deterministic;
`web_search` จะ normalize ตัวพิมพ์และ whitespace ของ query ก่อนเทียบซ้ำด้วย

### องค์ประกอบหลัก

| ส่วน | ทำหน้าที่ |
|---|---|
| **`react.py`** | สร้าง ReAct agent + system prompt + คำนวณ context stats |
| **`planner.py`** | LLM call เดียวจัดหมวด query เป็น simple/complex → ถ้า complex คืน step list ให้ `create_plan` ก่อนเริ่มทำงานจริง |
| **`graph.py`** | ผูก agent เข้ากับ LangGraph state machine, จัดการ retry เมื่อ synthesis ล้มเหลว, deterministic intercept สำหรับ search/research intent |
| **`llm.py`** | สร้าง `ChatOpenAI` client ชี้ไปที่ `mlx_vlm.server` (OpenAI-compatible API) |
| **`runtime_common.py`** | infra ร่วมระหว่าง CLI กับ Electron/backend — memory store, liveness check, skill detection (single source of truth ตาม Dual-Path Prohibition) |
| **`awake_engine.py`** | daemon thread คอยเช็ค standing trigger ที่ตั้งไว้ผ่าน `awake` tool (file/every/times/once) — fire แล้วส่ง query เข้า agent เองโดยไม่ต้องมีคนพิมพ์ |

### Context & Memory Management

- **Context trimming**: ตัดข้อความเก่าทิ้งเมื่อ conversation ยาวเกิน `CONTEXT_MAX_CHARS` (default 200K chars ≈ 50K tokens) — กัน context overflow บน session ยาว
- **Web cache**: ผลลัพธ์จาก `web_search` / `browse_url` / `browser_use` ถูกเก็บ raw ไว้ใน process memory แยกจาก message history — agent เห็นแค่ summary สั้น ๆ แล้วเรียก `recall_web` ถ้าต้องการเนื้อหาเต็ม
- **Persistent memory**: `remember` tool เขียนข้อเท็จจริงสำคัญลง `logs/memory.md` ข้าม session
- **Conversation history**: เก็บผ่าน LangGraph `SqliteSaver` (`logs/history.db`) — `/history` ใน CLI โหลดกลับมาได้

### Generation Tuning

| พารามิเตอร์ | ค่า default | ผลลัพธ์ |
|---|---|---|
| `TEMPERATURE` | 0.1 | คำตอบ deterministic, เหมาะกับ tool calling |
| `THINKING_BUDGET` | 1536 tokens | จำกัดเวลาที่โมเดล "คิด" ก่อนตอบ — ค่าเดียวกับ production profile |
| `REPETITION_PENALTY` | 1.05 | กัน thinking loop ซ้ำ ๆ โดยไม่กระทบ JSON ของ tool call |
| `RECURSION_LIMIT` | 60 | จำนวน step สูงสุดต่อ 1 query (รองรับ research 4 ขั้นตอน × ~12 tool calls) |
| `APC_ENABLED` | 1 | เปิด mlx_vlm Automatic Prefix Caching |
| `APC_EXACT_CACHE_ENTRIES` | 2 | เก็บ exact snapshots 2 ช่อง — เหมาะกับบทสนทนาเส้นตรงแบบ guarded prefix |
| `APC_EXACT_PREFIX_GUARD_TOKENS` | 64 | เก็บ reusable checkpoint ก่อน variable tail 64 tokens |

---

## เทคโนโลยีที่ใช้

| Layer | เทคโนโลยี | หน้าที่ |
|---|---|---|
| **LLM Runtime** | [MLX](https://github.com/ml-explore/mlx) | รัน Qwen แบบ quantized (4-bit) บน Apple Silicon GPU ผ่าน Metal |
| **Model** | `Qwen/Qwen3-14B-MLX-4bit` (default), `mlx-community/Qwen3.5-2B-OptiQ-4bit` (test VLM), `mlx-community/Qwen3.5-9B-4bit` (compact VLM), หรือ `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` (MoE) | 2B สำหรับทดสอบ, 9B สำหรับเครื่อง 16GB, 14B เป็นค่าเริ่มต้น, 35B เป็นตัวเลือกคุณภาพสูง |
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph) `create_react_agent` | ReAct loop, state graph, checkpointing |
| **LLM Client** | LangChain Core + `langchain-openai` | คุยกับ `mlx_vlm.server` ผ่าน OpenAI-compatible API |
| **Backend Server** | [FastAPI](https://fastapi.tiangolo.com/) + `uvicorn` | authenticated WebSocket/REST backend สำหรับ Electron และ custom clients |
| **Electron Desktop** | Electron + HTML/CSS/JS ใน `AGENT_UI/` | desktop chat UI, workspace/activity/history, runtime Settings |
| **CLI** | `prompt_toolkit` + `rich` | interactive terminal, autocomplete, markdown rendering, runtime Settings |
| **Web Search** | `ddgs` (DuckDuckGo) | ค้นหาข้อมูล real-time ไม่ต้องใช้ API key |
| **Web Scraping** | `trafilatura`, `requests`, Jina Reader | ดึงเนื้อหาเว็บ → clean text |
| **Browser Automation** | `playwright` + `browser-use` | ควบคุม browser จริงสำหรับเว็บ JS-heavy / SPA |
| **Data Processing** | `pandas`, `numpy` | วิเคราะห์ข้อมูล CSV/Excel |
| **Visualization** | `matplotlib` + Pillow | สร้างกราฟ พร้อมรองรับฟอนต์ไทย (Noto Sans Thai / Thonburi) |
| **Document Parsing** | `markitdown[pdf,docx,xlsx,xls]` | แปลง PDF/Word/Excel → markdown ให้ agent อ่านได้ |
| **Vision + OCR assist** | mlx-vlm + Apple Vision Framework (`pyobjc`) | โมเดล vision เห็น pixels โดยตรง; เรียก OCR/table/QR ช่วยอ่านข้อความเมื่อต้องการ — ถ้า backend เป็น text-only `read_image` จะ fallback เป็น full OCR แต่ `computer` จะถูกปิด |
| **Persistence** | SQLite (`langgraph.checkpoint.sqlite`) | เก็บ conversation history แบบ persistent |
| **Config** | `python-dotenv` | โหลด `.env` อัตโนมัติ — ปรับ config ได้โดยไม่แก้ code |
| **Sandboxing** | macOS `sandbox-exec` | จำกัด `bash` tool ให้เขียนไฟล์ได้เฉพาะใน workspace |

---

## Security

Agent ตัวนี้ออกแบบมาให้ "เขียนได้แค่ใน sandox แต่ดูเห็นได้กว้าง" — เน้นให้ agent ทำงานอัตโนมัติได้เต็มที่ โดยไม่เสี่ยงทำลายหรือหลุดข้อมูลของเครื่อง

### ข้อมูลอยู่ในเครื่องคุณเท่านั้น

โมเดล LLM รันบนเครื่องผ่าน `mlx_vlm.server` (MLX, Apple Silicon GPU) — บทสนทนา, ไฟล์, และ context ทั้งหมดที่ส่งเข้า/ออกจากโมเดล **ไม่ถูกส่งออกไปนอกเครื่อง** ไม่มี API call ไป cloud LLM provider ใด ๆ (OpenAI, Anthropic ฯลฯ) ในการทำงานปกติ

ข้อยกเว้นเดียวคือเมื่อ agent **เลือกเรียกใช้** เครื่องมือที่ต้องคุยกับอินเทอร์เน็ตตามคำสั่งของคุณเอง เช่น `web_search`, `browse_url`, `browser_use` — กรณีนี้เฉพาะ "คำค้น/URL" ที่จำเป็นเท่านั้นจะถูกส่งไปยังบริการนั้น ๆ (เช่น DuckDuckGo, เว็บปลายทาง) ไม่ใช่บทสนทนาทั้งหมด

### หลักการ

- **เขียนไฟล์ได้เฉพาะใน `workspace/`** — เครื่องมือ `write_file`, `edit`, และ `bash`/`python_exec` (เมื่อ spawn process เขียนไฟล์) ถูกจำกัดให้เขียนได้แค่ภายใต้ `workspace/` เท่านั้น พยายามเขียนไฟล์นอก workspace จะถูก block ทันที (เว้นแต่ตั้ง `V2_ALLOW_OUTSIDE=1` ซึ่งเป็น dev-only flag)
- **อ่านไฟล์ได้กว้างกว่า แต่ไม่ใช่ทุกที่** — `read_file`/`read_image`/`grep` อ่านไฟล์นอก `workspace/` ได้ (เช่นให้ agent ช่วยอ่านโค้ดในโปรเจคอื่น หรือเอกสารบนเครื่อง) แต่ path ที่เข้าข่าย "ระบบ/credentials" จะถูก block เสมอ ไม่ว่าจะตั้ง flag ใดก็ตาม
- **Path ที่ block ทั้งอ่านและเขียนเสมอ** — `/etc`, `/usr`, `/bin`, `/sbin`, `/lib`, `/System`, `/Library`, `/Applications`, `~/.ssh`, `~/.aws`, `~/.gnupg`
- **กัน symlink/`../` traversal** — ทุก path ผ่าน `os.path.realpath()` ก่อนเช็ค ป้องกัน trick เช่น สร้าง symlink ใน workspace ชี้ออกไป `~/.ssh` หรือใช้ `../../etc/passwd`

### การควบคุม `bash` / `python_exec` (process-level sandbox)

Tool ที่ spawn process จริง (`bash`, `bash_bg`, `python_exec`) ถูกครอบด้วย **macOS `sandbox-exec`** (Seatbelt) เพิ่มอีกชั้น แยกจาก path-guard ข้างบน:

- `(deny file-write*)` ครอบ `/etc`, `/usr`, `/bin`, `/sbin`, `/System`, `/Library`, `/Applications`, และโฟลเดอร์ผู้ใช้ที่สำคัญ — `~/Desktop`, `~/Documents`, `~/Downloads`, `~/Movies`, `~/Music`, `~/Pictures`, `~/Library`, `~/.ssh`, `~/.aws`, `~/.config`, `~/.gnupg`
- `(deny file-read*)` ครอบ `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.claude`, `~/.config`, และไฟล์เฉพาะที่ทุก process อ่านได้ตามสิทธิ์ระบบแต่มีความเสี่ยง — `/etc/passwd`, `/etc/group` (world-readable ตาม design ของ Unix, ไม่ใช่ credential แต่ enumerate user account ได้), `/etc/master.passwd`, `/etc/shadow`, `/etc/sudoers` — **ไม่ block ทั้งโฟลเดอร์ `/etc`** เพราะ deny ทั้งพาธจะพัง `/etc/ssl` (TLS trust store ที่ curl/git ต้องใช้) โดยไม่มีทางเปิด exception กลับมาได้ — sandbox-exec ให้ deny ชนะ allow เสมอเมื่อ path ซ้อนกัน ไม่ว่าจะเขียนก่อนหรือหลังใน profile ก็ตาม จึงต้อง block เฉพาะไฟล์ (`literal`) แทนที่จะ block ทั้งโฟลเดอร์ (`subpath`)
- `(allow file-write*)` เปิดเฉพาะ `workspace/` และ `/private/tmp`
- ทุก process มี **timeout** (`bash` default 30s, `python_exec` ปรับตาม workload) — กัน infinite loop ค้าง resource; งานที่ต้องรันนานกว่านั้นใช้ `bash_bg` แทน (register-then-poll, ไม่บล็อก)

### สรุป

| สิ่งที่ทำได้ | สิ่งที่ทำไม่ได้ |
|---|---|
| อ่านไฟล์/โค้ด/เอกสารทั่วเครื่อง (นอก system paths) | อ่าน/เขียน `~/.ssh`, `~/.aws`, `~/.gnupg`, `/etc`, `/System` ฯลฯ |
| เขียน/แก้/รันโค้ดใน `workspace/` | เขียนไฟล์นอก `workspace/` (รวม Desktop, Documents, Downloads ผ่าน `bash`) |
| รัน shell command / python script ผ่าน sandbox | escape sandbox ด้วย symlink หรือ `../` traversal |

ผลคือ agent ทำงานอัตโนมัติ (รัน loop, เรียก tool ต่อเนื่อง) ได้เต็มที่โดยไม่ต้องกังวลว่าจะไปลบ/แก้ไฟล์สำคัญของเครื่อง หรือหลุดอ่าน credentials โดยไม่ตั้งใจ

---

## Tools ที่มีให้ (28 tools)

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
| `read_file` | อ่านไฟล์ text/code รวมถึง PDF, Word, Excel — แปลงเป็น markdown อัตโนมัติ; ส่ง `path` เป็น list 2-4 ไฟล์เพื่ออ่านแบบ parallel หรือใช้ `requests` เมื่อแต่ละไฟล์มี filter/range ต่างกัน |
| `write_file` | สร้างไฟล์ใหม่ใน workspace (สำหรับไฟล์ที่ยังไม่มี) |
| `edit` | แก้ไขไฟล์ที่มีอยู่แบบ find & replace (รองรับ replace ทั้งหมด) |
| `grep` | ค้นหา regex pattern ข้ามไฟล์ในโฟลเดอร์ — คืนผลแบบ `file:line: content` |
| `workspace_ls` | แสดงรายการไฟล์ทั้งหมดใน workspace แบบ recursive tree |

### 💻 Code Execution

| Tool | คำอธิบาย |
|---|---|
| `python_exec` | รัน Python code ใน interpreter เดียวกับ agent — `pandas`, `numpy`, `matplotlib` พร้อมใช้ทันที |
| `bash` | รันคำสั่ง bash บนเครื่อง (cwd = workspace) สำหรับงาน system-level — รันใน macOS sandbox จำกัดการเขียนไฟล์นอก workspace, timeout 30s |
| `bash_bg` | รันคำสั่งที่ใช้เวลานานกว่า `bash`'s timeout แบบ background — start แล้ว poll/list/kill ทีหลังได้ ไม่บล็อก agent ระหว่างรอ (เหมาะกับ dev server, งาน build ยาว ๆ) |
| `plot` | สร้างกราฟด้วย matplotlib จาก Python code — รองรับฟอนต์ไทยเต็มรูปแบบ, บันทึกและเปิดไฟล์ให้อัตโนมัติ |

### 🖼️ Vision

| Tool | คำอธิบาย |
|---|---|
| `read_image` | progressive direct vision สำหรับไฟล์ local, URL และ `screen`; ส่ง `source` เป็น list ได้สูงสุด 10 ภาพเพื่อ resolve/encode แบบ parallel และเลือก OCR ด้วย `ocr=true` หรือ `ocr=[1,3]`; ภาพใหม่จะถูกส่งเป็น original ก่อน แล้วค่อย OCR หลัง model เห็นภาพ; text-only backend จะคืน full OCR fallback ต่อภาพ |

### 🖱️ Computer Use

| Tool | คำอธิบาย |
|---|---|
| `computer` | direct vision + guarded action บนหน้าจอจริงของเครื่อง ทีละ action — ต้องใช้ vision-capable model; จะตรวจ capability แบบไม่แตะ desktop ก่อนเริ่ม และถ้าเป็น text-only จะคืน `[unsupported]` โดยไม่ถ่าย screenshot/กด/พิมพ์/เปิด/เลื่อน และไม่ใช้ OCR แทน; ส่ง screenshot ล่าสุดให้ main VLM พร้อม `[OBS]` Accessibility/OCR ช่วยอ้างอิง; เป็น tool/state owner แยกจาก `read_image` และยังจำกัด action ต่อ turn |

### 🧠 Memory

| Tool | คำอธิบาย |
|---|---|
| `remember` | บันทึกข้อมูลสำคัญเกี่ยวกับผู้ใช้ลง `memory.md` แบบถาวร — จำได้ข้าม session |

### 📚 Knowledge Base (ต้องตั้งค่าเอง)

| Tool | คำอธิบาย |
|---|---|
| `rag_search` | ค้นหาความรู้ในฐานข้อมูลส่วนตัวของคุณเอง (BM25 + vector search) — **ไม่ได้ bundle engine หรือฐานข้อมูลมาให้** ต้อง clone [`ENDEAVOR_RAG_LITE`](https://github.com/halochamp/ENDEAVOR_RAG_LITE) (MiniLM + ChromaDB + BM25 + RRF) เป็นโฟลเดอร์พี่น้องชื่อ `ENDEAVOR_RAG_LITE` (อยู่นอกรีโปนี้ ระดับเดียวกัน — `git clone` ตรงๆ ได้ชื่อโฟลเดอร์ถูกอยู่แล้ว) — ถ้าไม่พบ engine จะตอบ `[error]` บอกวิธีตั้งค่าแทนที่จะ crash หรือเงียบ |

### 🔌 MCP (Model Context Protocol)

| Tool | คำอธิบาย |
|---|---|
| `mcp_list_tools` | ดู catalog ของ MCP server ที่ตั้งค่าไว้ หรือระบุ `tool_name` เพื่อดู description + input schema ของ tool เดียวก่อนเรียก |
| `mcp_call_tool` | เรียก tool ของ MCP server ด้วย `arguments_json` — รองรับทั้ง Streamable HTTP และ local stdio |
| `mcp_add_server` | ลงทะเบียน MCP server ระหว่างใช้งาน: HTTP ใช้ URL/headers; stdio ใช้ absolute executable + args และ `cwd` ที่อยู่ใน workspace |
| `mcp_remove_server` | ถอดเฉพาะ server ที่ลงทะเบียนผ่าน `mcp_add_server`; server ที่ developer กำหนดใน `config.MCP_SERVERS` จะไม่ถูกลบ |

MCP stdio ถูก spawn โดยไม่ผ่าน shell และอยู่ใต้ macOS sandbox เดียวกับ `bash`: เขียนได้เฉพาะ `workspace/` และ `/tmp` และยังติด sensitive-path read guards เดิม ส่วน registry อยู่ที่ `workspace/tool_mcp/servers.json` แบบ atomic และ permission `0600` เพื่อเก็บค่า config/HTTP headers ในเครื่อง โดย public repo **ไม่ bundle default MCP server ใด ๆ** มาให้

### 🔊 Audio

| Tool | คำอธิบาย |
|---|---|
| `speak` | อ่านข้อความออกเสียงผ่านลำโพงเครื่องนี้ด้วย macOS `say` — เสียงจะเล่นที่เครื่องที่รัน agent process เสมอ แม้เรียกผ่าน Telegram/remote host ก็ตาม |

### ⏰ Automation (standing triggers)

| Tool | คำอธิบาย |
|---|---|
| `awake` | ตั้ง trigger ให้ agent ทำงานเองโดยไม่ต้องมีคนพิมพ์ถาม — `file` (ไฟล์เปลี่ยน), `every` (ทุก N นาที), `times`/`run_at` (เวลาที่กำหนด), `once` (ครั้งเดียวหลัง delay), `screen` (เฝ้าจอด้วย OCR/visual signals — เห็นการเปลี่ยนแปลงและแจ้งได้ พร้อมใช้ screenshot direct vision และกดปุ่ม/action ง่ายๆ ต่อเองผ่าน `computer` ภายใต้ limit ที่เข้มกว่า) — ทำงานอยู่เบื้องหลังตราบใดที่ agent process ยังรันอยู่ |

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
- **Qwen3.5-2B-OptiQ 4-bit (lightweight test VLM):** ตัวเลือกสำหรับทดสอบ agent/runtime และเครื่องทรัพยากรจำกัด; รองรับ image input โดยตรง แต่ไม่ใช่ quality target หลักของโปรเจกต์
- **Qwen3.5-9B 4-bit (compact VLM):** ใช้งาน Agent TH ได้บน unified memory **16GB** และรองรับ image input โดยตรง
- **Qwen3-14B 4-bit (default):** unified memory ประมาณ **24GB ขั้นต่ำเชิงปฏิบัติ**, **32GB+ แนะนำ**
- **Qwen3.6-35B-A3B 4-bit (optional high-quality):** โมเดลใหญ่กว่าและใช้ RAM/swap สูงกว่า; บนเครื่อง RAM <24GB ระบบจะเตือนก่อนเริ่ม download/load แต่ผู้ใช้ยังยืนยันทำต่อได้
- ปริมาณ RAM ที่ใช้จริงขึ้นกับ context length, KV cache, tools และโปรแกรมอื่นที่เปิดพร้อมกัน ตัวเลขข้างต้นจึงเป็นแนวทางสำหรับการใช้งาน Agent ไม่ใช่เพียงการโหลด weights ให้สำเร็จ
- Python 3.11 via conda (`mlx` env)
- `mlx-vlm` installed (และติดตั้ง `mlx-lm` เป็น dependency ที่ server ใช้ร่วมกัน)
- (optional) `computer` direct screenshot vision works without extra setup; System Settings → Privacy & Security → **Accessibility** access for the process running the agent adds richer element data — the tool tells you in its own output (`ax=permission_required`) if this is off, no crash either way

> **หมายเหตุเรื่องโมเดล:** ค่า default ของโปรเจกต์ยังเป็น **Qwen3-14B-MLX-4bit**. มี **Qwen3.5-2B-OptiQ-4bit** สำหรับ testing/diagnostic และเครื่องทรัพยากรจำกัด; สำหรับ Mac unified memory **16GB** สามารถเลือก **Qwen3.5-9B-4bit** ซึ่งเป็น VLM และใช้ direct vision/computer ได้; ถ้าใช้ 14B text-only `read_image` จะ full OCR fallback และ `computer` จะ fail closed อย่างชัดเจน
>
> **Qwen3.6-35B-A3B (MoE)** ยังเลือกใช้ได้เมื่ออยากได้ headroom ด้าน reasoning/planning/tool calling มากขึ้น. ถ้าเครื่องมี RAM ต่ำกว่า 24GB Electron/CLI จะเตือนก่อน download/load เท่านั้น ไม่ได้บล็อก — ผู้ใช้ยืนยันแล้วระบบจะทำต่อ
>
> โมเดลเล็กระดับ 2B ไม่ใช่ target ที่แนะนำสำหรับงานจริงของโปรเจกต์ เพราะมีโอกาส tool-call ผิด, หลุด format หรือ reasoning ไม่พอสำหรับ workflow หลายขั้นมากขึ้น; ตัวเลือก 2B ถูกใส่ไว้โดยตั้งใจเพื่อการทดสอบและ diagnostic
>
> CLI `menu` และ Electron Settings ใช้ owner config เดียวกันสำหรับ **Model / Think Budget / Model Server Port**; เปลี่ยน Port จากฝั่งใดอีกฝั่งจะอ่านค่าต่อได้ทันที. Electron Settings ยังมี **Server Mode / Watchdog / Start / Stop / Reset** เพิ่มเติม. `Standalone` เป็นค่า default, ส่วน `Shared MAX` ใช้ server ทดสอบของ Agent MAX VLM แบบ read-only. env var `V2_MODEL` + `MLX_BASE_URL` ยังเป็น advanced override ที่ล็อก runtime (ดูหัวข้อ [Configuration](#configuration-env))

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

# 4. รัน agent — model server จะถูก Agent TH ดูแลให้อัตโนมัติ

# CLI (activate เอง)
conda activate mlx
python endeavor_agent.py

# CLI (ไม่ต้อง activate — run.sh จัดการให้)
bash run.sh

# Electron Desktop (จัดการ agent_server ให้เอง)
cd AGENT_UI
npm install   # ครั้งแรกเท่านั้น
npm start
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
| Runtime UI State | `V2_RUNTIME_SETTINGS_PATH` | optional path override สำหรับ Model/Think Budget/Server Mode/Model Server Port config กลางของ CLI + Electron |
| MLX Prefix Cache | `APC_ENABLED`, `APC_EXACT_CACHE_ENTRIES`, `APC_EXACT_PREFIX_GUARD_TOKENS` | ลด prefill/TTFT ของ prefix ที่ซ้ำกัน |
| Context Window | `V2_CONTEXT_MAX_CHARS` | ขยาย/ลด session length |
| Agent Server | `AGENT_SERVER_PORT`, `AGENT_SERVER_TOKEN`, `AGENT_AUTH_DISABLED` | ตั้งค่า authenticated backend สำหรับ Electron/custom clients |
| Workspace & Logs | `V2_WORKSPACE`, `V2_LOG_DIR`, `V2_LOG_MAX_ENTRIES` | path สำหรับไฟล์งานและ log |
| File & Vision Batch | `V2_READ_FILE_BATCH_MAX_FILES`, `V2_READ_FILE_BATCH_MAX_CHARS` | จำกัดจำนวนไฟล์และขนาดผลรวมของ parallel `read_file` batch; `read_image` จำกัด 10 ภาพต่อ batch ตาม vision turn budget |
| Web Tool Limits | `V2_WEB_SEARCH_MAX_RESULTS`, `V2_BROWSE_URL_MAX_CHARS`, ฯลฯ | จำกัดขนาดผลลัพธ์จาก web tools |
| Web Cache | `V2_WEB_CACHE_MAX_ENTRIES`, `V2_WEB_CACHE_MAX_BYTES` | จัดการ cache เนื้อหาเว็บ |
| Summarization | `V2_SUMMARY_MAX_CHARS`, `V2_SUMMARY_SKIP_LLM_BELOW` | ควบคุมการสรุปผลลัพธ์ก่อนเข้า context |

ดูรายละเอียดทั้งหมดพร้อม comment อธิบายในไฟล์ `.env.example`

---

## ต่อ Custom Client / Telegram (`agent_server.py`)

ปกติ `python endeavor_agent.py` คือ CLI ล้วน — ไม่เปิด port อะไร ส่วน Electron จะเปิด `agent_server.py` ให้เอง

ถ้าต้องการต่อ **custom client หรือ Telegram bot ของตัวเอง** สามารถรัน `agent_server.py` เป็น FastAPI backend แยก processได้:

```bash
python agent_server.py   # เปิด WebSocket + REST บน http://127.0.0.1:8765
```

| Endpoint | ใช้ทำอะไร |
|---|---|
| `ws://localhost:8765/ws` | real-time chat พร้อม token streaming (Electron/custom client) |
| `POST /chat` | sync request/response (สำหรับ Telegram bot ฯลฯ) |
| `POST /upload` | อัปโหลดไฟล์เข้า `workspace/uploads/` แล้วคืน hint text สำหรับ Electron/custom client |
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

default runtime model คือ **Qwen3-14B-MLX-4bit**. สามารถเลือก **Qwen3.5-2B-OptiQ-4bit (VLM)** สำหรับ testing/diagnostic, เครื่อง unified memory **16GB** สามารถเลือก **Qwen3.5-9B-4bit (VLM)** และถ้าต้องการคุณภาพ reasoning สูงขึ้นสามารถเลือก **Qwen3.6-35B-A3B (MoE)**; เครื่องที่มี RAM ต่ำกว่า 24GB จะได้รับคำเตือนก่อน download/load เฉพาะ 35B แต่ยังยืนยันใช้ได้. `read_image` บน 14B ใช้ OCR fallback ส่วน 2B/9B/35B รองรับ direct vision ตาม capability probe

วิธีปกติคือเลือกจาก **Electron Settings** หรือ CLI `menu` โดยใช้ `workspace/runtime_settings.json` ร่วมกัน. ใน `Standalone` Agent TH เป็น owner ของ model server และสามารถ restart model/ย้าย port ของตัวเองได้; Settings มี Watchdog + Start/Stop/Reset. ใน `Shared MAX` TH เป็น client read-only: Model ล็อกตาม MAX VLM, Think Budget ยังเป็นของ TH และ lifecycle controls ถูกปิด. เมื่อกลับจาก Shared MAX ระบบคืน **standalone model เดิมของ TH** ไม่เอา model ของ MAX มาทับค่าที่เคยเลือกไว้

สำหรับ custom backend/port ให้ใช้ advanced override ซึ่งต้องตั้งคู่กันและจะล็อก Model ใน UI:

```bash
export V2_MODEL="Qwen/Qwen3-14B-MLX-4bit"
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
