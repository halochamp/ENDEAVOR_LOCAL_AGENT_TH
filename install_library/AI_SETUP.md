# AI Setup Guide

This file is for AI assistants (Claude Code, Codex, Cursor, etc.) helping a user set up
**ENDEAVOR_LOCAL_AGENT_TH** on their machine. Follow these steps in order. Do not skip
the verification checks — each step gates the next.

## 0. Read first

- `../README.md` — full project overview (Thai)
- This is a **macOS Apple Silicon only** project (M1/M2/M3/M4/M5). If the user's machine
  is not `Darwin` + `arm64`, stop and tell them this project cannot run on their hardware.

> All commands below assume the current directory is the **project root**
> (the folder containing `endeavor_agent.py`), not `install_library/`.

## 1. Check prerequisites

```bash
uname -s   # must be "Darwin"
uname -m   # must be "arm64"
sysctl hw.memsize   # informational only; default is Qwen3-14B; 2B is test-only; 16GB can select Qwen3.5-9B VLM
command -v conda    # must exist — if missing, tell user to install Miniforge:
                     # https://github.com/conda-forge/miniforge
```

If the platform/Python/conda checks fail, stop and explain what's missing. RAM does not
block installation: Qwen3-14B is the default, `mlx-community/Qwen3.5-2B-OptiQ-4bit`
is available as a lightweight test/diagnostic VLM, while a 16 GB Mac can select
`mlx-community/Qwen3.5-9B-4bit` for the compact VLM path. If the user explicitly selects
Qwen3.6-35B on a Mac below 24 GB, warn that download/load may use substantial disk/RAM/swap;
continue only after the user confirms, because the warning is advisory rather than a hard block.

## 2. Install dependencies (one command, idempotent)

```bash
bash install_library/install.sh
```

This script:
- creates/reuses conda env `mlx` (Python 3.11)
- installs the hash-locked dependencies in `install_library/requirements.txt`
- attempts to install the optional Playwright Chromium browser (the agent still installs if this download fails)
- copies `.env.example` → `.env` if `.env` doesn't exist yet
- does **not** pre-download all model choices: Qwen3-14B is the default; Qwen3.5-2B test VLM, Qwen3.5-9B VLM and Qwen3.6-35B are fetched only when the user actually selects/starts them

Safe to re-run — it skips steps that are already done (existing env, satisfied pip
versions, already-downloaded chromium).

**`.env` config** — created automatically from `.env.example`. Defaults work out of the
box; the user only needs to edit it if they want to change the model, ports, or limits
(see comments in `.env` for each variable). No action needed unless the user asks.

**Thai font for the `plot` tool** — handled automatically by `install.sh` step 4:

| What | How |
|---|---|
| `pyobjc-framework-Cocoa/Quartz/CoreText` | pip (requirements.txt) |
| **Thonburi** (primary) | macOS system font — always available, no action needed |
| **Noto Sans Thai** (fallback) | `brew install --cask font-noto-sans-thai` — script runs this automatically if Homebrew is present |
| matplotlib font cache rebuild | script runs `matplotlib.font_manager._rebuild()` automatically after font check |

If the user does not have Homebrew, the script prints a warning but continues — Thonburi alone is sufficient for Thai graph rendering.

If the dependency install fails partway, read the error, fix the underlying issue
(e.g. missing system package), and re-run — do not skip with `--no-deps` or similar
shortcuts. If only the optional Chromium download fails, re-run
`python -m playwright install chromium` later when browser tools are needed.

## 3. Model server lifecycle

You normally **do not start `mlx_vlm.server` by hand anymore**. Agent TH owns its
standalone model-server lifecycle through `model_runtime.py`; the default owner port is
`:8085`, Settings can move it, and the same owner state is shared by CLI and Electron.
Start/Stop/Reset and the crash watchdog all act only on a TH-owned launcher.

There is one deliberate special case for development with Agent MAX VLM: if the selected
Shared MAX port (default `:8085`) contains a listener that can be verified as MAX VLM's
patched launcher, TH attaches as a **read-only client** and adopts the model actually
loaded by MAX. In that mode TH never Start/Stop/Reset/watchdogs or changes MAX's model.
An unrelated/generic listener is never adopted or killed.

For diagnostics, `python model_runtime.py status` is read-only and reports the current
mode, port, owner, loaded model, health, and desired state.

## 4. Run the agent

CLI (default):

```bash
conda activate mlx
cd <project_root>
python endeavor_agent.py
```

On first run it should print the banner with `N tools  ● online`. In Standalone mode,
TH reconciles its own model server first. In Shared MAX mode the MAX test server must
already be online on the selected shared port; TH will not start or repair MAX's server.

Electron Desktop (the other supported front end):

```bash
cd <project_root>/AGENT_UI
npm install   # first run only
npm start
```

Electron starts the authenticated `agent_server.py` backend for itself; Python
`model_runtime.py` owns the MLX lifecycle, not Electron. CLI and Electron share the same
Model / Think Budget / Server Mode / Port state in `workspace/runtime_settings.json`
(or the path supplied by `V2_RUNTIME_SETTINGS_PATH`). The text bar above the composer is
standalone host telemetry (`CPU · GPU · RAM · TOK · NET`) collected by Agent TH itself; it does
not require Server Monitor or a developer-machine path. CPU/RAM/network use `psutil`, GPU
is capability-detected from local platform tools when available and otherwise displays `—`.
`TOK` is the rolling five-second token/sec average from actual LLM streaming-generation callbacks;
it is model-agnostic and does not estimate from characters. After a generation ends, the recent
callbacks remain visible until they age out of the five-second window, then the meter returns to `0.0 t/s`. Telemetry events are scoped to the Electron WebSocket (`transport=desktop`)
so generic custom clients are not forced to consume desktop-only status frames. There is no bundled browser HTML UI.

## 5. Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `[error] ไม่พบ conda` | Miniforge not installed | install Miniforge, restart shell |
| install.sh exits at `[1/6]` | not Apple Silicon / not macOS | this project requires M1+ Mac |
| agent says model offline | Standalone owner failed to start, or Shared MAX server is offline/wrong port | check Settings / `python model_runtime.py status`; Shared MAX must already be running |
| out of memory / swap thrashing | selected model is too large for available memory | use Qwen3.5-2B-OptiQ-4bit for lightweight testing/diagnostic; on a 16GB Mac choose Qwen3.5-9B-4bit for normal compact use; Qwen3-14B remains the fresh default and Qwen3.6-35B remains optional |
| `playwright install chromium` fails | network/proxy issue | retry; required only for `browse_url`/`scrape_table`/`browser_use` tools |
| Thai text broken on plot (squares / floating vowels) | pyobjc not installed correctly | run `python -c "import Quartz, CoreText"` in the mlx env — if it fails, re-run `pip install pyobjc-framework-Quartz pyobjc-framework-CoreText` |
| Thai text OK but font looks wrong | Thonburi missing or wrong font picked | install Noto Sans Thai via `brew install --cask font-noto-sans-thai` and rebuild font cache |

## 6. Optional: custom client / Telegram bot backend

Only if the user wants to build an explicit custom client or bot integration:

```bash
python agent_server.py
```

First run auto-generates `.agent_token` (chmod 0600). Every request needs this token —
see README.md "ต่อ Custom Client / Telegram" section for the auth contract.

## 7. Helping a new user with day-to-day usage

Once setup is done, the user may ask the AI for help *using* the agent (not setting it
up). Key things to know:

**CLI commands** (typed inside `python endeavor_agent.py`):

| Command | What it does |
|---|---|
| `menu` | open the mode menu |
| `/research <topic>` | toggle into research skill mode (multi-step web research) |
| `/pdf_to_text <path>` | toggle into PDF → text skill mode |
| `/history` | reload previous conversation history (from `logs/history.db`) |
| `/compact` | compress/trim conversation context |
| `/clear` | start a fresh session |
| `/exit` | leave the current skill mode |
| `exit` / `ออก` | quit the program |

**Workspace** — the agent reads files from anywhere (except blocked system/credential
paths) but only **writes/creates files inside `workspace/`**. If the user asks the
agent to "save this file" or "create a script", point them to `workspace/` — that's
where outputs land. See README.md "Security" section for the full read/write model.

**Model server + runtime settings** — CLI and Electron share
`workspace/runtime_settings.json`: Model, Think Budget, Server Mode, and Port. CLI `menu`
can change Model / Think Budget / Model Server Port directly; Electron reads the same
owner file and therefore sees the CLI-selected port without a separate UI config. In
**Standalone**, Agent TH owns the verified launcher and may Start/Stop/Reset, watchdog,
switch model, or move its port itself. In **Shared MAX**, TH is a read-only client of a
verified Agent MAX VLM test server: the Model follows MAX's loaded model, Think Budget
remains TH-owned, and lifecycle/model controls cannot mutate MAX. Returning to Standalone
restores TH's previous standalone model instead of keeping MAX's shared model. The paired
`V2_MODEL` + non-default `MLX_BASE_URL` environment override remains the advanced
custom-backend path and locks runtime selection. Qwen3.5-2B-OptiQ-4bit is the lightweight
test/diagnostic VLM, Qwen3.5-9B-4bit is the compact vision-capable option for 16GB Macs;
Qwen3-14B remains the default text/tool model and uses
full-OCR fallback for `read_image`; Qwen3.6-35B remains the higher-quality vision-capable option.

**Restarting / stopping servers** — if the agent seems stuck, offline, or the user
wants a clean restart:

```bash
cd <project_root>
bash agent_stop.command
```

Then reopen CLI or Electron. In Standalone the owner runtime will reconcile the server;
in Shared MAX, start/recover MAX VLM separately because TH intentionally cannot control it.

**Where things are stored**:
- `logs/history.db` — conversation history (SQLite, via LangGraph checkpointer)
- `logs/memory.md` — facts the agent was told to `remember`
- `workspace/` — all files the agent creates/edits
- `.env` — user config (model, ports, limits)

**If the user reports a tool error** — most tool errors come back as a string starting
with `[error]` or `[BLOCKED]`. `[BLOCKED]` means the path-safety guard stopped a
read/write outside the allowed area (this is expected behavior, not a bug — explain
the Security model from README.md rather than trying to bypass it).

## 8. Quick reference / help cheat sheet

If the user just opens this file and asks "help" / "how do I use this" /
"ใช้งานยังไง" without a specific setup problem, use this cheat sheet to answer fast
instead of re-reading the whole README:

| User asks | Answer |
|---|---|
| "ใช้งานยังไง" / how do I start | Open CLI or Electron; TH manages its Standalone model server automatically. Shared MAX requires MAX VLM's test server to already be running |
| "model offline" / agent ขึ้น offline | Check Settings or `python model_runtime.py status`; in Shared MAX, recover MAX VLM separately |
| "เปลี่ยนโมเดล/port" / change model or port | ใช้ Electron Settings หรือ CLI `menu` → Model / Think Budget / Port; ทั้งคู่เขียน owner config เดียวกัน. Standalone เปลี่ยน model/port ได้เอง, Shared MAX ล็อก model ตาม MAX และ CLI เปลี่ยนได้เฉพาะ shared test-server port หลัง verify MAX server. Qwen3.5-2B เป็น test VLM, Qwen3.5-9B เป็น compact VLM, Qwen3-14B เป็น default, Qwen3.6-35B เป็น high-quality VLM |
| "port ถูกใช้อยู่" / port in use | Choose another Standalone port, or stop only the TH-owned server. A foreign/MAX listener is never killed automatically |
| "เซฟไฟล์ไว้ไหน" / where are my files | `workspace/` — agent can only write there |
| "ลืม conversation เก่า" / load old chat | `/history` in CLI, or use History in Electron (loads from `logs/history.db`) |
| "ปลอดภัยไหม" / is my data safe | Yes — model runs 100% locally via MLX, no cloud LLM calls. See README "Security" |
| "[BLOCKED] ..." error | Expected — path guard blocked read/write outside `workspace/` or to a protected system path. Not a bug |
| "ทำ tool/skill ใหม่ยังไง" / add a tool | README "ต่อยอดได้ยังไง?" section |
| "ต่อ client ของตัวเอง" / build own client | README "ต่อ Custom Client / Telegram" + step 6 above (`.agent_token` auth) |

For anything not covered here, read `../README.md` — it has full detail on tools,
architecture, config, and security model.
