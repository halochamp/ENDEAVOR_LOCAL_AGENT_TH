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
sysctl hw.memsize   # must be >= 48 GB for the default 35B model
command -v conda    # must exist — if missing, tell user to install Miniforge:
                     # https://github.com/conda-forge/miniforge
```

If any check fails, stop and explain to the user what's missing before continuing.

## 2. Install dependencies (one command, idempotent)

```bash
bash install_library/install.sh
```

This script:
- creates/reuses conda env `mlx` (Python 3.11)
- installs everything in `install_library/requirements.txt`
- installs the Playwright chromium browser
- copies `.env.example` → `.env` if `.env` doesn't exist yet

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

If it fails partway, read the error, fix the underlying issue (e.g. missing system
package), and re-run — do not skip with `--no-deps` or similar shortcuts.

## 3. Start the LLM server (must run in a separate terminal/process)

```bash
conda activate mlx
mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --port 8080
```

This is a **long-running foreground process**. Do not run it and then immediately try
to use the same terminal. If you (the AI) are starting this on the user's behalf, run
it in the background (e.g. `tmux` or `run_in_background`) and verify it's up before
proceeding:

```bash
curl -s http://localhost:8080/v1/models | head -c 200
```

Wait for a valid JSON response (model load can take 30s–2min depending on disk speed).

## 4. Run the agent

CLI (default):

```bash
conda activate mlx
cd <project_root>
python endeavor_agent.py
```

On first run it should print the banner with `N tools  ● online`. If it shows the
model as offline, the server in step 3 isn't reachable — check the port and that step 3
is still running.

Streamlit UI (if the user wants a browser-based chat interface instead of the CLI):

```bash
conda activate mlx
cd <project_root>
python agent_server.py &      # backend WebSocket/REST server
streamlit run streamlit_app.py
```

Both processes must be running — `agent_server.py` is the backend, `streamlit_app.py`
is the frontend. Same auth contract as step 6 below (`.agent_token`), but
`streamlit_app.py` reads the token itself, so the user doesn't need to do anything extra.

## 5. Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `[error] ไม่พบ conda` | Miniforge not installed | install Miniforge, restart shell |
| install.sh exits at `[1/5]` | not Apple Silicon / not macOS | this project requires M1+ Mac |
| agent says model offline | `mlx_lm.server` not running or wrong port | check step 3, confirm port 8080 |
| out of memory / swap thrashing | RAM < 48GB for 35B model | use a smaller model — see README "ใช้โมเดลอื่น" section, set `V2_MODEL` + `MLX_BASE_URL` |
| `playwright install chromium` fails | network/proxy issue | retry; required only for `browse_url`/`scrape_table`/`browser_use` tools |
| Thai text broken on plot (squares / floating vowels) | pyobjc not installed correctly | run `python -c "import Quartz, CoreText"` in the mlx env — if it fails, re-run `pip install pyobjc-framework-Quartz pyobjc-framework-CoreText` |
| Thai text OK but font looks wrong | Thonburi missing or wrong font picked | install Noto Sans Thai via `brew install --cask font-noto-sans-thai` and rebuild font cache |

## 6. Optional: own web UI / Telegram bot backend

Only if the user wants to build their **own** web UI or bot integration (separate from
the bundled Streamlit UI in step 4):

```bash
python agent_server.py
```

First run auto-generates `.agent_token` (chmod 0600). Every request needs this token —
see README.md "ต่อ Web UI / Telegram ของตัวเอง" section for the auth contract.
