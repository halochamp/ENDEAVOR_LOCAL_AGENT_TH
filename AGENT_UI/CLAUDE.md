# CLAUDE.md — AGENT_UI (Agent TH desktop app)

Tauri 2 app: a Rust core (`src-tauri/`) hosts the page (`ui/`), which talks to the authenticated `agent_server.py` of this
repo over WebSocket/REST. It replaces the earlier Electron UI; `window.electronAPI` keeps its name and shape so the page
logic (`ui/renderer.js`, `ui/lib/*.js`) is the same code. Root rules in `../CLAUDE.md` still apply.

## Run, build, test

```bash
npm install && npm run build      # -> src-tauri/target/release/bundle/macos/Agent TH.app (~5 MB)
npm test                          # node --test: renderer/lib + the shell contract (ui/bridge.js <-> src-tauri)
cd src-tauri && cargo test        # route allowlist, voice-id check, backoff, URL encoding
```

- `agent_start.command` (repo root) builds the app when it is missing or a source file is newer, then runs it.
- Rust 1.77+, Node.js and the Xcode command line tools are needed to build; a built app needs only the repo and its conda env.
- The app starts `agent_server.py` on :8765 with a fresh per-launch token and stops it on quit. The model server is **not**
  its business: `model_runtime.py` (started by the agent) owns it, so closing the app leaves a standalone model server up.
- A rebuilt binary is new and unsigned: macOS asks again for access to the folder the repo is in (Desktop/Documents).

## Where things are

| Piece | File |
|---|---|
| agent lifecycle: find the repo + conda python, per-launch token, start/supervise `agent_server.py`, crash backoff | `src-tauri/src/agent.rs` |
| one command per `window.electronAPI` method (dialogs, uploads, job-route allowlist, edit/delete, window modes, confirm) | `src-tauri/src/commands.rs` |
| REST to the agent (`X-Auth-Token`, streamed uploads, 1 MB answer cap) | `src-tauri/src/http.rs` |
| `window.electronAPI` over Tauri `invoke`, external links | `ui/bridge.js` |
| layout: full pages (settings, model, tts, pdf, translate) vs split (workspace, focus, history, activity) | `ui/renderer.js` `FULL_PAGE_PANELS` / `syncLayout` |
| Model page + its backend (`model_slots.py` decides what is downloaded; downloads use the pinned commit) | `ui/models.js`, `tools/models.py` |

## Rules

- Keep `window.electronAPI` stable; a method added to `ui/bridge.js` needs its command in `src-tauri/src/commands.rs` and
  `lib.rs` (a test fails otherwise).
- The page gets no plugin permissions (`capabilities/default.json`: `core:default` only). New file/process powers are new
  commands with their own checks. `test/shell_contract.test.js` enforces it.
- Never call `window.confirm` / `alert`: the dialog plugin replaces them with async functions (a Promise is always truthy, so
  `if (confirm(...))` answers yes without asking). Use `askConfirm()` in `renderer.js`, which asks through the shell.
- Edit and delete are decided by the agent backend (`/edit-access/authorize`, `/edit-access/delete`); the shell never unlinks.
- Model page RAM figures (`tools/models.py` `RAM_GB`) are measured peaks, never guesses; an unmeasured model says so.
