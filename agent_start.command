#!/bin/bash
# ENDEAVOR_LOCAL_AGENT_TH — double-click launcher (macOS Finder runs this in
# Terminal automatically because of the .command extension + executable bit).
#
# Does everything needed to reach a working desktop app from a totally fresh
# clone: one-time Python/conda setup if missing, then the AGENT_UI desktop app
# (Tauri 2: built once on this Mac, then opened). agent_server.py owns the local model-server lifecycle/watchdog itself;
# The desktop app is only the UI host. A verified Agent MAX VLM test server may be
# attached read-only instead of starting a competing server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# README/.env.example document ".env" (via `cp .env.example .env`) as the
# standard way to set V2_MODEL/MLX_BASE_URL/AGENT_SERVER_PORT/etc. A
# double-click launch has no shell session to have sourced it already, so
# without this, editing .env alone would silently do nothing here even
# though config.py picks it up fine on the Python side (python-dotenv).
# set -a exports every var the sourced file assigns, so the app and its child
# processes (and their own children) actually inherit it too.
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

echo "=== ENDEAVOR_LOCAL_AGENT_TH ==="
echo

_pause_and_exit() {
  echo
  read -r -p "กด Enter เพื่อปิดหน้าต่างนี้..." _
  exit 1
}

# ── 1. One-time Python/conda setup (skip if the "mlx" env already exists) ──
CONDA_CMD="${CONDA_EXE:-}"
if [[ -z "$CONDA_CMD" ]]; then
  CONDA_CMD="$(command -v conda 2>/dev/null || true)"
fi
if [[ -z "$CONDA_CMD" ]]; then
  for candidate in \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/mambaforge/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
    "/opt/homebrew/miniforge3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      CONDA_CMD="$candidate"
      break
    fi
  done
fi

if [[ -z "$CONDA_CMD" ]]; then
  echo "[error] ไม่พบ conda/Miniforge ในเครื่อง"
  echo "ติดตั้ง Miniforge ก่อน (https://github.com/conda-forge/miniforge) แล้วเปิดไฟล์นี้ใหม่"
  _pause_and_exit
fi

eval "$("$CONDA_CMD" shell.bash hook)"

if ! conda env list | grep -qE '^\s*mlx\s'; then
  echo "[setup] เปิดครั้งแรก — ยังไม่มี conda env \"mlx\" กำลังติดตั้งให้ (ครั้งเดียว อาจใช้เวลาสักพัก)..."
  echo
  if ! bash "$SCRIPT_DIR/install_library/install.sh"; then
    echo
    echo "[error] ติดตั้งไม่สำเร็จ — ดู error ด้านบน แก้แล้วลองเปิดไฟล์นี้ใหม่"
    _pause_and_exit
  fi
  echo
  echo "[setup] ติดตั้งเสร็จแล้ว"
  echo
fi

# ── 1b. Optional models: OpenThai OCR (~4 MB) and the OmniVoice voice for Text to Speech (~3 GB, asks first) ──
# Skips whatever is already there; a failure only means that feature stays off (see install_library/install_models.sh).
CONDA_CMD="$CONDA_CMD" AGENT_TH_FROM_START=1 bash "$SCRIPT_DIR/install_library/install_models.sh"
echo

# ── 2. AGENT_UI (Tauri desktop app) — built once on this Mac, rebuilt only when its sources change ──
UI_DIR="$SCRIPT_DIR/AGENT_UI"
APP="$UI_DIR/src-tauri/target/release/bundle/macos/Agent TH.app"
APP_BIN="$APP/Contents/MacOS/agent-th-ui"
cd "$UI_DIR" || { echo "[error] ไม่พบโฟลเดอร์ AGENT_UI"; _pause_and_exit; }

# A fresh clone has no compiled app: this section builds it, which needs Node.js (build scripts), Rust (the app itself)
# and Apple's command line tools (the linker). Later launches skip all of it unless a source file is newer than the app.
_needs_build() {
  [ ! -x "$APP_BIN" ] && return 0
  [ -n "$(find src-tauri/src src-tauri/Cargo.toml src-tauri/tauri.conf.json ui tools package.json \
        -type f -not -path 'ui/vendor/*' -newer "$APP_BIN" 2>/dev/null | head -1)" ]
}

if _needs_build; then
  echo "[setup] ยังไม่มีแอป Agent TH (หรือซอร์สเปลี่ยน) — กำลัง build ในเครื่องนี้ (ครั้งแรกใช้เวลาประมาณ 3-6 นาที)..."
  if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "[error] ไม่พบ Node.js ในเครื่อง"
    echo "ติดตั้งก่อนที่ https://nodejs.org (เลือกเวอร์ชัน LTS) แล้วเปิดไฟล์นี้ใหม่"
    _pause_and_exit
  fi
  if ! xcode-select -p >/dev/null 2>&1; then
    echo "[error] ไม่พบ Xcode Command Line Tools (ตัวเชื่อมโปรแกรมที่ Rust ต้องใช้)"
    echo "รัน   xcode-select --install   แล้วรอให้ติดตั้งเสร็จ จากนั้นเปิดไฟล์นี้ใหม่"
    _pause_and_exit
  fi
  # rustup puts cargo in ~/.cargo/bin, which a Finder double-click shell does not have on PATH yet
  export PATH="$HOME/.cargo/bin:$PATH"
  if ! command -v cargo >/dev/null 2>&1; then
    echo "[setup] ไม่พบ Rust — Agent TH desktop app เขียนด้วย Rust (Tauri 2)"
    read -r -p "ติดตั้ง Rust ด้วย rustup ตอนนี้เลยไหม (~1 GB, ใส่ใน ~/.cargo และ ~/.rustup)? [y/N] " _ans
    case "$_ans" in
      [yY]*)
        if ! curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; then
          echo "[error] ติดตั้ง Rust ไม่สำเร็จ — เช็ค internet หรือติดตั้งเองที่ https://rustup.rs แล้วเปิดไฟล์นี้ใหม่"
          _pause_and_exit
        fi
        export PATH="$HOME/.cargo/bin:$PATH"
        ;;
      *)
        echo "ติดตั้ง Rust เองที่ https://rustup.rs แล้วเปิดไฟล์นี้ใหม่ (หรือใช้ CLI: bash run.sh)"
        _pause_and_exit
        ;;
    esac
  fi
  if [ ! -d node_modules ]; then
    echo "[setup] กำลังติดตั้ง build tools + ไลบรารีหน้าจอ (npm install ครั้งเดียว)..."
    if ! npm install --no-audit --no-fund; then
      echo "[error] npm install ไม่สำเร็จ — เช็ค internet แล้วลองใหม่"
      _pause_and_exit
    fi
  fi
  if ! npm run build; then
    echo "[error] build ไม่สำเร็จ — ดู error ด้านบน (ส่วนใหญ่คือ Rust/Xcode ไม่ครบ) แก้แล้วเปิดไฟล์นี้ใหม่"
    _pause_and_exit
  fi
  echo "[setup] build เสร็จแล้ว"
  echo
fi

if [ ! -x "$APP_BIN" ]; then
  echo "[error] ไม่พบแอปที่ build ไว้: $APP"
  _pause_and_exit
fi

echo "[start] กำลังเปิด ENDEAVOR Agent — Agent TH จะดูแล model server + agent server ให้อัตโนมัติ..."
echo "        (macOS อาจถามสิทธิ์เข้าถึงโฟลเดอร์ครั้งแรก — กด Allow)"
echo
# Run the binary itself (not `open`) so the app inherits this shell's environment: .env, MLX_PYTHON, MLX_CONDA_ENV.
"$APP_BIN"

echo
echo "ปิดโปรแกรมแล้ว — กด Enter เพื่อปิดหน้าต่างนี้..."
read -r _
