#!/bin/bash
# ENDEAVOR_LOCAL_AGENT_TH — double-click launcher (macOS Finder runs this in
# Terminal automatically because of the .command extension + executable bit).
#
# Does everything needed to reach a working desktop app from a totally fresh
# clone: one-time Python/conda setup if missing, then the AGENT_UI (Electron)
# app, which auto-starts mlx_lm.server + agent_server.py itself and shuts
# them down again when the window closes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# README/.env.example document ".env" (via `cp .env.example .env`) as the
# standard way to set V2_MODEL/MLX_BASE_URL/AGENT_SERVER_PORT/etc. A
# double-click launch has no shell session to have sourced it already, so
# without this, editing .env alone would silently do nothing here even
# though config.py picks it up fine on the Python side (python-dotenv).
# set -a exports every var the sourced file assigns, so npm start's child
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

# ── 2. AGENT_UI (Electron) — installs its own node_modules once, then starts ──
cd "$SCRIPT_DIR/AGENT_UI" || { echo "[error] ไม่พบโฟลเดอร์ AGENT_UI"; _pause_and_exit; }

if ! command -v node >/dev/null 2>&1; then
  echo "[error] ไม่พบ Node.js ในเครื่อง"
  echo "ติดตั้งก่อนที่ https://nodejs.org (เลือกเวอร์ชัน LTS) แล้วเปิดไฟล์นี้ใหม่"
  _pause_and_exit
fi

if [ ! -d node_modules ]; then
  echo "[setup] เปิดครั้งแรก — กำลังติดตั้ง Electron + dependencies (~150MB ครั้งเดียว รอสักครู่)..."
  if ! npm install; then
    echo "[error] npm install ไม่สำเร็จ — เช็ค internet แล้วลองใหม่"
    _pause_and_exit
  fi
  echo "[setup] เสร็จแล้ว"
  echo
fi

echo "[start] กำลังเปิด ENDEAVOR Agent — mlx_lm.server + agent server จะเปิดให้อัตโนมัติ..."
echo
npm start

echo
echo "ปิดโปรแกรมแล้ว — กด Enter เพื่อปิดหน้าต่างนี้..."
read -r _
