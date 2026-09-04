#!/bin/bash
# ENDEAVOR_LOCAL_AGENT_TH — double-click cleanup: kill everything this repo
# might have running (mlx_vlm.server, agent_server.py, AGENT_UI's Electron
# window, a stuck CLI) so agent_start.command has a clean slate to start
# from. Kills processes only — never touches workspace/, memory, or config.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Same reasoning as agent_start.command: a double-click launch has no shell
# that already sourced .env, so without this, a MLX_BASE_URL/AGENT_SERVER_PORT
# override in .env would be invisible here and this script would kill the
# wrong (default) port instead of the one actually in use.
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

echo "=== ENDEAVOR_LOCAL_AGENT_TH — เคลียร์โปรแกรมที่ค้างอยู่ ==="
echo

# Mirrors config.py's MLX_BASE_URL override (README's RAM<48GB guidance) so
# this kills whatever port the user actually configured, not just 8085.
_DEFAULT_MLX_URL="http://localhost:8085/v1"
_MLX_URL="${MLX_BASE_URL:-$_DEFAULT_MLX_URL}"
MLX_PORT="$(printf '%s' "$_MLX_URL" | sed -nE 's#^https?://[^:/]+:([0-9]+).*#\1#p')"
[ -z "$MLX_PORT" ] && MLX_PORT=8085
AGENT_PORT="${AGENT_SERVER_PORT:-8765}"

_kill_port() {
  local port="$1" label="$2" pids
  pids="$(lsof -ti:"$port" 2>/dev/null)"
  if [ -n "$pids" ]; then
    echo "[kill] $label (:$port) — pid(s): $pids"
    echo "$pids" | xargs kill -9 2>/dev/null
  else
    echo "[skip] $label (:$port) — ไม่มีอะไรรันอยู่"
  fi
}

_kill_port "$MLX_PORT" "mlx_vlm.server"
_kill_port "$AGENT_PORT" "agent_server.py"

# Anything matched below is scoped by cwd == this repo (or AGENT_UI/) —
# never a blanket name/pattern kill, which would also hit an unrelated
# python/Electron process the user has open for something else entirely.
_pid_cwd() {
  lsof -a -p "$1" -d cwd -Fn 2>/dev/null | awk '/^n/{print substr($0,2)}'
}

# A stuck `python endeavor_agent.py` CLI (doesn't bind a port, so the
# lsof-by-port kills above never reach it).
_killed_cli=0
for pid in $(pgrep -f "python.*endeavor_agent\.py" 2>/dev/null); do
  cwd="$(_pid_cwd "$pid")"
  if [ "$cwd" = "$SCRIPT_DIR" ]; then
    echo "[kill] endeavor_agent.py CLI — pid $pid"
    kill -9 "$pid" 2>/dev/null
    _killed_cli=1
  fi
done
[ "$_killed_cli" -eq 0 ] && echo "[skip] endeavor_agent.py CLI — ไม่มีอะไรรันอยู่"

# AGENT_UI's Electron window — matched by both the resolved binary path
# (this app's own node_modules/electron, not some other Electron app like
# Slack/VS Code) AND cwd, same double-scoping main.js's own
# killOtherAppInstances() uses, so this never kills an unrelated app.
_agent_ui_dir="$SCRIPT_DIR/AGENT_UI"
_electron_bin="$_agent_ui_dir/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron"
if [ -x "$_electron_bin" ]; then
  _resolved_bin="$(cd "$(dirname "$_electron_bin")" && pwd)/$(basename "$_electron_bin")"
  _killed_ui=0
  for pid in $(pgrep -f "$_resolved_bin" 2>/dev/null); do
    if [ "$(_pid_cwd "$pid")" = "$_agent_ui_dir" ]; then
      echo "[kill] AGENT_UI window — pid $pid"
      kill -9 "$pid" 2>/dev/null
      _killed_ui=1
    fi
  done
  [ "$_killed_ui" -eq 0 ] && echo "[skip] AGENT_UI window — ไม่มีอะไรรันอยู่"
else
  echo "[skip] AGENT_UI window — ยังไม่เคย npm install"
fi

echo
echo "เคลียร์เสร็จแล้ว — เปิด agent_start.command ใหม่ได้เลย"
read -r -p "กด Enter เพื่อปิดหน้าต่างนี้..." _
