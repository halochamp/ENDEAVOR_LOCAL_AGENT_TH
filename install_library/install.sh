#!/bin/bash
# install.sh — ติดตั้ง dependencies ทั้งหมดของ ENDEAVOR_LOCAL_AGENT_TH
#
# ใช้:
#   bash install_library/install.sh
#
# ทำ:
#   1. ตรวจ macOS + Apple Silicon (mlx ใช้ได้เฉพาะ Apple Silicon)
#   2. สร้าง/activate conda env ชื่อ "mlx" (Python 3.11) ถ้ายังไม่มี
#   3. pip install -r install_library/requirements.txt
#   4. ตรวจและติดตั้ง Thai font สำหรับกราฟ (Noto Sans Thai ผ่าน Homebrew)
#   5. playwright install chromium (สำหรับ scrape_table / browser_use)
#   6. แสดงคำสั่งรันถัดไป (mlx_vlm.server + python endeavor_agent.py)

set -euo pipefail

ENV_NAME="mlx"
PY_VERSION="3.11"
DEFAULT_MODEL="unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
MIN_DEFAULT_MODEL_RAM_BYTES=$((48 * 1024 * 1024 * 1024))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== [1/6] ตรวจระบบ ==="
if [ "$(uname -s)" != "Darwin" ]; then
    echo "[error] ต้องใช้ macOS เท่านั้น (mlx ใช้ Metal/Apple Silicon)"
    exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
    echo "[error] ต้องใช้ Apple Silicon (M1/M2/M3/M4/M5) — เครื่องนี้คือ $(uname -m)"
    exit 1
fi
echo "macOS Apple Silicon — OK"

# The production 35B model is not a practical default on a small Mac.  Allow
# an explicit V2_MODEL override (in the shell or an existing .env) so users
# can install a smaller model deliberately, but fail early instead of
# downloading packages and then swapping/OOM-loading the default model.
RAM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || true)"
MODEL_OVERRIDE="${V2_MODEL:-}"
if [[ -z "$MODEL_OVERRIDE" && -f "$PROJ_DIR/.env" ]] && \
   grep -Eq '^[[:space:]]*V2_MODEL[[:space:]]*=' "$PROJ_DIR/.env"; then
    MODEL_OVERRIDE="configured"
fi
if [[ -z "$MODEL_OVERRIDE" && "$RAM_BYTES" =~ ^[0-9]+$ ]] && \
   (( RAM_BYTES < MIN_DEFAULT_MODEL_RAM_BYTES )); then
    RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
    echo "[error] โมเดลเริ่มต้น ${DEFAULT_MODEL} ต้องการ RAM อย่างน้อย 48GB; เครื่องนี้มีประมาณ ${RAM_GB}GB"
    echo "        ตั้งค่า V2_MODEL และ MLX_BASE_URL ให้เป็นรุ่นเล็กกว่า แล้วรัน installer ใหม่ เช่น:"
    echo "        export V2_MODEL=mlx-community/Qwen3-1.7B-4bit MLX_BASE_URL=http://localhost:8888/v1"
    exit 1
fi

echo ""
echo "=== [2/6] เช็ก conda ==="
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
    echo "[error] ไม่พบ conda — ติดตั้ง Miniforge ก่อน: https://github.com/conda-forge/miniforge"
    exit 1
fi

# โหลด conda เข้า shell ปัจจุบัน (กรณีรันผ่าน bash script ตรงๆ)
eval "$("$CONDA_CMD" shell.bash hook)"

if conda env list | grep -qE "^[[:space:]]*${ENV_NAME}[[:space:]]"; then
    echo "env '${ENV_NAME}' มีอยู่แล้ว — ใช้ของเดิม"
else
    echo "สร้าง conda env '${ENV_NAME}' (Python ${PY_VERSION})…"
    conda create -y -n "$ENV_NAME" python="$PY_VERSION"
fi
conda activate "$ENV_NAME"
echo "conda env: $(python --version) @ $(which python)"
if ! python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
then
    echo "[error] env '${ENV_NAME}' ไม่ได้ใช้ Python 3.11"
    exit 1
fi

echo ""
echo "=== [3/6] ติดตั้ง Python packages ==="
python -m pip install --upgrade pip
python -m pip install --require-hashes -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=== [4/6] ตรวจและติดตั้ง Thai font สำหรับกราฟ ==="

# Thonburi — system font ของ macOS พร้อมใช้เสมอ
echo "Thonburi (macOS system font) — พร้อมใช้งาน ✅"

# Noto Sans Thai — ตรวจก่อน ถ้าไม่มีให้ลงผ่าน Homebrew
NOTO_FOUND=false
for font_dir in "$HOME/Library/Fonts" "/Library/Fonts" "/System/Library/Fonts"; do
    if find "$font_dir" -iname "*NotoSansThai*" 2>/dev/null | grep -q .; then
        NOTO_FOUND=true
        break
    fi
done

if $NOTO_FOUND; then
    echo "Noto Sans Thai — พบแล้ว ✅"
else
    echo "Noto Sans Thai — ไม่พบ กำลังติดตั้ง..."
    if command -v brew &>/dev/null; then
        if brew install --cask font-noto-sans-thai 2>/dev/null; then
            echo "Noto Sans Thai — ติดตั้งสำเร็จ ✅"
        else
            echo "[warning] ติดตั้ง Noto Sans Thai ไม่สำเร็จ — Thonburi จะถูกใช้แทน (กราฟยังทำงานได้ปกติ)"
        fi
    else
        echo "[warning] ไม่พบ Homebrew — ข้าม Noto Sans Thai"
        echo "          ติดตั้ง Homebrew: https://brew.sh แล้วรัน:"
        echo "          brew install --cask font-noto-sans-thai"
    fi
fi

# rebuild matplotlib font cache เสมอ (ทั้ง Noto ใหม่และ Thonburi ที่อาจยังไม่ scan)
echo "กำลัง rebuild matplotlib font cache…"
python -c "import matplotlib.font_manager; matplotlib.font_manager._rebuild()" 2>/dev/null \
    && echo "matplotlib font cache — OK ✅" \
    || echo "[warning] rebuild font cache ไม่สำเร็จ (ไม่กระทบการทำงานหลัก)"

echo ""
echo "=== [5/6] ติดตั้ง Playwright browser (chromium) ==="
if ! python -m playwright install chromium; then
    echo "[warning] ติดตั้ง Chromium ไม่สำเร็จ — browser tools จะใช้ไม่ได้จนกว่าจะรัน:"
    echo "          conda activate ${ENV_NAME} && python -m playwright install chromium"
fi

echo ""
echo "=== [6/6] เสร็จแล้ว ==="

# สร้าง .env จาก .env.example ถ้ายังไม่มี
if [ ! -f "${PROJ_DIR}/.env" ] && [ -f "${PROJ_DIR}/.env.example" ]; then
    cp "${PROJ_DIR}/.env.example" "${PROJ_DIR}/.env"
    echo ".env สร้างจาก .env.example แล้ว (แก้ได้ที่ ${PROJ_DIR}/.env)"
fi

# Print the effective model/backend rather than a stale developer default.
# The values are read after .env creation so the next-step command matches the
# same precedence rules used by config.py.
EFFECTIVE_CONFIG="$(cd "$PROJ_DIR" && python - <<'PY'
from urllib.parse import urlparse
import config

url = urlparse(config.MLX_BASE_URL)
print(config.MODEL, url.hostname or "127.0.0.1", url.port or 8085, sep="\t")
PY
)"
IFS=$'\t' read -r EFFECTIVE_MODEL EFFECTIVE_HOST EFFECTIVE_PORT <<< "$EFFECTIVE_CONFIG"

cat <<EOF

ติดตั้งครบแล้ว ✅

ขั้นต่อไป:
  1. (optional) แก้ไขค่า config ที่ ${PROJ_DIR}/.env
     เช่น เปลี่ยน V2_MODEL หรือ AGENT_SERVER_PORT

  2. เปิด MLX server (terminal แยก):
     conda activate ${ENV_NAME}
     APC_ENABLED=1 APC_EXACT_CACHE_ENTRIES=2 APC_EXACT_PREFIX_GUARD_TOKENS=64 \\
     python -m mlx_vlm.server --model ${EFFECTIVE_MODEL} --host ${EFFECTIVE_HOST} --port ${EFFECTIVE_PORT}

  3. รัน agent — เลือกแบบที่ต้องการ:

     CLI (ง่ายสุด):
       conda activate ${ENV_NAME}
       cd "${PROJ_DIR}"
       python endeavor_agent.py

     Web UI (เปิด browser ที่ http://localhost:8765/ui):
       conda activate ${ENV_NAME}
       cd "${PROJ_DIR}"
       python agent_server.py

  หมายเหตุ: รุ่น 35B ต้องการ RAM >= 48GB
  รันโมเดลเล็กกว่าได้ — ดู .env และแก้ V2_MODEL + MLX_BASE_URL
EOF
