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
#   6. แสดงคำสั่งรันถัดไป (mlx_lm.server + python endeavor_agent.py)

set -e

ENV_NAME="mlx"
PY_VERSION="3.11"
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

echo ""
echo "=== [2/6] เช็ก conda ==="
if ! command -v conda &> /dev/null; then
    echo "[error] ไม่พบ conda — ติดตั้ง Miniforge ก่อน: https://github.com/conda-forge/miniforge"
    exit 1
fi

# โหลด conda เข้า shell ปัจจุบัน (กรณีรันผ่าน bash script ตรงๆ)
eval "$(conda shell.bash hook)"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "env '${ENV_NAME}' มีอยู่แล้ว — ใช้ของเดิม"
else
    echo "สร้าง conda env '${ENV_NAME}' (Python ${PY_VERSION})…"
    conda create -y -n "$ENV_NAME" python="$PY_VERSION"
fi
conda activate "$ENV_NAME"
echo "conda env: $(python --version) @ $(which python)"

echo ""
echo "=== [3/6] ติดตั้ง Python packages ==="
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

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
playwright install chromium

echo ""
echo "=== [6/6] เสร็จแล้ว ==="

# สร้าง .env จาก .env.example ถ้ายังไม่มี
if [ ! -f "${PROJ_DIR}/.env" ] && [ -f "${PROJ_DIR}/.env.example" ]; then
    cp "${PROJ_DIR}/.env.example" "${PROJ_DIR}/.env"
    echo ".env สร้างจาก .env.example แล้ว (แก้ได้ที่ ${PROJ_DIR}/.env)"
fi

cat <<EOF

ติดตั้งครบแล้ว ✅

ขั้นต่อไป:
  1. (optional) แก้ไขค่า config ที่ ${PROJ_DIR}/.env
     เช่น เปลี่ยน V2_MODEL หรือ AGENT_SERVER_PORT

  2. เปิด MLX server (terminal แยก):
     conda activate ${ENV_NAME}
     mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --port 8080

  3. รัน agent — เลือกแบบที่ต้องการ:

     CLI (ง่ายสุด):
       conda activate ${ENV_NAME}
       cd "${PROJ_DIR}"
       python endeavor_agent.py

     Streamlit UI (เปิด browser):
       conda activate ${ENV_NAME}
       cd "${PROJ_DIR}"
       python agent_server.py &
       streamlit run streamlit_app.py

  หมายเหตุ: รุ่น 35B ต้องการ RAM >= 48GB
  รันโมเดลเล็กกว่าได้ — ดู .env และแก้ V2_MODEL + MLX_BASE_URL
EOF
