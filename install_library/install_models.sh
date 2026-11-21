#!/bin/bash
# ENDEAVOR_LOCAL_AGENT_TH — optional models, downloaded here so nothing has to be fetched by hand:
#   1. OpenThai OCR  (Settings -> OCR, ~4 MB, every file checked against a pinned SHA-256)
#   2. the OmniVoice voice (Text to Speech): its own conda env "omnivoice" + ~3 GB of weights (CC-BY-NC, personal use only)
# Run by agent_start.command after the main env exists; safe to run again (every step checks first and skips what is there).
# A failure here never blocks the agent: these are optional features.
#
#   bash install_library/install_models.sh
#
# agent_start.command sets AGENT_TH_FROM_START=1: a voice install the user declined is then not asked again on every launch
# (marker: workspace/.voice_install_declined); running this script yourself always asks.
# Environment: AGENT_TH_SKIP_OCR=1 / AGENT_TH_SKIP_VOICE=1 skip a part; AGENT_TH_ASSUME_YES=1 skips the question before the
# voice download; V2_OPENTHAI_OCR_DIR, V2_TTS_OMNIVOICE_DIR, V2_TTS_OMNIVOICE_PYTHON, MLX_CONDA_ENV, CONDA_EXE as in README.

set -u
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_ENV="${MLX_CONDA_ENV:-mlx}"
VOICE_REPO="k2-fsa/OmniVoice"
VOICE_REVISION="c5fdb5ccb189668d56333f77ba2629f4cd7535f4"   # the commit the voice was tested with; a pin, never `main`

CONDA_CMD="${CONDA_CMD:-${CONDA_EXE:-}}"
[ -z "$CONDA_CMD" ] && CONDA_CMD="$(command -v conda 2>/dev/null || true)"
if [ -z "$CONDA_CMD" ]; then
  for candidate in "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" "$HOME/anaconda3/bin/conda" \
      "/opt/homebrew/Caskroom/miniforge/base/bin/conda" "/opt/homebrew/miniforge3/bin/conda" "/opt/homebrew/anaconda3/bin/conda"; do
    [ -x "$candidate" ] && { CONDA_CMD="$candidate"; break; }
  done
fi
if [ -z "$CONDA_CMD" ]; then
  echo "[models] ไม่พบ conda — ข้ามการติดตั้งโมเดลเสริม (OCR / เสียง)"
  exit 0
fi

_env_python() {   # absolute python of a conda env, or empty
  local base
  base="$("$CONDA_CMD" info --base 2>/dev/null)" || return 0
  [ -x "$base/envs/$1/bin/python" ] && echo "$base/envs/$1/bin/python"
}

# ── 1. OpenThai OCR ───────────────────────────────────────────────────────────
if [ "${AGENT_TH_SKIP_OCR:-0}" != "1" ]; then
  MLX_PY="$(_env_python "$MLX_ENV")"
  if [ -z "$MLX_PY" ]; then
    echo "[models] ไม่พบ conda env \"$MLX_ENV\" — ข้าม OCR"
  else
    if ! "$MLX_PY" -c "import pyclipper" >/dev/null 2>&1; then
      echo "[models] OCR: กำลังติดตั้ง pyclipper..."
      "$MLX_PY" -m pip install --quiet pyclipper || echo "[models] ติดตั้ง pyclipper ไม่สำเร็จ — OCR แบบ OpenThai จะยังใช้ไม่ได้"
    fi
    echo "[models] OCR (OpenThai): ตรวจไฟล์โมเดล..."
    if (cd "$PROJ_DIR" && "$MLX_PY" scripts/install_openthai_ocr.py); then
      :
    else
      echo "[models] ดาวน์โหลด OCR ไม่สำเร็จ — ใช้ Apple Vision ต่อได้; รัน bash install_library/install_models.sh ใหม่เพื่อลองอีกครั้ง"
    fi
  fi
fi

# ── 2. OmniVoice (Text to Speech) ─────────────────────────────────────────────
if [ "${AGENT_TH_SKIP_VOICE:-0}" != "1" ]; then
  VOICE_DIR="${V2_TTS_OMNIVOICE_DIR:-$HOME/Models/OmniVoice}"
  VOICE_PY="${V2_TTS_OMNIVOICE_PYTHON:-$(_env_python omnivoice)}"
  need_env=0; need_model=0
  { [ -z "$VOICE_PY" ] || [ ! -x "$VOICE_PY" ]; } && need_env=1
  { [ ! -f "$VOICE_DIR/model.safetensors" ] || [ ! -d "$VOICE_DIR/audio_tokenizer" ]; } && need_model=1

  if [ "$need_env" -eq 0 ] && [ "$need_model" -eq 0 ]; then
    echo "[models] เสียง (OmniVoice): พร้อมแล้ว ($VOICE_DIR)"
  else
    echo
    echo "[models] เสียงอ่านออกเสียง (Text to Speech) ยังไม่ครบ:"
    [ "$need_env" -eq 1 ] && echo "         - ต้องสร้าง conda env \"omnivoice\" (Python 3.11 + PyTorch, ราว 2-3 GB)"
    [ "$need_model" -eq 1 ] && echo "         - ต้องดาวน์โหลดโมเดล $VOICE_REPO (ราว 3 GB) ไป $VOICE_DIR"
    echo "         น้ำหนักโมเดลเสียงเป็น CC-BY-NC: ใช้ส่วนตัว/ไม่ใช้เชิงพาณิชย์เท่านั้น"
    answer="n"
    DECLINED_MARK="$PROJ_DIR/workspace/.voice_install_declined"
    if [ "${AGENT_TH_FROM_START:-0}" = "1" ] && [ -f "$DECLINED_MARK" ]; then
      echo "[models] เสียง (Text to Speech): ยังไม่ได้ติดตั้ง (เคยเลือกข้าม) — ติดตั้งภายหลังด้วย: bash install_library/install_models.sh"
      exit 0
    fi
    if [ "${AGENT_TH_ASSUME_YES:-0}" = "1" ]; then
      answer="y"
    elif [ -t 0 ]; then
      read -r -p "ติดตั้งเสียงตอนนี้เลยไหม? [Y/n] " answer
      [ -z "$answer" ] && answer="y"
      case "$answer" in [nN]*) mkdir -p "$PROJ_DIR/workspace" && : > "$DECLINED_MARK" ;; esac
    fi
    case "$answer" in
      [yY]*)
        if [ "$need_env" -eq 1 ]; then
          echo "[models] กำลังสร้าง env omnivoice..."
          "$CONDA_CMD" create -y -n omnivoice python=3.11 || { echo "[models] สร้าง env ไม่สำเร็จ — ข้ามเสียง"; exit 0; }
          VOICE_PY="$(_env_python omnivoice)"
          echo "[models] กำลังติดตั้งแพ็กเกจเสียง (ครั้งเดียว อาจใช้เวลาหลายนาที)..."
          "$VOICE_PY" -m pip install --quiet -r "$PROJ_DIR/install_library/requirements_voice.txt" \
            || { echo "[models] ติดตั้งแพ็กเกจเสียงไม่สำเร็จ — ข้ามเสียง"; exit 0; }
        fi
        if [ "$need_model" -eq 1 ]; then
          echo "[models] กำลังดาวน์โหลดโมเดลเสียง (~3 GB) ไป $VOICE_DIR ..."
          mkdir -p "$VOICE_DIR"
          if ! "$VOICE_PY" - "$VOICE_REPO" "$VOICE_REVISION" "$VOICE_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, revision, target = sys.argv[1:4]
snapshot_download(repo, revision=revision, local_dir=target)
PY
          then
            echo "[models] ดาวน์โหลดโมเดลเสียงไม่สำเร็จ — รัน bash install_library/install_models.sh ใหม่เพื่อต่อจากเดิม"
            exit 0
          fi
        fi
        if [ -f "$VOICE_DIR/model.safetensors" ] && [ -d "$VOICE_DIR/audio_tokenizer" ]; then
          echo "[models] เสียง (OmniVoice): ติดตั้งเสร็จแล้ว"
        else
          echo "[models] โมเดลเสียงยังไม่ครบใน $VOICE_DIR — รันใหม่เพื่อลองอีกครั้ง"
        fi
        ;;
      *)
        echo "[models] ข้ามเสียง — หน้า TTS จะบอกว่ายังใช้ไม่ได้ ติดตั้งภายหลังด้วย: bash install_library/install_models.sh"
        ;;
    esac
  fi
fi
exit 0
