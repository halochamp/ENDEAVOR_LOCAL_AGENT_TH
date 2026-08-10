#!/bin/bash
# ENDEAVOR_LOCAL_AGENT_TH — shortcut launcher
# Usage: bash run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Do not depend on the developer machine's Homebrew-Anaconda path.  `conda`
# may be a shell function, a PATH entry, or an un-initialised Miniforge
# installation, so handle all three common cases.
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
  echo "[error] conda/Miniforge not found. Install Miniforge or add conda to PATH." >&2
  exit 1
fi

# Load conda into this non-interactive shell before activating the project env.
eval "$("$CONDA_CMD" shell.bash hook)"
conda activate mlx
cd "$SCRIPT_DIR"
exec python endeavor_agent.py
