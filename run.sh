#!/bin/bash
# ENDEAVOR_LOCAL_AGENT_TH — shortcut launcher
# Usage: bash run.sh
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
conda activate mlx
python "$(dirname "$0")/endeavor_agent.py"
