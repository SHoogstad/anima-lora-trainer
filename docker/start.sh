#!/usr/bin/env bash
# First-boot setup + launcher for the Anima LoRA Trainer container.
#
# The image is OS + Intel GPU runtime only. This script builds the Python
# environment INSIDE the /workspace volume so it persists across container
# restarts and image rebuilds.
set -euo pipefail

WORKSPACE=/workspace
APP_DIR="${ANIMA_APP_DIR:-/workspace/app}"     # bind-mounted project source
VENV="$WORKSPACE/venv"
PY="$VENV/bin/python"

# PyTorch wheel index. Default = Intel XPU. Override for CUDA/CPU, e.g.
#   ANIMA_TORCH_INDEX=https://download.pytorch.org/whl/cu124
#   ANIMA_TORCH_INDEX=https://download.pytorch.org/whl/cpu
TORCH_INDEX="${ANIMA_TORCH_INDEX:-https://download.pytorch.org/whl/xpu}"

# Keep HF cache, datasets, and outputs on the volume too.
export HF_HOME="${HF_HOME:-$WORKSPACE/hf_home}"
mkdir -p "$HF_HOME" "$WORKSPACE/outputs" "$WORKSPACE/datasets" "$WORKSPACE/cache"

if [ ! -d "$VENV" ]; then
    echo "[start] creating venv at $VENV"
    python3 -m venv "$VENV"
    "$PY" -m pip install --upgrade pip wheel
fi

# Install torch (from the selected index) + the trainer. A marker file skips the
# slow reinstall on later boots; set ANIMA_REINSTALL=1 to force a refresh.
MARKER="$VENV/.anima_installed"
if [ ! -f "$MARKER" ] || [ "${ANIMA_REINSTALL:-0}" = "1" ]; then
    if [ ! -f "$APP_DIR/pyproject.toml" ]; then
        echo "[start] ERROR: no project found at $APP_DIR." >&2
        echo "[start] Bind-mount the repo there (see docker-compose.yml)." >&2
        exit 1
    fi
    echo "[start] installing PyTorch from $TORCH_INDEX"
    "$PY" -m pip install "torch>=2.8" --index-url "$TORCH_INDEX"
    echo "[start] installing anima-lora-trainer (+ WD14 tagging) from $APP_DIR"
    "$PY" -m pip install -e "$APP_DIR[tagging]"
    touch "$MARKER"
fi

echo "[start] backend report:"
"$PY" -m anima_trainer.cli --env || true

echo "[start] launching web UI on ${ANIMA_HOST:-0.0.0.0}:${ANIMA_PORT:-7860}"
exec "$PY" -m anima_trainer.webui \
    --host "${ANIMA_HOST:-0.0.0.0}" \
    --port "${ANIMA_PORT:-7860}"
