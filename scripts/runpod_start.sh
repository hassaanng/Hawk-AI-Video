#!/usr/bin/env bash
# RunPod pod entrypoint / startup script.
#
# Usage as a RunPod template:
#   - Container image: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 (or your
#     own pushed image built from docker/Dockerfile.backend)
#   - Container start command: bash scripts/runpod_start.sh
#   - Expose HTTP port: 8000
#   - Attach a Network Volume mounted at /workspace so model weights and
#     outputs survive pod stop/restart (RunPod's container disk does NOT
#     persist across a stop, only a Network Volume does).
set -euo pipefail

WORKSPACE="${RUNPOD_VOLUME_PATH:-/workspace}"
REPO_DIR="$WORKSPACE/ai-video-studio"

echo "[runpod_start] Workspace: $WORKSPACE"
nvidia-smi || { echo "[runpod_start] FATAL: nvidia-smi failed — this pod has no GPU attached."; exit 1; }

if [ ! -d "$REPO_DIR" ]; then
    echo "[runpod_start] First boot on this volume — cloning project into $REPO_DIR"
    mkdir -p "$REPO_DIR"
    # In practice you'd `git clone` your own repo here, or this script
    # assumes the project was already copied onto the volume by your
    # deployment pipeline (e.g. `runpodctl send`).
    cp -r /app/* "$REPO_DIR/" 2>/dev/null || true
fi

cd "$REPO_DIR/backend"

export DEPLOYMENT_MODE=runpod
export MODELS_CACHE_DIR="$WORKSPACE/models_cache"
export OUTPUTS_DIR="$WORKSPACE/outputs"
export UPLOADS_DIR="$WORKSPACE/uploads"
export TEMP_DIR="$WORKSPACE/temp"
export HF_HOME="$WORKSPACE/models_cache/hf"
export DB_PATH="$WORKSPACE/jobs.db"

mkdir -p "$MODELS_CACHE_DIR" "$OUTPUTS_DIR" "$UPLOADS_DIR" "$TEMP_DIR"

if [ ! -d ".venv" ]; then
    echo "[runpod_start] Creating virtualenv and installing dependencies (first boot only)..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

echo "[runpod_start] GPU detected:"
python3 -c "from app.core.gpu import detect_gpu; print(detect_gpu())"

echo "[runpod_start] Starting backend on 0.0.0.0:8000 ..."
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
