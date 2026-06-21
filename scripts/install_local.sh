#!/usr/bin/env bash
# Local installation script for a bare-metal/venv setup (no Docker).
# Run from the project root: bash scripts/install_local.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Video generation requires an NVIDIA GPU with CUDA."
    echo "Continuing install anyway (e.g. for TTS-only or frontend-only work)."
else
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
fi

echo "==> Checking for ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found. Installing via apt (Debian/Ubuntu)..."
    sudo apt-get update && sudo apt-get install -y ffmpeg
else
    ffmpeg -version | head -1
fi

echo "==> Setting up Python virtualenv (backend/.venv)..."
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "==> Installing Python dependencies (this downloads torch + CUDA libs, can take several minutes)..."
pip install -r requirements.txt

echo "==> Installing Piper TTS CLI binary..."
if ! command -v piper &> /dev/null; then
    PIPER_VERSION="2023.11.14-2"
    wget -q "https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_x86_64.tar.gz" -O /tmp/piper.tar.gz
    sudo tar -xzf /tmp/piper.tar.gz -C /opt
    sudo ln -sf /opt/piper/piper /usr/local/bin/piper
    rm /tmp/piper.tar.gz
fi

if [ ! -f ".env" ]; then
    echo "==> Creating .env from template"
    cp ../.env.example .env
fi

cd ../frontend
echo "==> Installing frontend dependencies..."
npm install

echo ""
echo "=================================================================="
echo " Install complete."
echo ""
echo " Next steps:"
echo "   1. Edit backend/.env  — set HF_TOKEN if any model repo requires"
echo "      a gated-access HuggingFace token (LTX-Video and Wan 2.1 do"
echo "      not; HunyuanVideo and SVD's repos may require accepting a"
echo "      license on huggingface.co first)."
echo "   2. Start the backend:  cd backend && source .venv/bin/activate"
echo "        uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "   3. Start the frontend: cd frontend && npm run dev"
echo "   4. Open http://localhost:5173"
echo "=================================================================="
