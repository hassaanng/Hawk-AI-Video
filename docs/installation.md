# Installation Guide

Two supported paths: **bare-metal/venv** (recommended for active
development and debugging) and **Docker Compose** (recommended for a
clean, reproducible deploy once things work). Both require a real
NVIDIA GPU on the host for video generation to function — TTS and the
web layer alone will run without one, but you won't get a video out.

## Prerequisites (both paths)

| Requirement | Minimum | Check with |
|---|---|---|
| NVIDIA GPU | 12GB VRAM (LTX-Video/SVD) | `nvidia-smi` |
| NVIDIA driver | 535+ (CUDA 12.1 compatible) | `nvidia-smi` (top-right corner) |
| Disk space | 100GB+ free (weights + outputs) | `df -h` |
| OS | Ubuntu 22.04 (or compatible) | — |
| Python | 3.11 | `python3 --version` |
| Node.js | 20+ | `node --version` |
| ffmpeg | 6.x | `ffmpeg -version` |

If `nvidia-smi` fails or shows no GPU, stop here and fix that first —
nothing below will produce a video without it, regardless of how
correctly the rest of the install goes.

---

## Path A: Bare-metal / venv (recommended first)

```bash
git clone <this-project> ai-video-studio
cd ai-video-studio
bash scripts/install_local.sh
```

What `install_local.sh` actually does, step by step (read this if you
want to do it manually or it fails partway):

1. Checks `nvidia-smi` and `ffmpeg` exist; installs ffmpeg via apt if
   missing.
2. Creates `backend/.venv` and installs `requirements.txt` — this step
   downloads ~6-8GB of PyTorch + CUDA libraries and takes 5-15 minutes
   on a typical connection.
3. Downloads and installs the Piper TTS CLI binary to
   `/usr/local/bin/piper`.
4. Copies `.env.example` to `backend/.env` if it doesn't exist yet.
5. Runs `npm install` in `frontend/`.

After it finishes:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`. The status bar in the top-right should
show your real GPU name and VRAM if everything is wired correctly —
if it instead shows "NO CUDA DEVICE", see `docs/debugging.md`.

### Pre-downloading model weights (optional but recommended)

By default, weights download lazily on first use of a given backend —
meaning your *first* generation request with, say, HunyuanVideo will
sit for many minutes downloading ~30GB before any frames are computed.
To avoid that surprise:

```bash
source backend/.venv/bin/activate
python3 scripts/download_models.py --model ltx-video
# or fetch everything up front:
python3 scripts/download_models.py --all
```

Some repos (notably HunyuanVideo and the Stable Video Diffusion repos)
require you to visit the model's HuggingFace page, accept its license,
and put a HuggingFace access token in `backend/.env` as `HF_TOKEN`
before download will succeed. LTX-Video and Wan 2.1 are open-access and
need no token.

---

## Path B: Docker Compose

Requires `nvidia-container-toolkit` on the host (this is what lets a
container see the GPU at all):

```bash
# Ubuntu/Debian — one-time host setup
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify the container runtime can see your GPU before going further:
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Then:

```bash
cd ai-video-studio
cp .env.example .env   # fill in HF_TOKEN / ANTHROPIC_API_KEY if desired
cd docker
docker compose --env-file ../.env up --build
```

Backend: `http://localhost:8000`. Frontend: `http://localhost:3000`.

The first `up` will be slow — building the CUDA base image layer and
installing the full `requirements.txt` inside the container. Weights
still download lazily into the `models_cache` named volume on first
use (or pre-download by exec'ing into the container and running
`scripts/download_models.py` as in Path A).

---

## Verifying the install

Regardless of path, confirm these three things work before trusting
the system to generate anything real:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/api/system/gpu
# {"available": true, "name": "NVIDIA ...", "total_vram_gb": ..., ...}
# If "available" is false here, generation requests WILL fail —
# fix the GPU/driver/CUDA setup before going further.

curl http://localhost:8000/api/system/models
# Lists ltx-video, wan2.1, hunyuan-video, svd with their VRAM floors.
```

Next: `docs/setup.md` for configuring which model/TTS engine is
default, and running your first real generation.
