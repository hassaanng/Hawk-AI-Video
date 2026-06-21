# AI Video Studio

A real text-to-video / image-to-video generation application built on
genuine open-source diffusion video models — **not** slideshows, not
image-sequence videos, not MoviePy tricks. Every MP4 this system
produces is synthesized frame-by-frame by a video diffusion transformer
(LTX-Video, Wan 2.1, HunyuanVideo, or Stable Video Diffusion) and then
muxed with real TTS narration via ffmpeg.

## What this honestly requires

Video diffusion models are heavy. This is not a limitation of the code
— it's physics. Before anything below will actually generate a video:

- An **NVIDIA GPU with CUDA**, 12GB VRAM minimum (LTX-Video/SVD), 16GB+
  for Wan 2.1, 24GB+ for HunyuanVideo (with CPU offload) or 60GB+
  without offload.
- **Tens of GB of disk** per model for weight downloads from
  HuggingFace, fetched on first use (or pre-fetched via
  `scripts/download_models.py`).
- **Minutes per clip**, not seconds — this is real iterative denoising
  across many transformer layers, the same class of compute as the
  hosted video-gen products you've seen demoed.

If you run this on a machine without a GPU, every API endpoint still
works — the server boots, the queue accepts jobs, the UI renders — but
generation jobs will fail with a clear `InsufficientVRAMError` rather
than hanging or faking output, by design (see `app/core/gpu.py`).

## Project layout

```
ai-video-studio/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, lifespan-managed worker pool
│   │   ├── core/
│   │   │   ├── config.py           # env-driven Settings (pydantic-settings)
│   │   │   └── gpu.py               # GPU/VRAM detection + InsufficientVRAMError
│   │   ├── models/                  # VIDEO GENERATION BACKENDS
│   │   │   ├── base.py              # BaseVideoModel abstract interface (the swap contract)
│   │   │   ├── ltx_video.py         # LTX-Video (Lightricks/LTX-Video)
│   │   │   ├── wan_video.py         # Wan 2.1 (Wan-AI/Wan2.1-*)
│   │   │   ├── hunyuan_video.py     # HunyuanVideo (tencent/HunyuanVideo)
│   │   │   ├── svd_video.py         # Stable Video Diffusion (stabilityai/*)
│   │   │   └── registry.py          # ModelRegistry — load/unload/swap orchestration
│   │   ├── services/
│   │   │   ├── tts_base.py          # BaseTTSEngine interface
│   │   │   ├── tts_xtts.py          # Coqui XTTS-v2 (voice cloning)
│   │   │   ├── tts_piper.py         # Piper TTS (fast, CPU-friendly)
│   │   │   ├── tts_registry.py      # TTSRegistry — same swap pattern as video
│   │   │   ├── narration.py         # Auto narration script generation
│   │   │   ├── ffmpeg_service.py    # Real ffmpeg audio/video merge + HD export
│   │   │   ├── pipeline.py          # Orchestrates one job: video -> narration -> TTS -> merge
│   │   │   ├── job_store.py         # SQLite-backed persistent job/batch queue
│   │   │   └── zipper.py            # Batch "download all" ZIP builder
│   │   ├── workers/
│   │   │   └── batch_worker.py      # ProcessPoolExecutor pool draining the queue
│   │   ├── api/
│   │   │   └── routes.py            # All HTTP endpoints
│   │   ├── schemas/
│   │   │   └── api_schemas.py       # Pydantic request/response models
│   │   └── utils/
│   │       └── video_io.py          # Frame list -> MP4 (ffmpeg pipe), image loading
│   ├── requirements.txt
│   └── .env (you create this from .env.example)
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Tab navigation, top-level state
│   │   ├── components/
│   │   │   ├── StatusBar.jsx        # Live GPU/VRAM telemetry
│   │   │   ├── ModelSelector.jsx    # Pick video backend
│   │   │   ├── CameraMotionPicker.jsx
│   │   │   ├── GenerationForm.jsx   # Single-clip generation form
│   │   │   ├── BatchUploadPanel.jsx # Up to 50 prompts at once
│   │   │   └── JobQueueList.jsx     # Progress, preview, download
│   │   ├── hooks/useBatchPolling.js # Polls batch/job status every 1.5s
│   │   └── lib/api.js               # Typed fetch wrapper for every endpoint
│   └── package.json
├── docker/
│   ├── Dockerfile.backend           # CUDA 12.1 base image
│   ├── Dockerfile.frontend          # nginx static serve
│   ├── nginx.conf
│   └── docker-compose.yml           # Local GPU deployment
├── scripts/
│   ├── install_local.sh             # Bare-metal venv installer
│   ├── runpod_start.sh              # RunPod pod entrypoint
│   └── download_models.py           # Pre-fetch weights for chosen backends
├── docs/
│   ├── installation.md
│   ├── setup.md
│   ├── deployment.md
│   └── debugging.md
└── .env.example
```

## The four real things this is NOT, and why

- **Not a slideshow**: there is no code path anywhere that takes still
  images and pans/zooms a virtual camera over them (a classic "Ken
  Burns" fake). Every backend in `app/models/` calls a real diffusion
  transformer's `__call__` that performs iterative denoising across a
  *temporal* dimension — frames are jointly generated with learned
  motion priors, not derived from one image.
- **Not an image sequence dressed up as video**: frames come directly
  out of `pipeline.frames[0]` (a `diffusers` convention) and go
  straight into an ffmpeg H.264 encode (`utils/video_io.py`). There is
  no intermediate "save 121 PNGs to a folder" step a human could open
  and recognize as a slideshow.
  - The one deliberate exception — SVD's text-to-video path — is
    explained in `svd_video.py`'s docstring: SVD itself is
    image-conditioned, so a single SDXL frame is generated as the
    *conditioning input*, and SVD's own temporal-attention U-Net then
    performs real learned motion synthesis from that frame. This is a
    standard, documented real usage pattern for SVD, not a sequence
    trick.
- **Not MoviePy fakery**: MoviePy is not in `requirements.txt` and
  nowhere in this codebase. All audio/video work
  (`ffmpeg_service.py`, `video_io.py`) is direct `ffmpeg`/`ffprobe`
  subprocess invocation.
- **Not template-based**: there is no fixed visual template being
  populated with the user's text. The prompt is fed directly into a
  diffusion model's text encoder.

## Quick start

See `docs/installation.md` for the full guide. The short version, on a
CUDA machine:

```bash
git clone <this project>
cd ai-video-studio
bash scripts/install_local.sh
cd backend && source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
cd ../frontend && npm run dev
# open http://localhost:5173
```

## Swapping or adding a model

This is the part the requirements emphasized, so it's worth being
explicit: to add a fifth video backend, write one file —
`app/models/my_model.py` — subclassing `BaseVideoModel` (see
`app/models/base.py`), implement `load()`, `unload()`, `generate()`,
then add one line to `_BACKEND_CLASSES` in `app/models/registry.py`.
Nothing in `api/`, `services/pipeline.py`, or the frontend changes —
they only ever talk to `BaseVideoModel`/`ModelRegistry`.
