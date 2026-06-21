# Setup Guide

This assumes `docs/installation.md` is done and `GET /api/system/gpu`
reports `"available": true`. This guide covers configuration choices
and walks through your first real single-clip generation and your
first batch.

## 1. Choosing a default model

Edit `backend/.env`:

```
DEFAULT_VIDEO_MODEL=ltx-video
```

Pick based on your GPU's free VRAM (`GET /api/system/gpu` ->
`free_vram_gb`), using the table below as your real-world budget. Each
backend's exact floor and tradeoffs also live in code as
`ModelRequirements.notes` (visible via `GET /api/system/models`).

| Backend | Min free VRAM | Speed | Best for |
|---|---|---|---|
| `ltx-video` | 12GB | Fastest | Iteration, most consumer GPUs |
| `svd` | 12GB | Fast | Best explicit motion control (`motion_bucket_id`); image-to-video native |
| `wan2.1` | 16GB (1.3B) / 30GB+ (auto-upgrades to 14B) | Medium | Best subject/character consistency across clips |
| `hunyuan-video` | 24GB (CPU-offloaded) / 60GB (full) | Slowest | Highest visual fidelity |

The registry (`app/models/registry.py`) enforces that only one video
model lives in VRAM at a time on single-GPU machines — switching
models between requests is normal and handled automatically; you don't
need to restart the server.

## 2. Choosing a default TTS engine

```
DEFAULT_TTS_ENGINE=xtts
```

- `xtts` (Coqui XTTS-v2): better prosody, supports **voice cloning** —
  upload a short reference clip via `POST /api/uploads/audio` and pass
  its returned path as `reference_audio_path` in a generation request
  to narrate in that voice.
- `piper`: much lighter, works on CPU, no cloning — good when your GPU
  is fully committed to video generation and you don't want TTS
  contending for VRAM. Piper voices download automatically into
  `models_cache/piper_voices/` on first use of a given voice name
  (default: `en_US-lessac-medium`). Browse other available voices at
  the [piper-voices HuggingFace repo](https://huggingface.co/rhasspy/piper-voices).

## 3. Narration generation

By default, narration text is auto-generated from the visual prompt
using a deterministic, fully offline template
(`app/services/narration.py::_fallback_narration`) that strips camera
jargon ("4k", "85mm lens", "cinematic") and reflows the remainder into
a spoken sentence.

If you set `ANTHROPIC_API_KEY` in `.env`, narration instead calls
Claude for noticeably more natural phrasing. This is optional — the
app works fully offline without it.

You can also bypass auto-generation entirely per-request by supplying
your own `narration_text` in the generation payload.

## 4. Your first single-clip generation

Via the UI (`http://localhost:5173`, "Single Clip" tab):
1. Pick a model card (defaults to your `DEFAULT_VIDEO_MODEL`).
2. Type a prompt, e.g. *"a hawk gliding over a misty canyon at dawn,
   golden hour light"*.
3. Pick a camera motion (e.g. "Dolly In") and motion strength.
4. Leave "Generate Voice Narration" checked, or uncheck for a silent
   clip.
5. Click **Render Clip**.

The right-hand "Render Queue" panel polls progress every 1.5 seconds.
A single clip typically takes 1-5 minutes on a 4090-class GPU with
LTX-Video at default settings (121 frames, 30 steps, 720p) — multiply
that several-fold for Wan 2.1's 14B variant or HunyuanVideo.

Equivalent via curl:

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a hawk gliding over a misty canyon at dawn, golden hour light",
    "model_name": "ltx-video",
    "camera_motion": "dolly_in",
    "motion_strength": 0.6,
    "enable_voiceover": true,
    "tts_engine": "xtts"
  }'
# -> {"id": "<batch_id>", "jobs": [{"id": "<job_id>", "status": "queued", ...}]}

# poll:
curl http://localhost:8000/api/jobs/<job_id>
# once status == "done":
curl -OJ http://localhost:8000/api/jobs/<job_id>/download
```

## 5. Image-to-video

Only `ltx-video`, `wan2.1`, and `svd` support this (`hunyuan-video`
does not — see its `ModelRequirements.notes`). In the UI, switch the
"GENERATION MODE" tab to "Image → Video", upload a still, then write a
prompt describing how the scene should move/evolve. The uploaded image
becomes the literal first frame the model animates from — it is not
re-drawn or stylized first.

## 6. Consistent character generation

There's no single dedicated "character ID" button, because the
techniques that actually work depend on the backend:

- **Wan 2.1** (recommended for this): has the strongest subject
  consistency of the four backends per its model card and our notes in
  `wan_video.py`. Generate your first clip of a character, then for
  subsequent clips of "the same" character, reuse the same `seed` and
  keep the descriptive portion of the prompt (appearance, clothing,
  defining features) identical while only changing the
  action/setting clause.
- **Image-to-video chaining**: generate (or supply) one strong
  reference image of your character, then use `image-to-video` mode
  for every subsequent clip with that same reference image as
  `init_image_path` — this anchors the model to consistent visual
  identity far more reliably than text-only re-description, because
  the model is conditioning on real pixels of that character rather
  than re-imagining them from words each time.

## 7. Running a batch of 50

"Batch (up to 50)" tab: paste one prompt per line, or upload a `.txt`
file with one prompt per line. Pick one model and TTS setting applied
to every clip in the batch (per-clip overrides aren't exposed in the
UI but are if you call `POST /api/generate/batch` directly with a
`prompts` array where each item can set any field independently).

The queue then runs jobs through `app/workers/batch_worker.py`'s pool.
With `MAX_PARALLEL_GPU_JOBS=1` (the correct single-GPU default), jobs
run strictly sequentially — 50 clips at ~2-4 min each is realistically
a 2-4 hour batch, not a "fire and forget in 5 minutes" operation. Plan
accordingly, and see `docs/deployment.md` for multi-GPU parallelism if
you need real throughput.

When the batch finishes (or partially finishes — some clips can fail
independently without blocking the rest), the "Download All" button
activates and serves a ZIP of every successfully completed clip.
