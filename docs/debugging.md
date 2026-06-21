# Debugging Guide

This guide is organized by symptom. Several entries below are bugs
that were actually hit and fixed while building this project — they're
documented here because the same class of bug is likely to recur in
your environment if anything shifts (a library version, an OS).

## "GET /api/system/gpu returns available: false"

This is the first thing to check, always — nothing downstream matters
until this is fixed.

```bash
nvidia-smi
```

- **Command not found** → NVIDIA driver isn't installed on this
  machine at all. Install it before anything else.
- **Command works, shows your GPU** but the API still says
  unavailable → torch's CUDA build doesn't match your driver/CUDA
  runtime. Check:
  ```bash
  python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  ```
  If `torch.cuda.is_available()` is `False` here too, reinstall torch
  with the CUDA wheel matching your driver's max supported CUDA
  version (shown in the top-right of `nvidia-smi` output):
  ```bash
  pip uninstall torch torchvision torchaudio
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  ```
- **Inside Docker** → confirm `--gpus all` (or the Compose
  `deploy.resources.reservations.devices` block) is actually present
  and that `nvidia-container-toolkit` is installed on the *host*, not
  just inside the container. Test in isolation:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
  ```
  If this bare test fails, the problem is entirely in your Docker/GPU
  host setup, unrelated to this app.

## "InsufficientVRAMError" raised mid-batch

This is the app working as intended (`app/core/gpu.py::require_vram`)
— it is explicitly designed to fail loudly rather than let a CUDA OOM
happen mid-generation with a useless stack trace. Real fixes:

- Switch to a lighter backend (`ltx-video` or `svd` need only 12GB).
- Lower resolution/frame count in the request.
- If running other GPU processes simultaneously (another notebook, a
  second app), free that VRAM first — `nvidia-smi` shows exactly what
  else is resident.
- For HunyuanVideo specifically: confirm CPU offload actually engaged
  (check backend logs for `"enabling sequential CPU offload"`) — if
  your free VRAM was *just* above the 60GB full-precision threshold at
  load time but another process grabbed VRAM mid-run, you can get an
  OOM despite offload being available; lower `MAX_PARALLEL_GPU_JOBS`
  or restart the backend to get a clean VRAM baseline.

## A batch job hangs forever in "queued", never moves to "running"

Check the backend's own logs first — `WorkerPool started with N
worker process(es)` should appear at startup. If it's missing, the
worker pool failed to start; check for a stack trace right after that
log line in `app/main.py`'s lifespan startup.

If the pool started fine but jobs never get claimed, check
`max_parallel_gpu_jobs` (`MAX_PARALLEL_GPU_JOBS` in `.env`) — if
something external left `len(self._in_flight)` stuck at capacity (e.g.
a worker subprocess hard-crashed in a way that didn't trigger the
`add_done_callback`), restart the backend process. On startup, the app
automatically requeues any job left in `running` state by a prior
process lifetime (`job_store.requeue_stale_running_jobs()`, called
from `main.py`'s lifespan) — you'll see a log line like `"Requeued N
job(s) that were left RUNNING..."` if this fires. This exists
specifically because `RUNNING` at boot time always means the worker
that was processing it no longer exists (in-memory dispatch state
isn't persisted, only job *data* is).

## ffmpeg-related crashes

### `ValueError: flush of closed file` from `video_io.py`

This was a real bug caught during development: calling
`proc.stdin.close()` manually and then calling `proc.communicate()`
afterward causes `communicate()` to try writing to (and flushing) an
already-closed stdin pipe. **Already fixed** in the shipped
`video_io.py` — it reads `proc.stderr` directly and calls
`proc.wait()` instead of `communicate()` after manually closing stdin.
If you've modified this file and reintroduce a `communicate()` call
after a manual `stdin.close()`, you'll see this exact error again —
don't mix the two patterns.

### "ffmpeg frame export failed" with a long stderr dump

Almost always one of:
- **Resolution not divisible cleanly** — some encoders are picky about
  odd dimensions with `yuv420p` (which requires even width/height).
  `video_io.py` resizes any mismatched frame to the requested
  `(width, height)`, but if you've set a custom odd resolution (e.g.
  721×405), fix it to an even number.
- **ffmpeg not actually on PATH** inside whatever environment is
  running the worker subprocess specifically — note this can differ
  from your shell's PATH if you're running under systemd, a different
  user, or a container with a minimal PATH. Check from inside the
  exact process context:
  ```bash
  python3 -c "import shutil; print(shutil.which('ffmpeg'), shutil.which('ffprobe'))"
  ```

### Narration audio "merges" but final video has no sound

Check `get_duration_seconds()` actually returned a real duration for
both inputs — run `ffprobe` manually on both the raw video and raw
TTS output to confirm neither is a zero-byte/corrupt file before
assuming the merge logic itself is at fault. A common root cause:
the TTS engine (especially Piper, which shells out to a separate CLI
binary) failed silently — check that `piper` is actually on PATH and
its voice files downloaded successfully into
`models_cache/piper_voices/`.

## Worker pool / asyncio process doesn't exit cleanly on shutdown

This was a real bug caught during development. Two distinct causes,
both already fixed in the shipped code, documented here in case you
modify this file and reintroduce either:

1. **`multiprocessing.Manager()` created in `__init__`** rather than
   in `start()`. A `Manager()` spawns a background process the moment
   it's instantiated — if that happens at module import time (e.g. as
   a class-level default or in `__init__` of a module-level singleton)
   it spawns even when the worker pool is never started, and isn't
   cleaned up by `stop()`. Fixed by creating the `Manager` lazily
   inside `start()` and explicitly calling `self._manager.shutdown()`
   inside `stop()`.
2. **Blocking `Queue.get()` with no timeout**, awaited via
   `loop.run_in_executor(None, self._progress_queue.get)`. Cancelling
   the asyncio task wrapping this does *not* interrupt the underlying
   thread-pool thread, which stays blocked inside the C-level `.get()`
   call indefinitely — keeping the process alive past
   `asyncio.run()` returning, even though every visible `await` looks
   cancelled. Fixed by polling with a short timeout
   (`self._progress_queue.get(True, 1.0)`) and catching `queue.Empty`
   in a loop, so the executor thread regularly returns control and a
   cancellation actually takes effect within ~1 second.

If you see the backend process refuse to exit after Ctrl+C / SIGTERM,
check `app/workers/batch_worker.py`'s `_progress_loop` and `stop()`
haven't regressed to either pattern above.

## "ModuleNotFoundError: No module named 'TTS'" / 'piper' / 'diffusers'

These are intentionally **not** installed in any lightweight dev/test
environment that doesn't have a GPU — `requirements.txt` pins real,
heavy ML packages. If you're doing backend development on a
non-GPU machine (e.g. iterating on `api/routes.py` or
`job_store.py`), you don't need the GPU-dependent packages installed
at all for those layers — every model-backend file
(`ltx_video.py`, `wan_video.py`, etc.) does its `import torch` /
`import diffusers` *inside* `load()` and `generate()`, not at module
top level, specifically so the rest of the app (FastAPI routes, the
job queue, the frontend) can be developed and tested without a GPU or
the multi-GB ML dependency stack present. Only install the full
`requirements.txt` (including `torch`, `diffusers`, `TTS`) on the
machine that will actually run inference.

## Piper voice download fails (404 or connection error)

Check the voice name matches piper-voices' actual naming convention
exactly (`{lang}_{REGION}-{name}-{quality}`, e.g.
`en_US-lessac-medium`) — `tts_piper.py`'s `_ensure_voice` derives the
HuggingFace path directly from this string, so a typo'd voice name
produces a 404 rather than a helpful error. Browse exact valid names
at https://huggingface.co/rhasspy/piper-voices/tree/main before
setting a custom `tts_voice`.

## XTTS-v2 "speaker_wav" / voice cloning sounds wrong or robotic

- The reference clip should be **6-30 seconds**, single speaker, clean
  (minimal background noise/music) — XTTS-v2's cloning quality is
  quite sensitive to reference clip quality; a noisy or very short
  clip is the most common cause of poor cloned output.
- Confirm the uploaded file is actually being passed through — check
  the job's `params_json` in the SQLite DB contains a real
  `reference_audio_path` pointing at an existing file under
  `uploads/`, not a stale/deleted path.

## Frontend shows "Failed to fetch" for every action

- Confirm the backend is actually running and confirm which port —
  the Vite dev proxy (`frontend/vite.config.js`) forwards `/api/*` to
  `http://localhost:8000` by default; if you've moved the backend to a
  different port, update `vite.config.js`'s proxy target to match.
- In production (built frontend, not `npm run dev`), there is no Vite
  proxy — you need nginx (or another reverse proxy) doing this job
  instead, as configured in `docker/nginx.conf`. If you're serving the
  built frontend without that proxy in front of it, `/api/*` calls
  will 404 against whatever static file server you're using, since
  there's nothing there to forward them to the backend.

## Batch ZIP download button stays disabled forever

The "Download All" button only enables once `batch.zip_path` is
non-null, which only gets set once *every* job in the batch reaches a
terminal state (`done`, `failed`, or `canceled`) —
`_maybe_finalize_batch` in `batch_worker.py` won't build the zip while
any job is still `queued` or `running`. If the batch genuinely has all
jobs finished but the button is still disabled, check
`GET /api/batches/<id>` directly to see each job's actual status; a
single job silently stuck in `running` (e.g. from the worker-crash
scenario described above) will block zip finalization for the whole
batch indefinitely.
