# Deployment Guide

## Local deployment

Covered in `docs/installation.md` (Path A and Path B). For a
persistent local deployment (not just `npm run dev`), build the
frontend statically and serve it behind nginx as the Docker Compose
setup does, or behind any reverse proxy of your choice pointed at the
backend's `:8000`.

Production checklist for a local deployment:
- Set `CORS_ORIGINS` in `.env` to your actual frontend origin instead
  of the dev-time wildcard.
- Run the backend under a process supervisor (systemd, supervisord) so
  it restarts on crash — note that a single video job OOM-crashing a
  worker *subprocess* does not crash the FastAPI server itself (that's
  the whole point of the `ProcessPoolExecutor` design in
  `batch_worker.py`), but the parent process itself should still be
  supervised against unrelated failures.
- Mount `models_cache/`, `outputs/`, and `uploads/` on a disk with
  real headroom — see the per-model VRAM table in `docs/setup.md` for
  weight sizes; expect 10-40GB per model backend you actually use.

## RunPod deployment

RunPod is well-suited to this app because video generation is bursty
(you want a big GPU only while actively rendering, not 24/7).

### Option 1: On-Demand Pod (simplest)

1. On RunPod, choose a GPU type matching your target backend(s):
   - LTX-Video / SVD only → RTX 4090 (24GB) or similar is plenty.
   - Wan 2.1 14B or comfortable headroom → A100 80GB.
   - HunyuanVideo full-precision → A100/H100 80GB.
2. Container image: start from
   `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` (matches
   `docker/Dockerfile.backend`), or build and push your own image from
   that Dockerfile to a registry RunPod can pull from.
3. **Attach a Network Volume** (not just container disk) and mount it
   at `/workspace`. This is the single most important RunPod-specific
   decision: container disk on a Pod is ephemeral and is wiped on
   stop; a Network Volume persists. `scripts/runpod_start.sh` is
   written assuming `/workspace` is a Network Volume — it places
   `models_cache/`, `outputs/`, `uploads/`, and the SQLite job DB
   there specifically so a stopped-and-restarted pod doesn't
   re-download 30GB of weights or lose your job history.
4. Set the container start command to:
   ```
   bash /app/scripts/runpod_start.sh
   ```
   (adjust the path to wherever your image places the repo).
5. Expose HTTP port `8000`.
6. Set environment variables in the RunPod template: at minimum
   `DEPLOYMENT_MODE=runpod`; optionally `HF_TOKEN`,
   `ANTHROPIC_API_KEY`, `DEFAULT_VIDEO_MODEL`.
7. Deploy the frontend separately (e.g. on Vercel/Netlify/any static
   host, or RunPod's own port-mapped static serve) pointed at the
   pod's public RunPod proxy URL for the backend, or simply run the
   frontend's `npm run build` output through the same pod on a second
   exposed port using nginx, mirroring `docker/Dockerfile.frontend`.

### Option 2: RunPod Serverless (for spiky/batch workloads)

For the 50-prompt batch use case specifically, Serverless can be more
cost-effective than holding an On-Demand pod running idle between
bursts. The architecture here is *compatible* with Serverless but
requires one real change: Serverless workers are stateless,
short-lived, and don't share a filesystem across invocations by
default, so:

- Point `MODELS_CACHE_DIR` at a Network Volume attached to the
  Serverless endpoint (RunPod supports this) so each cold-started
  worker doesn't re-download weights.
- The SQLite `job_store.py` would need to move to the Network Volume
  too, OR — cleaner for genuinely serverless/stateless workers — swap
  `job_store.py`'s SQLite backend for a hosted queue (Redis,
  RunPod's own queue API, or a managed Postgres). The `JobStore` class
  in `app/services/job_store.py` is intentionally the *only* file that
  knows about SQLite specifically; everything else calls its async
  methods (`create_batch`, `claim_next_queued_job`, etc.), so swapping
  its internals for a different backend is a contained change.
- Each Serverless invocation should call the same
  `app/services/pipeline.py::run_generation_pipeline` function — that
  function has no FastAPI or queue-specific assumptions baked in, it's
  pure "given these params, produce this MP4," which is exactly the
  shape a Serverless handler needs.

This project ships the On-Demand Pod path fully wired
(`scripts/runpod_start.sh`); the Serverless path is documented here as
the concrete next step rather than implemented, since it requires a
real decision about which hosted queue/storage you want to commit to.

## Multi-GPU parallelism (either local or RunPod multi-GPU instance)

By default `MAX_PARALLEL_GPU_JOBS=1`, which is *correct* for a single
GPU — running two video diffusion pipelines on one GPU at once mostly
just causes both to OOM or thrash. If your machine genuinely has
multiple GPUs:

1. Set `MAX_PARALLEL_GPU_JOBS=<number of GPUs>` in `.env`.
2. In `app/workers/batch_worker.py`, the `WorkerPool` currently pins
   all pool workers to GPU index `0` via `_worker_init`'s
   `initargs=(0,)` in `start()`. For real multi-GPU parallelism, change
   this to launch one `ProcessPoolExecutor` per GPU index (0..N-1)
   rather than one pool of N workers all pinned to GPU 0 — each
   executor's `initializer` should receive its own GPU index so
   `CUDA_VISIBLE_DEVICES` is set distinctly per pool. This is a
   deliberate, documented manual step rather than auto-detected,
   because correctly partitioning batch dispatch across N independent
   GPU-pinned pools is a real architectural decision (e.g. whether to
   round-robin jobs, or let faster GPUs claim more) you should make
   for your specific hardware rather than have silently guessed for
   you.

## Scaling the batch system beyond 50

`MAX_BATCH_SIZE=50` in config is a request-shape limit
(`BatchGenerationRequest` in `api_schemas.py` enforces `max_length=50`
via Pydantic), not a queue depth limit — the SQLite job store has no
practical limit on total queued jobs across multiple batch
submissions. Raise `MAX_BATCH_SIZE` if you genuinely want larger
single submissions; just budget wall-clock time accordingly (see the
batch timing math in `docs/setup.md` section 7).
