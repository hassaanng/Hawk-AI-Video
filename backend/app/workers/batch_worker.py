"""
Background worker.

Video generation is a long (seconds to many minutes), CPU/GPU-bound,
blocking operation — it must never run on FastAPI's asyncio event loop
or every other request (including progress polling for OTHER jobs)
would stall behind it. This module runs the actual generation in a
`ProcessPoolExecutor` worker, while a lightweight asyncio task polls
the SQLite queue and dispatches work.

Why a process pool rather than a thread pool: PyTorch/CUDA contexts
and diffusers pipelines are not safely shared across threads in the
same process for this kind of long-running stateful inference, and a
crash in one generation (OOM, corrupt weights, etc.) must not be able
to take down the FastAPI server process itself. A dedicated subprocess
also lets the model registry's "only one model in VRAM" policy hold
cleanly without coordinating with the API process's own imports.

For multi-GPU machines, set MAX_PARALLEL_GPU_JOBS = number of GPUs;
each pool worker pins itself to one GPU via CUDA_VISIBLE_DEVICES,
giving real parallel batch throughput rather than just concurrency.
"""
from __future__ import annotations

import asyncio
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

from app.core.config import settings
from app.services.job_store import BatchStatus, JobStatus, job_store

logger = logging.getLogger(__name__)


def _worker_init(gpu_index: int) -> None:
    """Run once per pool process at startup: pin this process to one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def _run_job_in_subprocess(job: dict, progress_queue) -> dict:
    """
    Entry point executed inside the ProcessPoolExecutor worker. Imports
    are deliberately local to this function: importing torch/diffusers
    at module level would load CUDA in the FastAPI parent process too,
    which we don't want.
    """
    import json

    from app.services.pipeline import run_generation_pipeline

    params = json.loads(job["params_json"])

    def on_progress(frac: float, msg: str) -> None:
        progress_queue.put({"job_id": job["id"], "progress": frac, "message": msg})

    try:
        final_path = run_generation_pipeline(
            prompt=params["prompt"],
            model_name=params.get("model_name", settings.default_video_model),
            negative_prompt=params.get("negative_prompt"),
            mode=params.get("mode", "text-to-video"),
            init_image_path=params.get("init_image_path"),
            width=params.get("width", settings.default_width),
            height=params.get("height", settings.default_height),
            num_frames=params.get("num_frames", settings.default_num_frames),
            fps=params.get("fps", settings.default_fps),
            num_inference_steps=params.get("num_inference_steps", 30),
            guidance_scale=params.get("guidance_scale", 7.0),
            seed=params.get("seed"),
            camera_motion=params.get("camera_motion", "static"),
            motion_strength=params.get("motion_strength", 1.0),
            enable_voiceover=params.get("enable_voiceover", True),
            narration_text=params.get("narration_text"),
            tts_engine_name=params.get("tts_engine", settings.default_tts_engine),
            tts_voice=params.get("tts_voice"),
            reference_audio_path=params.get("reference_audio_path"),
            on_progress=on_progress,
        )
        return {"ok": True, "job_id": job["id"], "output_path": str(final_path)}
    except Exception as exc:  # noqa: BLE001 - we want to surface ANY failure to the job record
        tb = traceback.format_exc()
        logger.error("Job %s failed: %s\n%s", job["id"], exc, tb)
        return {"ok": False, "job_id": job["id"], "error": f"{exc}\n{tb[-2000:]}"}


class WorkerPool:
    def __init__(self, num_workers: int | None = None) -> None:
        self.num_workers = num_workers or settings.max_parallel_gpu_jobs
        self._manager: Manager | None = None
        self._progress_queue = None
        self._executor: ProcessPoolExecutor | None = None
        self._poll_task: asyncio.Task | None = None
        self._progress_task: asyncio.Task | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._in_flight: dict[str, asyncio.Future] = {}
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        # Manager (and its background process) is created lazily here
        # rather than in __init__, so importing this module never spawns
        # a subprocess as a side effect — only actually starting the pool
        # does.
        self._manager = Manager()
        self._progress_queue = self._manager.Queue()
        self._executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=(0,),  # single-GPU default; multi-GPU launches one pool per GPU index in main.py
        )
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        self._progress_task = asyncio.create_task(self._progress_loop())
        logger.info("WorkerPool started with %d worker process(es).", self.num_workers)

    async def stop(self) -> None:
        self._running = False
        for task in (self._dispatch_task, self._progress_task):
            if task:
                task.cancel()
        for task in (self._dispatch_task, self._progress_task):
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if self._manager:
            self._manager.shutdown()
            self._manager = None
        self._progress_queue = None

    async def _dispatch_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                if len(self._in_flight) < self.num_workers:
                    job = await job_store.claim_next_queued_job()
                    if job:
                        future = loop.run_in_executor(
                            self._executor, _run_job_in_subprocess, job, self._progress_queue
                        )
                        self._in_flight[job["id"]] = future
                        future.add_done_callback(
                            lambda f, jid=job["id"]: asyncio.create_task(self._on_job_done(jid, f))
                        )
                        await self._maybe_update_batch_running(job["batch_id"])
                        continue  # try to claim more immediately if capacity remains
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Dispatch loop error")
                await asyncio.sleep(2.0)

    async def _progress_loop(self) -> None:
        import queue as _queue_mod

        loop = asyncio.get_event_loop()
        while self._running:
            try:
                # Manager().Queue().get() blocks the calling thread
                # indefinitely with no native async support. Using a
                # short timeout (rather than a bare blocking .get()) means
                # the executor thread returns control regularly, so
                # cancelling this task during shutdown doesn't leave a
                # thread-pool thread permanently blocked keeping the
                # process alive.
                item = await loop.run_in_executor(None, self._progress_queue.get, True, 1.0)
                await job_store.update_job_progress(item["job_id"], item["progress"], item["message"])
            except _queue_mod.Empty:
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Progress loop error")

    async def _on_job_done(self, job_id: str, future: asyncio.Future) -> None:
        self._in_flight.pop(job_id, None)
        try:
            result = future.result()
        except Exception as exc:  # process crashed, OOM-killed, etc.
            await job_store.fail_job(job_id, f"Worker process error: {exc}")
        else:
            if result["ok"]:
                await job_store.complete_job(job_id, result["output_path"])
            else:
                await job_store.fail_job(job_id, result["error"])

        job = await job_store.get_job(job_id)
        if job:
            await self._maybe_finalize_batch(job["batch_id"])

    async def _maybe_update_batch_running(self, batch_id: str) -> None:
        batch = await job_store.get_batch(batch_id)
        if batch and batch["status"] == BatchStatus.QUEUED.value:
            await job_store.update_batch_status(batch_id, BatchStatus.RUNNING)

    async def _maybe_finalize_batch(self, batch_id: str) -> None:
        jobs = await job_store.get_jobs_for_batch(batch_id)
        statuses = {j["status"] for j in jobs}
        if statuses <= {JobStatus.DONE.value}:
            await self._zip_and_finalize(batch_id, BatchStatus.DONE)
        elif statuses <= {JobStatus.DONE.value, JobStatus.FAILED.value} and not (
            JobStatus.QUEUED.value in statuses or JobStatus.RUNNING.value in statuses
        ):
            status = BatchStatus.PARTIAL if JobStatus.DONE.value in statuses else BatchStatus.FAILED
            await self._zip_and_finalize(batch_id, status)

    async def _zip_and_finalize(self, batch_id: str, status: BatchStatus) -> None:
        from app.services.zipper import build_batch_zip

        zip_path = await build_batch_zip(batch_id)
        await job_store.update_batch_status(batch_id, status, zip_path=str(zip_path) if zip_path else None)


worker_pool = WorkerPool()
