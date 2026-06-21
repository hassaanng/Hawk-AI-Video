"""
API routes.

Design choice: even a "single video" request is implemented as a
batch-of-one under the hood (one row in `batches`, one row in `jobs`).
This means there is exactly one code path for queueing, progress
polling, and download — no special-cased "single job" logic duplicating
the batch logic with subtly different bugs.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.gpu import detect_gpu
from app.models.registry import registry as video_registry
from app.schemas.api_schemas import (
    BatchGenerationRequest,
    BatchOut,
    GpuInfoOut,
    JobOut,
    ModelInfoOut,
    SingleGenerationRequest,
)
from app.services.job_store import job_store
from app.services.tts_registry import tts_registry

router = APIRouter(prefix="/api")


# ---- system info ------------------------------------------------------

@router.get("/system/gpu", response_model=GpuInfoOut)
async def get_gpu_info():
    info = detect_gpu()
    return GpuInfoOut(**{k: v for k, v in info.__dict__.items() if k != "compute_capability"})


@router.get("/system/models", response_model=list[ModelInfoOut])
async def list_models():
    out = []
    for name, req in video_registry.list_models().items():
        out.append(
            ModelInfoOut(
                name=name,
                min_vram_gb=req.min_vram_gb,
                supports_text_to_video=req.supports_text_to_video,
                supports_image_to_video=req.supports_image_to_video,
                supports_camera_control=req.supports_camera_control,
                max_frames_recommended=req.max_frames_recommended,
                notes=req.notes,
            )
        )
    return out


@router.get("/system/tts-engines")
async def list_tts_engines():
    return {"engines": tts_registry.list_engines()}


# ---- uploads -----------------------------------------------------------

@router.post("/uploads/image")
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename or "image.png").suffix or ".png"
    dest = settings.uploads_dir / f"{uuid.uuid4().hex}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": str(dest)}


@router.post("/uploads/audio")
async def upload_audio(file: UploadFile = File(...)):
    """For XTTS voice cloning: a short reference clip of the target voice."""
    ext = Path(file.filename or "ref.wav").suffix or ".wav"
    dest = settings.uploads_dir / f"{uuid.uuid4().hex}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": str(dest)}


# ---- generation (single = batch of one) --------------------------------

def _params_to_dict(params: SingleGenerationRequest) -> dict:
    return params.model_dump()


@router.post("/generate", response_model=BatchOut)
async def generate_single(request: SingleGenerationRequest):
    batch_id = await job_store.create_batch([_params_to_dict(request)])
    return await _serialize_batch(batch_id)


@router.post("/generate/batch", response_model=BatchOut)
async def generate_batch(request: BatchGenerationRequest):
    if len(request.prompts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.prompts)} exceeds max of {settings.max_batch_size}.",
        )
    batch_id = await job_store.create_batch([_params_to_dict(p) for p in request.prompts])
    return await _serialize_batch(batch_id)


# ---- batch / job status --------------------------------------------------

async def _serialize_batch(batch_id: str) -> BatchOut:
    batch = await job_store.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    jobs = await job_store.get_jobs_for_batch(batch_id)
    return BatchOut(**batch, jobs=[JobOut(**j) for j in jobs])


@router.get("/batches", response_model=list[BatchOut])
async def list_batches():
    batches = await job_store.list_batches()
    out = []
    for b in batches:
        jobs = await job_store.get_jobs_for_batch(b["id"])
        out.append(BatchOut(**b, jobs=[JobOut(**j) for j in jobs]))
    return out


@router.get("/batches/{batch_id}", response_model=BatchOut)
async def get_batch(batch_id: str):
    return await _serialize_batch(batch_id)


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str):
    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobOut(**job)


# ---- downloads -----------------------------------------------------------

@router.get("/jobs/{job_id}/download")
async def download_job_video(job_id: str):
    job = await job_store.get_job(job_id)
    if not job or not job["output_path"]:
        raise HTTPException(status_code=404, detail="Video not ready or job not found")
    path = Path(job["output_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=f"video_{job_id[:8]}.mp4")


@router.get("/batches/{batch_id}/download-all")
async def download_batch_zip(batch_id: str):
    batch = await job_store.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if not batch["zip_path"]:
        raise HTTPException(status_code=409, detail="Batch is not finished yet, or no jobs succeeded")
    path = Path(batch["zip_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Zip file missing on disk")
    return FileResponse(path, media_type="application/zip", filename=f"batch_{batch_id[:8]}.zip")
