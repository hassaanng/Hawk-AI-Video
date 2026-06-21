"""
Pydantic schemas defining the HTTP API contract. Kept separate from
the internal dataclasses in app/models/base.py and app/services/* on
purpose: the API shape (what a frontend client sends/receives) and the
internal pipeline shape are allowed to evolve independently.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GenerationParams(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    model_name: str = "ltx-video"
    negative_prompt: str | None = None
    mode: Literal["text-to-video", "image-to-video"] = "text-to-video"
    init_image_path: str | None = None  # set server-side after upload; ignored if client sends it
    reference_image_path: str | None = None
    width: int = 1280
    height: int = 720
    num_frames: int = 121
    fps: int = 24
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    seed: int | None = None
    camera_motion: Literal[
        "static", "pan_left", "pan_right", "zoom_in", "zoom_out", "orbit", "dolly_in"
    ] = "static"
    motion_strength: float = Field(default=1.0, ge=0.0, le=1.0)

    enable_voiceover: bool = True
    narration_text: str | None = None  # if None, auto-generated from prompt
    tts_engine: Literal["xtts", "piper"] = "xtts"
    tts_voice: str | None = None
    reference_audio_path: str | None = None  # for XTTS voice cloning


class SingleGenerationRequest(GenerationParams):
    pass


class BatchGenerationRequest(BaseModel):
    prompts: list[GenerationParams] = Field(..., min_length=1, max_length=50)


class JobOut(BaseModel):
    id: str
    batch_id: str
    idx_in_batch: int
    prompt: str
    status: str
    progress: float
    status_message: str | None
    output_path: str | None
    error: str | None
    created_at: float
    updated_at: float

    @property
    def download_url(self) -> str | None:
        return f"/api/jobs/{self.id}/download" if self.output_path else None


class BatchOut(BaseModel):
    id: str
    status: str
    total: int
    created_at: float
    updated_at: float
    zip_path: str | None
    jobs: list[JobOut] = []


class ModelInfoOut(BaseModel):
    name: str
    min_vram_gb: float
    supports_text_to_video: bool
    supports_image_to_video: bool
    supports_camera_control: bool
    max_frames_recommended: int
    notes: str


class GpuInfoOut(BaseModel):
    available: bool
    device_count: int
    name: str | None
    total_vram_gb: float | None
    free_vram_gb: float | None
    cuda_version: str | None
    driver_version: str | None
