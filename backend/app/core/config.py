"""
Centralized configuration for AI Video Studio.

Everything here is overridable via environment variables (or a `.env` file
in /backend). This is intentional: the same codebase runs unmodified on a
local 4090 box and on a RunPod pod — only the env values differ.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field


BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # ---- Deployment -------------------------------------------------
    deployment_mode: Literal["local", "runpod"] = Field(
        default="local", validation_alias="DEPLOYMENT_MODE"
    )
    # If true, force CPU paths where supported (TTS only — video models
    # require CUDA and will raise a clear error instead of silently
    # running at unusable speed).
    force_cpu: bool = Field(default=False, validation_alias="FORCE_CPU")

    # ---- Paths --------------------------------------------------------
    models_cache_dir: Path = Field(
        default=BACKEND_ROOT / "models_cache", validation_alias="MODELS_CACHE_DIR"
    )
    outputs_dir: Path = Field(
        default=BACKEND_ROOT / "outputs", validation_alias="OUTPUTS_DIR"
    )
    uploads_dir: Path = Field(
        default=BACKEND_ROOT / "uploads", validation_alias="UPLOADS_DIR"
    )
    temp_dir: Path = Field(
        default=BACKEND_ROOT / "temp", validation_alias="TEMP_DIR"
    )
    db_path: Path = Field(
        default=BACKEND_ROOT / "jobs.db", validation_alias="DB_PATH"
    )

    # ---- HuggingFace ---------------------------------------------------
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")
    hf_home: Path = Field(
        default=BACKEND_ROOT / "models_cache" / "hf", validation_alias="HF_HOME"
    )

    # ---- Video generation defaults ------------------------------------
    default_video_model: str = Field(
        default="ltx-video", validation_alias="DEFAULT_VIDEO_MODEL"
    )
    default_width: int = Field(default=1280, validation_alias="DEFAULT_WIDTH")
    default_height: int = Field(default=720, validation_alias="DEFAULT_HEIGHT")
    default_num_frames: int = Field(default=121, validation_alias="DEFAULT_NUM_FRAMES")
    default_fps: int = Field(default=24, validation_alias="DEFAULT_FPS")

    # ---- TTS ------------------------------------------------------------
    default_tts_engine: Literal["xtts", "piper"] = Field(
        default="xtts", validation_alias="DEFAULT_TTS_ENGINE"
    )
    piper_voices_dir: Path = Field(
        default=BACKEND_ROOT / "models_cache" / "piper_voices",
        validation_alias="PIPER_VOICES_DIR",
    )

    # ---- Concurrency / batch --------------------------------------------
    # Most consumer/A100 single-GPU boxes can only hold ONE video diffusion
    # pipeline resident in VRAM at a time. max_parallel_gpu_jobs > 1 is only
    # safe on multi-GPU machines (see worker pool GPU-affinity logic).
    max_parallel_gpu_jobs: int = Field(default=1, validation_alias="MAX_PARALLEL_GPU_JOBS")
    max_batch_size: int = Field(default=50, validation_alias="MAX_BATCH_SIZE")

    # ---- RunPod -----------------------------------------------------------
    runpod_api_key: str | None = Field(default=None, validation_alias="RUNPOD_API_KEY")
    runpod_volume_path: Path = Field(
        default=Path("/workspace"), validation_alias="RUNPOD_VOLUME_PATH"
    )

    # ---- Server -------------------------------------------------------------
    cors_origins: list[str] = Field(default=["*"], validation_alias="CORS_ORIGINS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Ensure runtime directories exist at import time.
for d in (
    settings.models_cache_dir,
    settings.outputs_dir,
    settings.uploads_dir,
    settings.temp_dir,
    settings.hf_home,
    settings.piper_voices_dir,
):
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(settings.hf_home))
if settings.hf_token:
    os.environ.setdefault("HF_TOKEN", settings.hf_token)
