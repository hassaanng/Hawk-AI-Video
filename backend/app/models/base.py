"""
Abstract interface every video generation backend must implement.

This is the contract that makes "model swapping" real rather than a
marketing phrase: the API layer, the batch queue, and the frontend never
import diffusers/HunyuanVideo/Wan code directly. They only ever call
methods on `BaseVideoModel`. Adding a new model = writing one subclass
and registering it in `registry.py`; nothing else in the codebase changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal


@dataclass
class VideoGenerationRequest:
    prompt: str
    negative_prompt: str | None = None
    mode: Literal["text-to-video", "image-to-video"] = "text-to-video"
    init_image_path: Path | None = None
    # Used for consistent-character generation: a reference image whose
    # subject should be preserved across one or many generated clips.
    reference_image_path: Path | None = None
    width: int = 1280
    height: int = 720
    num_frames: int = 121
    fps: int = 24
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    seed: int | None = None
    # Real camera movement directive, translated by each backend into
    # whatever conditioning mechanism it supports (motion-bucket id for
    # SVD, camera LoRA / motion strength for Wan, prompt-engineering
    # fallback for models with no native camera control).
    camera_motion: Literal[
        "static", "pan_left", "pan_right", "zoom_in", "zoom_out", "orbit", "dolly_in"
    ] = "static"
    motion_strength: float = 1.0  # 0..1, backend-normalized


@dataclass
class VideoGenerationResult:
    output_path: Path
    width: int
    height: int
    num_frames: int
    fps: int
    duration_seconds: float
    seed_used: int
    model_name: str


ProgressCallback = Callable[[float, str], None]  # (0..1 progress, status message)


@dataclass
class ModelRequirements:
    min_vram_gb: float
    supports_text_to_video: bool = True
    supports_image_to_video: bool = False
    supports_camera_control: bool = False
    max_frames_recommended: int = 121
    notes: str = ""


class BaseVideoModel(ABC):
    """
    Every concrete backend (LTX-Video, Wan 2.1, HunyuanVideo, SVD, ...)
    subclasses this. Lifecycle:

        model = SomeModel()
        model.load()                  # loads weights onto GPU, idempotent
        result = model.generate(req, on_progress=cb)
        model.unload()                # frees VRAM for the next model
    """

    name: str
    requirements: ModelRequirements

    def __init__(self) -> None:
        self._pipeline = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load weights onto GPU. Must be safe to call multiple times (no-op if loaded)."""

    @abstractmethod
    def unload(self) -> None:
        """Free VRAM. Must be safe to call even if never loaded."""

    @abstractmethod
    def generate(
        self,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResult:
        """Run inference and write an MP4 to disk, returning its metadata."""

    def validate_request(self, request: VideoGenerationRequest) -> None:
        if request.mode == "image-to-video" and not self.requirements.supports_image_to_video:
            raise ValueError(f"{self.name} does not support image-to-video generation.")
        if request.mode == "image-to-video" and request.init_image_path is None:
            raise ValueError("image-to-video mode requires init_image_path.")
        if request.num_frames > self.requirements.max_frames_recommended:
            raise ValueError(
                f"{self.name} recommends <= {self.requirements.max_frames_recommended} frames; "
                f"got {request.num_frames}. Longer clips fragment quality on this backend — "
                f"generate multiple clips and concatenate instead."
            )
