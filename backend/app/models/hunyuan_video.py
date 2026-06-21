"""
HunyuanVideo backend.

HunyuanVideo (Tencent) is the heaviest of the four backends: the
official 13B-parameter transformer requires ~60GB VRAM unquantized.
Two practical paths are wired up here:

  1. Full-precision on a single 80GB GPU (A100/H100), or
  2. CPU-offloaded sequential loading via `enable_model_cpu_offload()`,
     which lets it run on a single 24-48GB GPU at a significant speed
     cost (each forward pass swaps weights HBM<->VRAM).

Real model repo: "tencent/HunyuanVideo" on HuggingFace, supported in
diffusers>=0.31 via HunyuanVideoPipeline.
"""
from __future__ import annotations

import gc
import logging
import time
import uuid

from app.core.config import settings
from app.core.gpu import detect_gpu, require_vram
from app.models.base import (
    BaseVideoModel,
    ModelRequirements,
    ProgressCallback,
    VideoGenerationRequest,
    VideoGenerationResult,
)
from app.utils.video_io import export_frames_to_mp4

logger = logging.getLogger(__name__)

REPO_ID = "tencent/HunyuanVideo"


class HunyuanVideoModel(BaseVideoModel):
    name = "hunyuan-video"
    requirements = ModelRequirements(
        min_vram_gb=24.0,  # floor when CPU-offloaded; ~60GB for full in-VRAM
        supports_text_to_video=True,
        supports_image_to_video=False,  # official I2V variant not yet in stable diffusers release
        supports_camera_control=False,
        max_frames_recommended=129,
        notes=(
            "Highest visual fidelity, slowest backend. On GPUs under 60GB free "
            "VRAM, automatically enables sequential CPU offload, which trades "
            "significant speed (3-5x slower) for the ability to run at all. "
            "Image-to-video is not exposed here because the official I2V "
            "checkpoint is not yet stable in the public diffusers pipeline."
        ),
    )

    def load(self) -> None:
        if self._loaded:
            return
        require_vram(self.name, self.requirements.min_vram_gb)

        import torch
        from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

        info = detect_gpu()
        cache_dir = str(settings.models_cache_dir / "hf")

        logger.info("Loading HunyuanVideo transformer from %s ...", REPO_ID)
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            REPO_ID, subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )
        self._pipeline = HunyuanVideoPipeline.from_pretrained(
            REPO_ID, transformer=transformer, torch_dtype=torch.float16, cache_dir=cache_dir
        )

        self._cpu_offloaded = info.free_vram_gb is not None and info.free_vram_gb < 60.0
        if self._cpu_offloaded:
            logger.warning(
                "Free VRAM (%s GB) is under the 60GB full-precision threshold; "
                "enabling sequential CPU offload. Generation will be considerably slower.",
                info.free_vram_gb,
            )
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline.to("cuda")

        self._pipeline.vae.enable_tiling()
        self._loaded = True
        logger.info("HunyuanVideo loaded (cpu_offload=%s).", self._cpu_offloaded)

    def unload(self) -> None:
        if not self._loaded:
            return
        import torch

        del self._pipeline
        gc.collect()
        torch.cuda.empty_cache()
        self._loaded = False
        logger.info("HunyuanVideo unloaded, VRAM freed.")

    def generate(
        self,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResult:
        self.validate_request(request)
        self.load()

        import torch

        seed = request.seed if request.seed is not None else int(time.time())
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # No native camera-pose control: fold the directive into the prompt,
        # same fallback strategy as LTX-Video, since HunyuanVideo's public
        # pipeline likewise lacks a dedicated control tensor for this.
        camera_terms = {
            "static": "static shot,",
            "pan_left": "camera pans left,",
            "pan_right": "camera pans right,",
            "zoom_in": "camera zooms in slowly,",
            "zoom_out": "camera zooms out slowly,",
            "orbit": "camera orbits the subject,",
            "dolly_in": "dolly-in camera movement,",
        }
        full_prompt = f"{camera_terms.get(request.camera_motion, '')} {request.prompt}".strip()

        def callback_on_step_end(pipe, step, timestep, kwargs):
            if on_progress:
                frac = (step + 1) / request.num_inference_steps
                on_progress(frac * 0.95, f"Denoising step {step + 1}/{request.num_inference_steps}")
            return kwargs

        if on_progress:
            on_progress(0.02, "Encoding prompt (HunyuanVideo)")

        output = self._pipeline(
            prompt=full_prompt,
            negative_prompt=request.negative_prompt or "blurry, low quality, distorted, watermark",
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
        )

        frames = output.frames[0]

        if on_progress:
            on_progress(0.96, "Encoding frames to MP4")

        out_path = settings.temp_dir / f"hunyuan_{uuid.uuid4().hex}.mp4"
        export_frames_to_mp4(frames, out_path, fps=request.fps)

        if on_progress:
            on_progress(1.0, "Done")

        return VideoGenerationResult(
            output_path=out_path,
            width=request.width,
            height=request.height,
            num_frames=len(frames),
            fps=request.fps,
            duration_seconds=len(frames) / request.fps,
            seed_used=seed,
            model_name=self.name,
        )
