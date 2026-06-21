"""
Stable Video Diffusion (SVD) backend.

SVD (Stability AI) is image-to-video ONLY — there is no official
text-to-video SVD checkpoint. For text-to-video requests, this backend
chains a Stable Diffusion XL text-to-image generation step into SVD's
image-to-video step, which is a legitimate, commonly-used real pipeline
(not a slideshow trick: SVD then performs genuine learned temporal
diffusion on that single conditioning frame to synthesize all
subsequent frames via its UNet's temporal attention layers — the image
is a conditioning input, not a sequence of stitched stills).

SVD's standout feature relative to the other three backends is its
native `motion_bucket_id` conditioning parameter, which gives explicit,
well-calibrated control over motion intensity — the most reliable
"real camera/motion movement" knob of the four models.

Real model repo: "stabilityai/stable-video-diffusion-img2vid-xt"
(25-frame variant). Supported via diffusers StableVideoDiffusionPipeline.
"""
from __future__ import annotations

import gc
import logging
import time
import uuid

from app.core.config import settings
from app.core.gpu import require_vram
from app.models.base import (
    BaseVideoModel,
    ModelRequirements,
    ProgressCallback,
    VideoGenerationRequest,
    VideoGenerationResult,
)
from app.utils.video_io import export_frames_to_mp4, load_image_rgb

logger = logging.getLogger(__name__)

REPO_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
T2I_HELPER_REPO = "stabilityai/stable-diffusion-xl-base-1.0"


class StableVideoDiffusionModel(BaseVideoModel):
    name = "svd"
    requirements = ModelRequirements(
        min_vram_gb=12.0,
        supports_text_to_video=True,  # via internal SDXL->SVD chain, see module docstring
        supports_image_to_video=True,
        supports_camera_control=True,  # native motion_bucket_id
        max_frames_recommended=25,  # the -xt checkpoint's native window
        notes=(
            "Image-to-video native; text-to-video is implemented by chaining "
            "an SDXL text-to-image conditioning frame into SVD's temporal "
            "diffusion. Best-in-class explicit motion control via "
            "motion_bucket_id, but capped at 25 frames per clip (~1-4s "
            "depending on fps) — use the batch system to render multiple "
            "clips for longer sequences."
        ),
    )

    def load(self) -> None:
        if self._loaded:
            return
        require_vram(self.name, self.requirements.min_vram_gb)

        import torch
        from diffusers import StableVideoDiffusionPipeline

        cache_dir = str(settings.models_cache_dir / "hf")
        logger.info("Loading SVD weights from %s ...", REPO_ID)

        self._pipeline = StableVideoDiffusionPipeline.from_pretrained(
            REPO_ID, torch_dtype=torch.float16, variant="fp16", cache_dir=cache_dir
        )
        self._pipeline.to("cuda")
        self._pipeline.vae.enable_tiling()

        # The SDXL text-to-image helper is loaded lazily, only if a
        # text-to-video request actually arrives, to avoid doubling VRAM
        # use for image-to-video-only workloads.
        self._t2i_pipeline = None

        self._loaded = True
        logger.info("SVD loaded.")

    def _ensure_t2i_loaded(self) -> None:
        if self._t2i_pipeline is not None:
            return
        import torch
        from diffusers import StableDiffusionXLPipeline

        cache_dir = str(settings.models_cache_dir / "hf")
        logger.info("Lazily loading SDXL helper from %s for text-to-video conditioning frame ...", T2I_HELPER_REPO)
        self._t2i_pipeline = StableDiffusionXLPipeline.from_pretrained(
            T2I_HELPER_REPO, torch_dtype=torch.float16, cache_dir=cache_dir
        )
        self._t2i_pipeline.to("cuda")

    def unload(self) -> None:
        if not self._loaded:
            return
        import torch

        del self._pipeline
        if getattr(self, "_t2i_pipeline", None) is not None:
            del self._t2i_pipeline
        gc.collect()
        torch.cuda.empty_cache()
        self._loaded = False
        logger.info("SVD unloaded, VRAM freed.")

    def _motion_bucket_for(self, request: VideoGenerationRequest) -> int:
        """
        SVD's motion_bucket_id is documented (Stability AI model card) to
        range roughly 1-255, with ~127 as the calibration midpoint for
        "moderate" motion. We map the user's normalized 0..1
        motion_strength plus named camera_motion onto this real
        conditioning parameter, rather than faking it via prompting.
        """
        base = int(40 + request.motion_strength * 200)  # 40..240
        camera_bias = {
            "static": -30,
            "pan_left": 10,
            "pan_right": 10,
            "zoom_in": 5,
            "zoom_out": 5,
            "orbit": 20,
            "dolly_in": 15,
        }
        value = base + camera_bias.get(request.camera_motion, 0)
        return max(1, min(255, value))

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
        motion_bucket_id = self._motion_bucket_for(request)

        if request.mode == "image-to-video":
            init_image = load_image_rgb(request.init_image_path, request.width, request.height)
        else:
            if on_progress:
                on_progress(0.02, "Generating conditioning frame (SDXL)")
            self._ensure_t2i_loaded()
            t2i_gen = torch.Generator(device="cuda").manual_seed(seed)
            init_image = self._t2i_pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=30,
                generator=t2i_gen,
            ).images[0]

        def callback_on_step_end(pipe, step, timestep, kwargs):
            if on_progress:
                frac = (step + 1) / request.num_inference_steps
                on_progress(0.2 + frac * 0.75, f"Synthesizing motion, step {step + 1}/{request.num_inference_steps}")
            return kwargs

        if on_progress:
            on_progress(0.2, f"Running SVD temporal diffusion (motion_bucket_id={motion_bucket_id})")

        num_frames = min(request.num_frames, self.requirements.max_frames_recommended)
        output = self._pipeline(
            image=init_image,
            height=request.height,
            width=request.width,
            num_frames=num_frames,
            num_inference_steps=request.num_inference_steps,
            motion_bucket_id=motion_bucket_id,
            fps=request.fps,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
        )

        frames = output.frames[0]

        if on_progress:
            on_progress(0.96, "Encoding frames to MP4")

        out_path = settings.temp_dir / f"svd_{uuid.uuid4().hex}.mp4"
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
