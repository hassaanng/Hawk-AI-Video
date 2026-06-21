"""
LTX-Video backend.

LTX-Video (Lightricks) is a DiT-based video diffusion model distributed
through the `diffusers` library as of diffusers>=0.30. It is the
lightest of the four backends here (fits in ~12-24GB VRAM depending on
resolution/frame count) and the fastest, making it the recommended
default for development and for anyone without an A100/H100.

Real model repo: "Lightricks/LTX-Video" on HuggingFace.
Pipelines used: LTXPipeline (text-to-video), LTXImageToVideoPipeline (image-to-video).
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

REPO_ID = "Lightricks/LTX-Video"


class LTXVideoModel(BaseVideoModel):
    name = "ltx-video"
    requirements = ModelRequirements(
        min_vram_gb=12.0,
        supports_text_to_video=True,
        supports_image_to_video=True,
        supports_camera_control=False,  # no native camera-pose conditioning; uses prompt-injection fallback
        max_frames_recommended=161,
        notes=(
            "Fastest, lowest-VRAM backend of the four. Best default for iteration. "
            "Camera movement is approximated via prompt conditioning rather than "
            "an explicit control signal — see _apply_camera_motion_to_prompt."
        ),
    )

    def load(self) -> None:
        if self._loaded:
            return
        require_vram(self.name, self.requirements.min_vram_gb)

        import torch
        from diffusers import LTXPipeline, LTXImageToVideoPipeline

        logger.info("Loading LTX-Video weights from %s ...", REPO_ID)
        dtype = torch.bfloat16

        self._t2v_pipeline = LTXPipeline.from_pretrained(
            REPO_ID, torch_dtype=dtype, cache_dir=str(settings.models_cache_dir / "hf")
        )
        self._i2v_pipeline = LTXImageToVideoPipeline.from_pretrained(
            REPO_ID, torch_dtype=dtype, cache_dir=str(settings.models_cache_dir / "hf")
        )

        self._t2v_pipeline.to("cuda")
        self._i2v_pipeline.vae = self._t2v_pipeline.vae
        self._i2v_pipeline.transformer = self._t2v_pipeline.transformer
        self._i2v_pipeline.to("cuda")

        # VAE tiling keeps peak VRAM down for 720p+ outputs at the cost of
        # a small amount of speed - worthwhile tradeoff on 24GB cards.
        self._t2v_pipeline.vae.enable_tiling()

        self._loaded = True
        logger.info("LTX-Video loaded.")

    def unload(self) -> None:
        if not self._loaded:
            return
        import torch

        del self._t2v_pipeline
        del self._i2v_pipeline
        gc.collect()
        torch.cuda.empty_cache()
        self._loaded = False
        logger.info("LTX-Video unloaded, VRAM freed.")

    def _apply_camera_motion_to_prompt(self, prompt: str, request: VideoGenerationRequest) -> str:
        """
        LTX-Video has no dedicated camera-pose control tensor in its public
        pipeline, so real camera movement is induced the way the model's
        own authors recommend: explicit, front-loaded motion language in
        the prompt, which the DiT's temporal attention reliably responds to.
        """
        directives = {
            "static": "static locked-off camera shot,",
            "pan_left": "smooth cinematic camera pan to the left,",
            "pan_right": "smooth cinematic camera pan to the right,",
            "zoom_in": "slow cinematic camera zoom in, dolly forward,",
            "zoom_out": "slow cinematic camera zoom out, pulling back,",
            "orbit": "cinematic orbit camera movement circling the subject,",
            "dolly_in": "cinematic dolly-in tracking shot moving toward the subject,",
        }
        prefix = directives.get(request.camera_motion, "")
        return f"{prefix} {prompt}".strip()

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
        full_prompt = self._apply_camera_motion_to_prompt(request.prompt, request)

        def callback_on_step_end(pipe, step, timestep, kwargs):
            if on_progress:
                frac = (step + 1) / request.num_inference_steps
                on_progress(frac * 0.95, f"Denoising step {step + 1}/{request.num_inference_steps}")
            return kwargs

        if on_progress:
            on_progress(0.02, "Encoding prompt")

        common_kwargs = dict(
            prompt=full_prompt,
            negative_prompt=request.negative_prompt or "blurry, distorted, low quality, watermark, text",
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
        )

        if request.mode == "image-to-video":
            init_image = load_image_rgb(request.init_image_path, request.width, request.height)
            output = self._i2v_pipeline(image=init_image, **common_kwargs)
        else:
            output = self._t2v_pipeline(**common_kwargs)

        frames = output.frames[0]  # list[PIL.Image] for the single batch item

        if on_progress:
            on_progress(0.96, "Encoding frames to MP4")

        out_path = settings.temp_dir / f"ltx_{uuid.uuid4().hex}.mp4"
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
