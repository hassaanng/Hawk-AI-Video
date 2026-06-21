"""
Per-job pipeline: this is what actually runs for one prompt, whether
it arrived as a single ad-hoc request or as item #37 of a 50-prompt
batch. Wires together, in order:

    1. Video generation (via the model registry — backend-agnostic)
    2. Narration script generation (auto, unless caller supplied text)
    3. TTS synthesis of that narration (via the TTS registry)
    4. ffmpeg merge of narration audio onto the generated video
    5. Final HD MP4 guarantee + move into outputs_dir

Progress is reported through a single callback so both the single-job
API endpoint and the batch worker get live progress without duplicating
this logic.
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Callable

from app.core.config import settings
from app.models.base import VideoGenerationRequest
from app.models.registry import registry as video_registry
from app.services.ffmpeg_service import merge_audio_video, transcode_to_hd_mp4
from app.services.narration import generate_narration
from app.services.tts_base import TTSRequest
from app.services.tts_registry import tts_registry

logger = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


def run_generation_pipeline(
    *,
    prompt: str,
    model_name: str,
    negative_prompt: str | None = None,
    mode: str = "text-to-video",
    init_image_path: Path | None = None,
    width: int = 1280,
    height: int = 720,
    num_frames: int = 121,
    fps: int = 24,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.0,
    seed: int | None = None,
    camera_motion: str = "static",
    motion_strength: float = 1.0,
    enable_voiceover: bool = True,
    narration_text: str | None = None,
    tts_engine_name: str = "xtts",
    tts_voice: str | None = None,
    reference_audio_path: Path | None = None,
    on_progress: ProgressFn | None = None,
) -> Path:
    """
    Runs the full pipeline synchronously (intended to be called from
    inside a worker thread/process — see workers/batch_worker.py — not
    directly on the FastAPI event loop, since video generation is a
    long blocking GPU operation).

    Returns the path to the final MP4 in settings.outputs_dir.
    """

    def report(frac: float, msg: str) -> None:
        if on_progress:
            on_progress(frac, msg)
        logger.info("[pipeline] %.0f%% - %s", frac * 100, msg)

    report(0.0, "Queued")

    model = video_registry.get_model(model_name)

    video_req = VideoGenerationRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        mode=mode,  # type: ignore[arg-type]
        init_image_path=init_image_path,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        camera_motion=camera_motion,  # type: ignore[arg-type]
        motion_strength=motion_strength,
    )

    def video_progress(frac: float, msg: str) -> None:
        # Video generation occupies the 0% -> 70% slice of total job
        # progress; narration + TTS + merge fill the remainder.
        report(frac * 0.70, msg)

    video_result = model.generate(video_req, on_progress=video_progress)
    report(0.70, "Video generated")

    if not enable_voiceover:
        final_path = settings.outputs_dir / f"{uuid.uuid4().hex}.mp4"
        transcode_to_hd_mp4(video_result.output_path, final_path)
        video_result.output_path.unlink(missing_ok=True)
        report(1.0, "Done (no voiceover requested)")
        return final_path

    report(0.72, "Generating narration script")
    script = narration_text or generate_narration(prompt)

    report(0.78, f"Synthesizing voiceover ({tts_engine_name})")
    tts_engine = tts_registry.get_engine(tts_engine_name)
    tts_result = tts_engine.synthesize(
        TTSRequest(
            text=script,
            voice=tts_voice,
            reference_audio_path=reference_audio_path,
            language="en",
            speed=1.0,
        )
    )

    report(0.90, "Merging narration with video")
    merged_path = settings.temp_dir / f"merged_{uuid.uuid4().hex}.mp4"
    merge_audio_video(video_result.output_path, tts_result.output_path, merged_path)

    report(0.96, "Final HD export")
    final_path = settings.outputs_dir / f"{uuid.uuid4().hex}.mp4"
    transcode_to_hd_mp4(merged_path, final_path)

    # Clean up intermediates — only the final muxed MP4 belongs in outputs_dir.
    for p in (video_result.output_path, tts_result.output_path, merged_path):
        Path(p).unlink(missing_ok=True)

    report(1.0, "Done")
    return final_path
