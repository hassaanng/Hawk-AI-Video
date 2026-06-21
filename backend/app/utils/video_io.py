"""
Shared frame <-> MP4 and image loading utilities used by every backend.

Centralizing this means there is exactly one place that knows how
diffusers' `frames` output (a list of PIL.Image objects) becomes a real
H.264 MP4 on disk, and exactly one place that knows how an uploaded
init image gets resized/cropped to a model's required resolution.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def load_image_rgb(path: Path, target_width: int, target_height: int) -> Image.Image:
    """Load an image, convert to RGB, and center-crop/resize to the exact
    resolution the video model's image-to-video pipeline expects."""
    img = Image.open(path).convert("RGB")

    src_ratio = img.width / img.height
    dst_ratio = target_width / target_height

    if src_ratio > dst_ratio:
        new_height = img.height
        new_width = int(new_height * dst_ratio)
    else:
        new_width = img.width
        new_height = int(new_width / dst_ratio)

    left = (img.width - new_width) // 2
    top = (img.height - new_height) // 2
    img = img.crop((left, top, left + new_width, top + new_height))
    img = img.resize((target_width, target_height), Image.LANCZOS)
    return img


def export_frames_to_mp4(frames: list[Image.Image], out_path: Path, fps: int) -> Path:
    """
    Encode a list of PIL frames into a real H.264 MP4 via ffmpeg, piping
    raw RGB bytes over stdin rather than writing intermediate PNGs to
    disk (faster, avoids filename-ordering bugs, leaves no frame litter).
    """
    if not frames:
        raise ValueError("No frames produced by the model — cannot export an empty video.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = frames[0].size

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        "-movflags", "+faststart",
        str(out_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            if frame.size != (width, height):
                frame = frame.resize((width, height), Image.LANCZOS)
            proc.stdin.write(frame.convert("RGB").tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read()
        proc.wait(timeout=300)
    except Exception:
        proc.kill()
        raise

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame export failed (code {proc.returncode}): {stderr.decode(errors='ignore')[-2000:]}")

    return out_path
