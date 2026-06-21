"""
GPU detection utilities.

Video diffusion models are VRAM-hungry and each backend has a real,
measured floor below which it will OOM or silently produce garbage
output (truncated frames, NaN tiles). This module is the single source
of truth other modules query before deciding whether a model can even
be attempted on the current hardware, so we fail fast with an actionable
error instead of a CUDA OOM stack trace fifteen minutes into a batch job.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # pragma: no cover - torch should always be installed
    torch = None  # type: ignore


@dataclass
class GPUInfo:
    available: bool
    device_count: int
    name: str | None
    total_vram_gb: float | None
    free_vram_gb: float | None
    cuda_version: str | None
    driver_version: str | None
    compute_capability: tuple[int, int] | None


def _nvidia_smi_driver_version() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip().splitlines()[0]
    except Exception:
        return None


def detect_gpu() -> GPUInfo:
    if torch is None or not torch.cuda.is_available():
        return GPUInfo(
            available=False,
            device_count=0,
            name=None,
            total_vram_gb=None,
            free_vram_gb=None,
            cuda_version=None,
            driver_version=_nvidia_smi_driver_version(),
            compute_capability=None,
        )

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)

    return GPUInfo(
        available=True,
        device_count=torch.cuda.device_count(),
        name=props.name,
        total_vram_gb=round(total_bytes / (1024**3), 2),
        free_vram_gb=round(free_bytes / (1024**3), 2),
        cuda_version=torch.version.cuda,
        driver_version=_nvidia_smi_driver_version(),
        compute_capability=(props.major, props.minor),
    )


class InsufficientVRAMError(RuntimeError):
    """Raised when the active GPU does not meet a model's documented floor."""


def require_vram(model_name: str, required_gb: float) -> GPUInfo:
    """
    Raise a clear, actionable error if the current GPU can't fit the model
    instead of letting diffusers/torch throw an opaque CUDA OOM mid-generation.
    """
    info = detect_gpu()
    if not info.available:
        raise InsufficientVRAMError(
            f"No CUDA-capable GPU detected. '{model_name}' requires a GPU with "
            f"at least {required_gb} GB VRAM. Video diffusion models cannot run "
            f"on CPU in any practical timeframe. If you are on RunPod, confirm "
            f"the pod template actually attached a GPU (check `nvidia-smi`)."
        )
    if info.free_vram_gb is not None and info.free_vram_gb < required_gb:
        raise InsufficientVRAMError(
            f"'{model_name}' needs ~{required_gb} GB free VRAM but only "
            f"{info.free_vram_gb} GB is currently free on {info.name} "
            f"(total: {info.total_vram_gb} GB). Free VRAM by stopping other "
            f"jobs/processes, switching to a lighter model (try LTX-Video), "
            f"or enabling CPU offload in the model's settings."
        )
    return info
