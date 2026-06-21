"""
Model registry: the single switchboard for "model swapping".

API routes, the batch queue, and workers never `import` a concrete
backend (LTXVideoModel, WanVideoModel, ...) directly. They ask this
registry for "the model named X" and get back something that satisfies
`BaseVideoModel`. Adding a fifth backend later means:

    1. Write app/models/my_new_model.py subclassing BaseVideoModel.
    2. Add one line to _BACKEND_CLASSES below.

Nothing in api/, services/, or workers/ changes.

This module also owns the "only one video pipeline resident in VRAM at
a time" policy for single-GPU machines: `get_model()` will unload
whatever was previously active before loading the newly requested one,
unless MAX_PARALLEL_GPU_JOBS > 1 (multi-GPU setups handle this in the
worker pool via GPU affinity instead — see workers/gpu_pool.py).
"""
from __future__ import annotations

import logging
import threading

from app.core.config import settings
from app.models.base import BaseVideoModel, ModelRequirements
from app.models.hunyuan_video import HunyuanVideoModel
from app.models.ltx_video import LTXVideoModel
from app.models.svd_video import StableVideoDiffusionModel
from app.models.wan_video import WanVideoModel

logger = logging.getLogger(__name__)

_BACKEND_CLASSES: dict[str, type[BaseVideoModel]] = {
    "ltx-video": LTXVideoModel,
    "wan2.1": WanVideoModel,
    "hunyuan-video": HunyuanVideoModel,
    "svd": StableVideoDiffusionModel,
}


class ModelRegistry:
    def __init__(self) -> None:
        self._instances: dict[str, BaseVideoModel] = {}
        self._active_name: str | None = None
        self._lock = threading.Lock()

    def list_models(self) -> dict[str, ModelRequirements]:
        return {name: cls.requirements for name, cls in _BACKEND_CLASSES.items()}

    def get_model(self, name: str, exclusive_vram: bool = True) -> BaseVideoModel:
        """
        Return a loaded model instance for `name`. If `exclusive_vram` is
        True (the default, correct setting for single-GPU machines) and a
        *different* model is currently loaded, that model is unloaded
        first to free VRAM before the requested one loads.
        """
        if name not in _BACKEND_CLASSES:
            raise ValueError(
                f"Unknown model '{name}'. Available: {', '.join(_BACKEND_CLASSES)}"
            )

        with self._lock:
            if name not in self._instances:
                self._instances[name] = _BACKEND_CLASSES[name]()

            if exclusive_vram and self._active_name and self._active_name != name:
                prev = self._instances.get(self._active_name)
                if prev is not None and prev.is_loaded:
                    logger.info("Swapping models: unloading '%s' to load '%s'", self._active_name, name)
                    prev.unload()

            model = self._instances[name]
            model.load()
            self._active_name = name
            return model

    def unload_all(self) -> None:
        with self._lock:
            for model in self._instances.values():
                if model.is_loaded:
                    model.unload()
            self._active_name = None

    @property
    def active_model_name(self) -> str | None:
        return self._active_name


registry = ModelRegistry()


def default_model_name() -> str:
    return settings.default_video_model
