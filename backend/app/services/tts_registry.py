"""
TTS registry — same swap pattern as app/models/registry.py. TTS engines
are far lighter than video models, so unlike the video registry, both
engines may be kept loaded simultaneously without VRAM pressure
concerns (XTTS-v2 is ~2GB; Piper is CPU-resident ONNX, negligible).
"""
from __future__ import annotations

import logging
import threading

from app.core.config import settings
from app.services.tts_base import BaseTTSEngine
from app.services.tts_piper import PiperEngine
from app.services.tts_xtts import XTTSEngine

logger = logging.getLogger(__name__)

_ENGINE_CLASSES: dict[str, type[BaseTTSEngine]] = {
    "xtts": XTTSEngine,
    "piper": PiperEngine,
}


class TTSRegistry:
    def __init__(self) -> None:
        self._instances: dict[str, BaseTTSEngine] = {}
        self._lock = threading.Lock()

    def list_engines(self) -> list[str]:
        return list(_ENGINE_CLASSES)

    def get_engine(self, name: str) -> BaseTTSEngine:
        if name not in _ENGINE_CLASSES:
            raise ValueError(f"Unknown TTS engine '{name}'. Available: {', '.join(_ENGINE_CLASSES)}")
        with self._lock:
            if name not in self._instances:
                self._instances[name] = _ENGINE_CLASSES[name]()
            engine = self._instances[name]
        engine.load()
        return engine


tts_registry = TTSRegistry()


def default_tts_name() -> str:
    return settings.default_tts_engine
