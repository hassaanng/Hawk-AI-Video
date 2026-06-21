"""
TTS backend interface + registry — mirrors app/models/registry.py exactly
so the swap pattern stays consistent across both video and voice.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TTSRequest:
    text: str
    voice: str | None = None  # backend-specific voice id/name
    reference_audio_path: Path | None = None  # for voice cloning (XTTS)
    language: str = "en"
    speed: float = 1.0


@dataclass
class TTSResult:
    output_path: Path
    duration_seconds: float
    sample_rate: int


class BaseTTSEngine(ABC):
    name: str

    def __init__(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @abstractmethod
    def synthesize(self, request: TTSRequest) -> TTSResult:
        ...
