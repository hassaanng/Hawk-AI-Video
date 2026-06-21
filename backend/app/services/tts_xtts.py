"""
Coqui XTTS-v2 backend.

XTTS-v2 ("coqui/XTTS-v2" via the `TTS` package, also mirrored as
"coqui/XTTS-v2" on HuggingFace) is a real multi-speaker, multilingual,
zero-shot voice-cloning TTS model. It is the higher-quality, heavier of
the two TTS options wired up here — runs comfortably on GPU, runnable
(slower) on CPU as well, unlike the video models.
"""
from __future__ import annotations

import gc
import logging
import uuid

from app.core.config import settings
from app.services.tts_base import BaseTTSEngine, TTSRequest, TTSResult

logger = logging.getLogger(__name__)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# A small set of built-in neutral reference voices shipped with the
# model itself (Coqui's package bundles short reference clips per
# speaker). If the user supplies their own reference_audio_path for
# cloning, that takes priority over these named presets.
BUILTIN_SPEAKERS = ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Andrew Chipper"]


class XTTSEngine(BaseTTSEngine):
    name = "xtts"

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from TTS.api import TTS

        device = "cuda" if torch.cuda.is_available() and not settings.force_cpu else "cpu"
        logger.info("Loading XTTS-v2 on %s ...", device)
        self._tts = TTS(MODEL_NAME).to(device)
        self._loaded = True
        logger.info("XTTS-v2 loaded.")

    def unload(self) -> None:
        if not self._loaded:
            return
        import torch

        del self._tts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    def synthesize(self, request: TTSRequest) -> TTSResult:
        self.load()
        out_path = settings.temp_dir / f"xtts_{uuid.uuid4().hex}.wav"

        speaker_wav = str(request.reference_audio_path) if request.reference_audio_path else None
        speaker_name = request.voice if (request.voice in BUILTIN_SPEAKERS and not speaker_wav) else None
        if not speaker_wav and not speaker_name:
            speaker_name = BUILTIN_SPEAKERS[0]

        self._tts.tts_to_file(
            text=request.text,
            file_path=str(out_path),
            speaker_wav=speaker_wav,
            speaker=speaker_name,
            language=request.language,
            speed=request.speed,
        )

        import soundfile as sf
        info = sf.info(str(out_path))

        return TTSResult(output_path=out_path, duration_seconds=info.duration, sample_rate=info.samplerate)
