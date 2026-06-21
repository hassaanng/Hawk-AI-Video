"""
Piper TTS backend.

Piper (Rhasspy/OHF) is a fast, small, CPU-friendly neural TTS engine
distributed as ONNX voice models (rhasspy/piper-voices on HuggingFace).
It trades some naturalness versus XTTS-v2 for being dramatically
lighter and fast enough to run on CPU-only machines — useful when the
GPU is fully occupied by video generation and narration needs to be
synthesized concurrently without contending for VRAM.
"""
from __future__ import annotations

import json
import logging
import subprocess
import uuid
import wave
from pathlib import Path

from app.core.config import settings
from app.services.tts_base import BaseTTSEngine, TTSRequest, TTSResult

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "en_US-lessac-medium"
VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


class PiperEngine(BaseTTSEngine):
    name = "piper"

    def load(self) -> None:
        # Piper is a CLI/ONNX-runtime tool, not a torch module held in
        # memory the way the video models are — "loading" here just means
        # making sure the requested voice files exist locally, downloading
        # them on first use.
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def _ensure_voice(self, voice: str) -> tuple[Path, Path]:
        """
        Voice naming convention follows piper-voices' own layout, e.g.
        'en_US-lessac-medium' -> en/en_US/lessac/medium/en_US-lessac-medium.onnx
        Downloads both the .onnx model and its .json config on first use,
        then caches them under settings.piper_voices_dir permanently.
        """
        lang, name, quality = voice.split("-")
        lang_short = lang.split("_")[0]
        rel_dir = f"{lang_short}/{lang}/{name}/{quality}"
        onnx_path = settings.piper_voices_dir / f"{voice}.onnx"
        json_path = settings.piper_voices_dir / f"{voice}.onnx.json"

        if not onnx_path.exists():
            url = f"{VOICE_BASE_URL}/{rel_dir}/{voice}.onnx"
            logger.info("Downloading Piper voice model: %s", url)
            subprocess.run(["wget", "-q", "-O", str(onnx_path), url], check=True)
        if not json_path.exists():
            url = f"{VOICE_BASE_URL}/{rel_dir}/{voice}.onnx.json"
            subprocess.run(["wget", "-q", "-O", str(json_path), url], check=True)

        return onnx_path, json_path

    def synthesize(self, request: TTSRequest) -> TTSResult:
        self.load()
        voice = request.voice or DEFAULT_VOICE
        onnx_path, json_path = self._ensure_voice(voice)

        out_path = settings.temp_dir / f"piper_{uuid.uuid4().hex}.wav"

        cmd = [
            "piper",
            "--model", str(onnx_path),
            "--config", str(json_path),
            "--output_file", str(out_path),
            "--length_scale", str(1.0 / max(request.speed, 0.1)),
        ]
        proc = subprocess.run(cmd, input=request.text.encode("utf-8"), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Piper TTS failed: {proc.stderr.decode(errors='ignore')}")

        with wave.open(str(out_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)

        return TTSResult(output_path=out_path, duration_seconds=duration, sample_rate=rate)
