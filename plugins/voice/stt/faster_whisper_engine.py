"""Faster-Whisper STT Engine Implementation."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FasterWhisperEngine:
    """STT engine using faster-whisper (CTranslate2 backend)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model: Optional[Any] = None
        self._model_loaded = False

    async def load_model(self) -> None:
        """Load the Whisper model asynchronously."""
        if self._model_loaded:
            return

        model_size = self.config.get("model", "base")
        device = self.config.get("device", "cpu")
        compute_type = self.config.get("compute_type", "int8")

        logger.info(f"Loading faster-whisper model: {model_size} on {device} ({compute_type})")

        def _load():
            from faster_whisper import WhisperModel
            return WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(Path.home() / ".cache" / "faster-whisper"),
            )

        self.model = await asyncio.to_thread(_load)
        self._model_loaded = True
        logger.info("faster-whisper model loaded successfully")

    async def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        """Transcribe audio to text."""
        if not self._model_loaded:
            await self.load_model()

        lang = language or self.config.get("language", "ja")
        beam_size = self.config.get("beam_size", 5)
        vad_filter = self.config.get("vad_filter", True)

        def _transcribe():
            segments, info = self.model.transcribe(
                audio,
                language=lang if lang != "auto" else None,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=self.config.get("vad_parameters", {}),
            )
            return " ".join(seg.text for seg in segments).strip()

        text = await asyncio.to_thread(_transcribe)
        logger.debug(f"Transcribed: {text[:100]}...")
        return text

    async def transcribe_streaming(self, audio_chunks: list[np.ndarray], language: Optional[str] = None) -> str:
        """Transcribe streaming audio chunks (concatenates first, then transcribes)."""
        # For true streaming, we'd need a different approach
        # This concatenates chunks for simplicity
        if not audio_chunks:
            return ""
        audio = np.concatenate(audio_chunks)
        return await self.transcribe(audio, language)