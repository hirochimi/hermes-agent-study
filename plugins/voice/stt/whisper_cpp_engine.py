"""Whisper.cpp STT Engine Implementation."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class WhisperCppEngine:
    """STT engine using whisper.cpp via Python bindings."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model: Optional[Any] = None
        self._model_loaded = False

    async def load_model(self) -> None:
        """Load the Whisper model asynchronously."""
        if self._model_loaded:
            return

        model_name = self.config.get("model", "base")
        # whisper.cpp uses .bin model files
        model_path = Path.home() / ".cache" / "whisper.cpp" / f"ggml-{model_name}.bin"

        logger.info(f"Loading whisper.cpp model: {model_path}")

        def _load():
            from whisper_cpp_python import Whisper
            return Whisper(str(model_path))

        self.model = await asyncio.to_thread(_load)
        self._model_loaded = True
        logger.info("whisper.cpp model loaded successfully")

    async def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        """Transcribe audio to text."""
        if not self._model_loaded:
            await self.load_model()

        lang = language or self.config.get("language", "ja")

        def _transcribe():
            # whisper.cpp expects float32 in [-1, 1]
            if audio.dtype != np.float32:
                audio_float = audio.astype(np.float32) / 32768.0
            else:
                audio_float = audio
            return self.model.transcribe(audio_float, language=lang if lang != "auto" else None)

        text = await asyncio.to_thread(_transcribe)
        logger.debug(f"Transcribed: {text[:100]}...")
        return text