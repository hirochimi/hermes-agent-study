"""Vosk STT Engine Implementation (offline, lightweight)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VoskEngine:
    """STT engine using Vosk (offline, lightweight, good for wake word + STT)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model: Optional[Any] = None
        self.recognizer: Optional[Any] = None
        self._model_loaded = False
        self.sample_rate = config.get("sample_rate", 16000)

    async def load_model(self) -> None:
        """Load the Vosk model asynchronously."""
        if self._model_loaded:
            return

        model_name = self.config.get("model", "vosk-model-small-ja-0.22")
        model_path = Path.home() / ".cache" / "vosk" / model_name

        logger.info(f"Loading Vosk model: {model_path}")

        def _load():
            from vosk import Model, KaldiRecognizer
            model = Model(str(model_path))
            recognizer = KaldiRecognizer(model, self.sample_rate)
            recognizer.SetWords(True)
            return model, recognizer

        self.model, self.recognizer = await asyncio.to_thread(_load)
        self._model_loaded = True
        logger.info("Vosk model loaded successfully")

    async def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        """Transcribe audio to text (Vosk works best with streaming)."""
        if not self._model_loaded:
            await self.load_model()

        # Vosk expects int16
        if audio.dtype != np.int16:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio

        def _transcribe():
            self.recognizer.AcceptWaveform(audio_int16.tobytes())
            result = json.loads(self.recognizer.FinalResult())
            return result.get("text", "").strip()

        text = await asyncio.to_thread(_transcribe)
        logger.debug(f"Transcribed: {text[:100]}...")
        return text

    async def transcribe_streaming(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Transcribe a single audio chunk (streaming mode)."""
        if not self._model_loaded:
            await self.load_model()

        if audio_chunk.dtype != np.int16:
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
        else:
            audio_int16 = audio_chunk

        def _process():
            if self.recognizer.AcceptWaveform(audio_int16.tobytes()):
                result = json.loads(self.recognizer.Result())
                return result.get("text", "").strip()
            else:
                partial = json.loads(self.recognizer.PartialResult())
                return partial.get("partial", "")

        return await asyncio.to_thread(_process)