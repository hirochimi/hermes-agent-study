"""Coqui TTS Engine Implementation (higher quality, more models)."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CoquiEngine:
    """TTS engine using Coqui TTS (supports many models, higher quality)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name: str = config.get("coqui_model", "tts_models/ja/kokoro/tacotron2-DDC")
        self.tts: Optional[Any] = None
        self._loaded = False

    async def load_model(self) -> None:
        """Load the Coqui TTS model."""
        if self._loaded:
            return

        logger.info(f"Loading Coqui TTS model: {self.model_name}")

        def _load():
            from TTS.api import TTS
            return TTS(self.model_name, progress_bar=False)

        self.tts = await asyncio.to_thread(_load)
        self._loaded = True
        logger.info("Coqui TTS model loaded successfully")

    async def synthesize(self, text: str, speaker: Optional[str] = None) -> np.ndarray:
        """Synthesize text to audio."""
        if not self._loaded:
            await self.load_model()

        def _synthesize():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name

            try:
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker=speaker,
                    language="ja",
                )

                import wave
                with wave.open(output_path, "rb") as wav:
                    frames = wav.readframes(wav.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16)

                return audio

            finally:
                try:
                    Path(output_path).unlink(missing_ok=True)
                except Exception:
                    pass

        audio = await asyncio.to_thread(_synthesize)
        logger.debug(f"Synthesized {len(audio)} samples")
        return audio