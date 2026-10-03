"""Porcupine Wake Word Engine Implementation (commercial, very accurate)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PorcupineEngine:
    """Wake word detection using Picovoice Porcupine (commercial, free tier available)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.access_key: Optional[str] = config.get("porcupine_access_key")
        self.keywords: list[str] = config.get("keywords", ["エルメス", "hermes"])
        self.sensitivities: list[float] = config.get("sensitivities", [0.5] * len(self.keywords))
        self.handle: Optional[Any] = None
        self._running = False
        self._callback: Optional[Callable[[str], None]] = None
        self._task: Optional[asyncio.Task] = None
        self.frame_length: int = 512  # Porcupine default
        self.sample_rate: int = 16000

    def set_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function to call when wake word detected."""
        self._callback = callback

    async def start(self) -> None:
        """Start the Porcupine engine."""
        if self._running:
            return

        if not self.access_key:
            raise ValueError("Porcupine access key required. Get free key at https://console.picovoice.ai/")

        logger.info(f"Starting Porcupine with keywords: {self.keywords}")

        def _init():
            import pvporcupine
            # Build keyword file paths for custom keywords
            # For built-in keywords, use pvporcupine.KEYWORDS
            keyword_paths = []
            for kw in self.keywords:
                if kw in pvporcupine.KEYWORDS:
                    keyword_paths.append(kw)
                else:
                    # Custom keyword .ppn file path
                    keyword_paths.append(str(Path.home() / ".cache" / "porcupine" / f"{kw}.ppn"))

            return pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=keyword_paths,
                sensitivities=self.sensitivities,
            )

        self.handle = await asyncio.to_thread(_init)
        self.frame_length = self.handle.frame_length
        self.sample_rate = self.handle.sample_rate
        self._running = True
        logger.info("Porcupine started")

    async def stop(self) -> None:
        """Stop the Porcupine engine."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.handle:
            await asyncio.to_thread(self.handle.delete)
            self.handle = None
        logger.info("Porcupine stopped")

    def process_audio(self, audio_chunk: np.ndarray) -> list[str]:
        """Process audio chunk and return detected wake words."""
        if not self._running or self.handle is None:
            return []

        # Porcupine expects int16
        if audio_chunk.dtype != np.int16:
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
        else:
            audio_int16 = audio_chunk

        # Process in frames
        detected = []
        for i in range(0, len(audio_int16), self.frame_length):
            frame = audio_int16[i:i + self.frame_length]
            if len(frame) < self.frame_length:
                break

            def _process():
                return self.handle.process(frame)

            keyword_index = asyncio.run(asyncio.to_thread(_process))

            if keyword_index >= 0:
                keyword = self.keywords[keyword_index]
                detected.append(keyword)
                logger.info(f"Wake word detected: {keyword}")
                if self._callback:
                    self._callback(keyword)

        return detected

    async def run_continuous(self, audio_stream) -> None:
        """Run continuous wake word detection from audio stream."""
        self._task = asyncio.current_task()
        try:
            async for chunk in audio_stream:
                if not self._running:
                    break
                self.process_audio(chunk)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass