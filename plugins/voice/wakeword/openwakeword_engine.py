"""OpenWakeWord Wake Word Engine Implementation."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OpenWakeWordEngine:
    """Wake word detection using openWakeWord (open source, ONNX-based)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name: str = config.get("openwakeword_model", "hey_jarvis")
        self.sensitivity: float = config.get("sensitivity", 0.5)
        self.keywords: list[str] = config.get("keywords", ["エルメス", "hermes"])
        self.model: Optional[Any] = None
        self._running = False
        self._audio_buffer = deque(maxlen=16000 * 2)  # 2 seconds at 16kHz
        self._callback: Optional[Callable[[str], None]] = None
        self._task: Optional[asyncio.Task] = None

    def set_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function to call when wake word detected."""
        self._callback = callback

    async def start(self) -> None:
        """Start the wake word engine."""
        if self._running:
            return

        logger.info(f"Starting openWakeWord with model: {self.model_name}")

        def _load():
            from openwakeword import Model
            # Resolve model path if needed
            model_path = self.model_name
            if not model_path.startswith("/") and not model_path.startswith("."):
                import openwakeword
                paths = openwakeword.get_pretrained_model_paths()
                for path in paths:
                    if self.model_name in path:
                        model_path = path
                        break
            return Model(wakeword_model_paths=[model_path])

        self.model = await asyncio.to_thread(_load)
        self._running = True
        logger.info("openWakeWord started")

    async def stop(self) -> None:
        """Stop the wake word engine."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.model = None
        logger.info("openWakeWord stopped")

    def process_audio(self, audio_chunk: np.ndarray) -> list[str]:
        """Process audio chunk and return detected wake words."""
        if not self._running or self.model is None:
            return []

        # openWakeWord expects float32 in [-1, 1]
        if audio_chunk.dtype != np.float32:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk

        # Add to buffer
        self._audio_buffer.extend(audio_float)

        # Process when we have enough audio (openWakeWord uses 1280 samples per frame)
        detected = []
        if len(self._audio_buffer) >= 1280:
            frame = np.array(list(self._audio_buffer)[-1280:], dtype=np.float32)

            def _predict():
                return self.model.predict(frame)

            # Use asyncio.to_thread since we might be called from async context
            try:
                # If we're already in an event loop, use to_thread
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    prediction = asyncio.run_coroutine_threadsafe(
                        asyncio.to_thread(_predict), loop
                    ).result()
                else:
                    prediction = asyncio.run(asyncio.to_thread(_predict))
            except RuntimeError:
                # Fallback to direct call if we can't manage the loop
                prediction = _predict()

            # Check predictions for our keywords
            for keyword in self.keywords:
                # Map our keywords to model outputs
                model_key = self._map_keyword(keyword)
                if model_key in prediction and prediction[model_key] >= self.sensitivity:
                    detected.append(keyword)
                    logger.info(f"Wake word detected: {keyword} (score: {prediction[model_key]:.3f})")
                    if self._callback:
                        self._callback(keyword)

        return detected

    def _map_keyword(self, keyword: str) -> str:
        """Map our keyword to openWakeWord model output key."""
        # openWakeWord models typically have keys like "hey_jarvis", "hey_mycroft", etc.
        keyword_lower = keyword.lower()
        if "hermes" in keyword_lower or "エルメス" in keyword:
            return "hey_jarvis"  # Default model, customize as needed
        return keyword_lower.replace(" ", "_")

    async def run_continuous(self, audio_stream) -> None:
        """Run continuous wake word detection from audio stream."""
        self._task = asyncio.current_task()
        try:
            async for chunk in audio_stream:
                if not self._running:
                    break
                self.process_audio(chunk)
                await asyncio.sleep(0.01)  # Small yield
        except asyncio.CancelledError:
            pass
