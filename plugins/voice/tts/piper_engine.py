"""Piper TTS Engine Implementation (fast, lightweight, good quality)."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PiperEngine:
    """TTS engine using Piper (ONNX-based, fast, multi-speaker)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.voice_model: str = config.get("model", "ja_JP-kurokumo-medium")
        self.speaker_id: int = config.get("speaker_id", 0)
        self.length_scale: float = config.get("length_scale", 1.0)
        self.noise_scale: float = config.get("noise_scale", 0.667)
        self.noise_w: float = config.get("noise_w", 0.8)
        self._piper_path: Optional[Path] = None
        self._model_path: Optional[Path] = None
        self._config_path: Optional[Path] = None
        self._loaded = False

    async def load_model(self) -> None:
        """Ensure Piper binary and voice model are available."""
        if self._loaded:
            return

        # Check for piper binary
        self._piper_path = Path.home() / ".local" / "bin" / "piper"
        if not self._piper_path.exists():
            # Try system path
            import shutil
            piper_sys = shutil.which("piper")
            if piper_sys:
                self._piper_path = Path(piper_sys)
            else:
                raise RuntimeError("Piper binary not found. Install with: pip install piper-tts")

        # Model path (piper-tts downloads automatically on first use)
        # We just need to know the voice name
        self._loaded = True
        logger.info(f"Piper TTS ready with voice: {self.voice_model}")

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio (returns int16 numpy array)."""
        if not self._loaded:
            await self.load_model()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        try:
            cmd = [
                str(self._piper_path),
                "--model", self.voice_model,
                "--speaker", str(self.speaker_id),
                "--length_scale", str(self.length_scale),
                "--noise_scale", str(self.noise_scale),
                "--noise_w", str(self.noise_w),
                "--output_file", output_path,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(input=text.encode())

            if proc.returncode != 0:
                raise RuntimeError(f"Piper failed: {stderr.decode()}")

            # Read WAV file
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

    async def synthesize_streaming(self, text: str):
        """Synthesize text to streaming audio chunks (generator)."""
        # Piper doesn't natively stream, but we can chunk the output
        audio = await self.synthesize(text)
        chunk_size = 4096
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]