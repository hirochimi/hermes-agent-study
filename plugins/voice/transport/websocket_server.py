"""WebSocket Transport Server for Voice Pipeline."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from aiohttp import web
import websockets

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio streaming configuration."""
    sample_rate: int = 16000
    channels: int = 1
    format: str = "int16"
    chunk_size: int = 1024
    opus_bitrate: int = 32000
    opus_frame_duration: int = 20  # ms


class WebSocketTransportServer:
    """WebSocket server for bidirectional voice communication with clients (e.g., Android)."""

    def __init__(
        self,
        config: dict[str, Any],
        stt_engine: Any = None,
        tts_engine: Any = None,
        wakeword_engine: Any = None,
    ):
        self.config = config
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 8765)
        self.ssl_enabled = config.get("ssl_enabled", False)
        self.ssl_cert = config.get("ssl_cert_path")
        self.ssl_key = config.get("ssl_key_path")

        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.wakeword_engine = wakeword_engine

        self.audio_config = AudioConfig(
            sample_rate=config.get("sample_rate", 16000),
            opus_bitrate=config.get("opus_bitrate", 32000),
            opus_frame_duration=config.get("opus_frame_duration", 20),
        )

        self._server: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._clients: set[web.WebSocketResponse] = set()
        self._running = False

        # Opus encoder/decoder for efficient streaming
        self._opus_encoder = None
        self._opus_decoder = None

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            return

        # Initialize Opus
        await self._init_opus()

        app = web.Application()
        app.router.add_get("/ws", self._handle_websocket)
        app.router.add_get("/health", self._handle_health)

        self._server = web.AppRunner(app)
        await self._server.setup()

        ssl_context = None
        if self.ssl_enabled and self.ssl_cert and self.ssl_key:
            import ssl
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(self.ssl_cert, self.ssl_key)

        self._site = web.TCPSite(self._server, self.host, self.port, ssl_context=ssl_context)
        await self._site.start()

        self._running = True
        logger.info(f"WebSocket transport server started on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False

        # Close all client connections
        for ws in self._clients:
            await ws.close()
        self._clients.clear()

        if self._site:
            await self._site.stop()
        if self._server:
            await self._server.cleanup()

        logger.info("WebSocket transport server stopped")

    async def _init_opus(self) -> None:
        """Initialize Opus encoder/decoder."""
        try:
            import opuslib
            self._opus_encoder = opuslib.Encoder(
                self.audio_config.sample_rate,
                self.audio_config.channels,
                opuslib.APPLICATION_VOIP,
            )
            self._opus_encoder.bitrate = self.audio_config.opus_bitrate
            self._opus_decoder = opuslib.Decoder(
                self.audio_config.sample_rate,
                self.audio_config.channels,
            )
            logger.info("Opus codec initialized")
        except ImportError:
            logger.warning("opuslib not installed, using raw PCM")

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "ok",
            "clients": len(self._clients),
            "stt": self.stt_engine is not None,
            "tts": self.tts_engine is not None,
            "wakeword": self.wakeword_engine is not None,
        })

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._clients.add(ws)
        logger.info(f"Client connected: {request.remote}, total: {len(self._clients)}")

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    await self._handle_text_message(ws, msg.data)
                elif msg.type == web.WSMsgType.BINARY:
                    await self._handle_binary_message(ws, msg.data)
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            self._clients.discard(ws)
            logger.info(f"Client disconnected: {request.remote}, total: {len(self._clients)}")

        return ws

    async def _handle_text_message(self, ws: web.WebSocketResponse, data: str) -> None:
        """Handle text message (JSON commands)."""
        try:
            msg = json.loads(data)
            msg_type = msg.get("type")

            if msg_type == "config":
                await self._send_config(ws)
            elif msg_type == "stt":
                # Trigger STT on buffered audio
                audio_b64 = msg.get("audio")
                if audio_b64:
                    audio = self._decode_audio(audio_b64)
                    text = await self.stt_engine.transcribe(audio) if self.stt_engine else ""
                    await ws.send_json({"type": "stt_result", "text": text})
            elif msg_type == "tts":
                text = msg.get("text", "")
                if text and self.tts_engine:
                    audio = await self.tts_engine.synthesize(text)
                    audio_b64 = self._encode_audio(audio)
                    await ws.send_json({"type": "tts_audio", "audio": audio_b64})
            elif msg_type == "wakeword":
                # Configure wake word
                pass
            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client: {data[:100]}")
        except Exception as e:
            logger.error(f"Error handling text message: {e}")
            await ws.send_json({"type": "error", "message": str(e)})

    async def _handle_binary_message(self, ws: web.WebSocketResponse, data: bytes) -> None:
        """Handle binary audio data (Opus or PCM)."""
        try:
            # Decode audio
            if self._opus_decoder and len(data) > 2:
                # Assume Opus encoded
                try:
                    pcm = self._opus_decoder.decode(data, self.audio_config.chunk_size)
                    audio = np.frombuffer(pcm, dtype=np.int16)
                except Exception:
                    # Fallback to raw PCM
                    audio = np.frombuffer(data, dtype=np.int16)
            else:
                audio = np.frombuffer(data, dtype=np.int16)

            # Process wake word if enabled
            if self.wakeword_engine:
                detected = self.wakeword_engine.process_audio(audio)
                if detected:
                    await ws.send_json({"type": "wakeword", "keywords": detected})

            # Buffer for STT (could implement VAD here)
            # For now, just echo back for testing
            if self.tts_engine:
                # Echo test: convert to text and back
                pass

        except Exception as e:
            logger.error(f"Error handling binary message: {e}")

    async def _send_config(self, ws: web.WebSocketResponse) -> None:
        """Send current configuration to client."""
        await ws.send_json({
            "type": "config",
            "audio": {
                "sample_rate": self.audio_config.sample_rate,
                "channels": self.audio_config.channels,
                "format": self.audio_config.format,
                "chunk_size": self.audio_config.chunk_size,
            },
            "stt_enabled": self.stt_engine is not None,
            "tts_enabled": self.tts_engine is not None,
            "wakeword_enabled": self.wakeword_engine is not None,
        })

    def _encode_audio(self, audio: np.ndarray) -> str:
        """Encode audio to base64 (Opus if available, else PCM)."""
        if self._opus_encoder:
            # Convert to bytes
            if audio.dtype != np.int16:
                audio = (audio * 32767).astype(np.int16)
            pcm_bytes = audio.tobytes()
            # Encode with Opus
            opus_data = self._opus_encoder.encode(pcm_bytes, self.audio_config.chunk_size)
            return base64.b64encode(opus_data).decode()
        else:
            return base64.b64encode(audio.tobytes()).decode()

    def _decode_audio(self, audio_b64: str) -> np.ndarray:
        """Decode base64 audio to numpy array."""
        data = base64.b64decode(audio_b64)
        if self._opus_decoder:
            try:
                pcm = self._opus_decoder.decode(data, self.audio_config.chunk_size)
                return np.frombuffer(pcm, dtype=np.int16)
            except Exception:
                pass
        return np.frombuffer(data, dtype=np.int16)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        if not self._clients:
            return

        data = json.dumps(message)
        for ws in self._clients.copy():
            try:
                await ws.send_str(data)
            except Exception:
                self._clients.discard(ws)