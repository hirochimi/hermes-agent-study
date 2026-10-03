"""Voice Pipeline Plugin for Hermes Agent.

This plugin adds voice capabilities (STT, TTS, Wake Word, Transport) to Hermes Agent
via the plugin system. It follows the adapter pattern for swappable backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli.plugins import PluginContext


@dataclass
class VoiceConfig:
    """Voice pipeline configuration loaded from voice.yaml"""
    memory_budget: dict[str, int]
    audio: dict[str, Any]
    wakeword: dict[str, Any]
    stt: dict[str, Any]
    tts: dict[str, Any]
    transport: dict[str, Any]
    llm: dict[str, Any]
    decision_ai: dict[str, Any]
    logging: dict[str, Any]


class VoicePlugin:
    """Main voice plugin class managing all voice sub-components."""

    def __init__(self, ctx: PluginContext, config: VoiceConfig):
        self.ctx = ctx
        self.config = config
        self.stt_engine: Optional[Any] = None
        self.tts_engine: Optional[Any] = None
        self.wakeword_engine: Optional[Any] = None
        self.transport_server: Optional[Any] = None
        self._running = False

    async def start(self) -> None:
        """Initialize and start all voice components."""
        if self.config.wakeword.get("enabled", True):
            await self._init_wakeword()
        if self.config.stt.get("enabled", True):
            await self._init_stt()
        if self.config.tts.get("enabled", True):
            await self._init_tts()
        if self.config.transport.get("enabled", True):
            await self._init_transport()
        self._running = True

    async def stop(self) -> None:
        """Stop all voice components."""
        self._running = False
        if self.transport_server:
            await self.transport_server.stop()
        if self.wakeword_engine:
            await self.wakeword_engine.stop()
        # STT/TTS are stateless, no stop needed

    async def _init_wakeword(self) -> None:
        """Initialize wake word detection engine."""
        engine = self.config.wakeword.get("engine", "openwakeword")
        if engine == "openwakeword":
            from .wakeword.openwakeword_engine import OpenWakeWordEngine
            self.wakeword_engine = OpenWakeWordEngine(self.config.wakeword)
        elif engine == "porcupine":
            from .wakeword.porcupine_engine import PorcupineEngine
            self.wakeword_engine = PorcupineEngine(self.config.wakeword)
        else:
            raise ValueError(f"Unknown wakeword engine: {engine}")
        await self.wakeword_engine.start()

    async def _init_stt(self) -> None:
        """Initialize STT engine."""
        engine = self.config.stt.get("engine", "faster-whisper")
        if engine == "faster-whisper":
            from .stt.faster_whisper_engine import FasterWhisperEngine
            self.stt_engine = FasterWhisperEngine(self.config.stt)
        elif engine == "whisper.cpp":
            from .stt.whisper_cpp_engine import WhisperCppEngine
            self.stt_engine = WhisperCppEngine(self.config.stt)
        elif engine == "vosk":
            from .stt.vosk_engine import VoskEngine
            self.stt_engine = VoskEngine(self.config.stt)
        else:
            raise ValueError(f"Unknown STT engine: {engine}")
        await self.stt_engine.load_model()

    async def _init_tts(self) -> None:
        """Initialize TTS engine."""
        engine = self.config.tts.get("engine", "piper")
        if engine == "piper":
            from .tts.piper_engine import PiperEngine
            self.tts_engine = PiperEngine(self.config.tts)
        elif engine == "coqui":
            from .tts.coqui_engine import CoquiEngine
            self.tts_engine = CoquiEngine(self.config.tts)
        elif engine == "styletts2":
            from .tts.styletts2_engine import StyleTTS2Engine
            self.tts_engine = StyleTTS2Engine(self.config.tts)
        else:
            raise ValueError(f"Unknown TTS engine: {engine}")
        await self.tts_engine.load_model()

    async def _init_transport(self) -> None:
        """Initialize transport server for client communication."""
        protocol = self.config.transport.get("protocol", "websocket")
        if protocol == "websocket":
            from .transport.websocket_server import WebSocketTransportServer
            self.transport_server = WebSocketTransportServer(
                self.config.transport,
                self.stt_engine,
                self.tts_engine,
                self.wakeword_engine,
            )
        elif protocol == "grpc":
            from .transport.grpc_server import GrpcTransportServer
            self.transport_server = GrpcTransportServer(
                self.config.transport,
                self.stt_engine,
                self.tts_engine,
                self.wakeword_engine,
            )
        else:
            raise ValueError(f"Unknown transport protocol: {protocol}")
        await self.transport_server.start()


def load_voice_config(config_path: Optional[Path] = None) -> VoiceConfig:
    """Load voice configuration from YAML file."""
    import hermes_yaml as yaml

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "voice.yaml"

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return VoiceConfig(**data)


async def register(ctx: PluginContext) -> None:
    """Plugin entry point - called by Hermes plugin system."""
    # Load configuration
    config = load_voice_config()

    # Create and start voice plugin
    voice_plugin = VoicePlugin(ctx, config)

    # Register tools for voice control
    ctx.register_tool(
        name="voice_start",
        description="Start the voice pipeline",
        parameters={"type": "object", "properties": {}},
        handler=lambda _: voice_plugin.start(),
    )
    ctx.register_tool(
        name="voice_stop",
        description="Stop the voice pipeline",
        parameters={"type": "object", "properties": {}},
        handler=lambda _: voice_plugin.stop(),
    )
    ctx.register_tool(
        name="voice_status",
        description="Get voice pipeline status",
        parameters={"type": "object", "properties": {}},
        handler=lambda _: {
            "running": voice_plugin._running,
            "wakeword": voice_plugin.wakeword_engine is not None,
            "stt": voice_plugin.stt_engine is not None,
            "tts": voice_plugin.tts_engine is not None,
            "transport": voice_plugin.transport_server is not None,
        },
    )

    # Register hook for agent lifecycle
    async def on_session_start(**kwargs):
        if not voice_plugin._running:
            await voice_plugin.start()

    async def on_session_end(**kwargs):
        if voice_plugin._running:
            await voice_plugin.stop()

    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_end", on_session_end)

    # Store reference for other plugins/tools to access
    ctx.set_state("voice_plugin", voice_plugin)

    # Auto-start if configured
    if ctx.get_config("voice.auto_start", True):
        await voice_plugin.start()