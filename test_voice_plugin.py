#!/usr/bin/env python3
"""Test script for voice plugin components."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_config_load():
    """Test loading voice configuration."""
    import hermes_yaml as yaml
    config_path = Path(__file__).parent / "config" / "voice.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print("✓ Config loaded successfully")
    print(f"  Memory budget: {config['memory_budget']}")
    print(f"  STT engine: {config['stt']['engine']}")
    print(f"  TTS engine: {config['tts']['engine']}")
    print(f"  Wakeword engine: {config['wakeword']['engine']}")
    print(f"  Transport: {config['transport']['protocol']}")
    return config

async def test_plugin_import():
    """Test importing the voice plugin."""
    try:
        from plugins.voice import load_voice_config, VoicePlugin
        print("✓ Voice plugin imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import voice plugin: {e}")
        return False

async def test_stt_engines():
    """Test STT engine imports."""
    engines = [
        ("faster-whisper", "plugins.voice.stt.faster_whisper_engine", "FasterWhisperEngine"),
        ("whisper.cpp", "plugins.voice.stt.whisper_cpp_engine", "WhisperCppEngine"),
        ("vosk", "plugins.voice.stt.vosk_engine", "VoskEngine"),
    ]
    for name, module, class_name in engines:
        try:
            mod = __import__(module, fromlist=[class_name])
            cls = getattr(mod, class_name)
            print(f"✓ {name} engine imported: {cls.__name__}")
        except Exception as e:
            print(f"⚠ {name} engine not available: {e}")

async def test_tts_engines():
    """Test TTS engine imports."""
    engines = [
        ("piper", "plugins.voice.tts.piper_engine", "PiperEngine"),
        ("coqui", "plugins.voice.tts.coqui_engine", "CoquiEngine"),
        ("styletts2", "plugins.voice.tts.styletts2_engine", "StyleTTS2Engine"),
    ]
    for name, module, class_name in engines:
        try:
            mod = __import__(module, fromlist=[class_name])
            cls = getattr(mod, class_name)
            print(f"✓ {name} engine imported: {cls.__name__}")
        except Exception as e:
            print(f"⚠ {name} engine not available: {e}")

async def test_wakeword_engines():
    """Test WakeWord engine imports."""
    engines = [
        ("openWakeWord", "plugins.voice.wakeword.openwakeword_engine", "OpenWakeWordEngine"),
        ("porcupine", "plugins.voice.wakeword.porcupine_engine", "PorcupineEngine"),
    ]
    for name, module, class_name in engines:
        try:
            mod = __import__(module, fromlist=[class_name])
            cls = getattr(mod, class_name)
            print(f"✓ {name} engine imported: {cls.__name__}")
        except Exception as e:
            print(f"⚠ {name} engine not available: {e}")

async def test_transport():
    """Test transport server imports."""
    try:
        from plugins.voice.transport.websocket_server import WebSocketTransportServer
        print(f"✓ WebSocket transport imported: {WebSocketTransportServer.__name__}")
    except Exception as e:
        print(f"⚠ WebSocket transport not available: {e}")

    try:
        from plugins.voice.transport.grpc_server import GrpcTransportServer
        print(f"✓ gRPC transport imported: {GrpcTransportServer.__name__}")
    except Exception as e:
        print(f"⚠ gRPC transport not available: {e}")

async def main():
    print("=" * 50)
    print("Voice Plugin Component Tests")
    print("=" * 50)

    await test_config_load()
    print()
    await test_plugin_import()
    print()
    await test_stt_engines()
    print()
    await test_tts_engines()
    print()
    await test_wakeword_engines()
    print()
    await test_transport()
    print()
    print("=" * 50)
    print("Tests completed")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())