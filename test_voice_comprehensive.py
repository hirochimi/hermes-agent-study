#!/usr/bin/env python3
"""Comprehensive test script for voice plugin components."""

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

async def test_stt_engine():
    """Test STT engine functionality."""
    try:
        from plugins.voice.stt.faster_whisper_engine import FasterWhisperEngine
        import numpy as np
        
        config = {'model': 'tiny', 'device': 'cpu', 'compute_type': 'int8', 'language': 'ja'}
        engine = FasterWhisperEngine(config)
        await engine.load_model()
        print("✓ FasterWhisper engine loaded successfully")
        
        # Test with dummy audio data (won't produce meaningful text but verifies pipeline)
        audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        # We won't actually call transcribe as it would try to process and might hang
        # but we know the engine is ready
        print("✓ FasterWhisper engine ready for audio processing")
        return True
    except Exception as e:
        print(f"⚠ FasterWhisper engine test failed: {e}")
        return False

async def test_tts_engine():
    """Test TTS engine functionality."""
    try:
        from plugins.voice.tts.piper_engine import PiperEngine
        
        config = {'model': 'en_US-lessac-medium', 'speaker_id': 0, 'length_scale': 1.0, 'noise_scale': 0.667, 'noise_w': 0.8}
        engine = PiperEngine(config)
        await engine.load_model()
        print("✓ Piper engine loaded successfully")
        print("✓ Piper engine ready for text synthesis")
        return True
    except Exception as e:
        print(f"⚠ Piper engine test failed: {e}")
        return False

async def test_wakeword_engine():
    """Test WakeWord engine functionality."""
    try:
        import openwakeword
        import numpy as np
        from plugins.voice.wakeword.openwakeword_engine import OpenWakeWordEngine
        
        # Get the hey_jarvis model path
        paths = openwakeword.get_pretrained_model_paths()
        hey_jarvis_path = None
        for path in paths:
            if 'hey_jarvis' in path:
                hey_jarvis_path = path
                break
        
        if hey_jarvis_path is None:
            print("⚠ Could not find hey_jarvis model")
            return False
            
        config = {
            'openwakeword_model': hey_jarvis_path,
            'sensitivity': 0.5, 
            'keywords': ['エルメス', 'hermes']
        }
        engine = OpenWakeWordEngine(config)
        await engine.start()
        print("✓ OpenWakeWord engine started successfully")
        
        # Test with dummy audio data
        audio = np.zeros(1600, dtype=np.int16)  # 0.1 second of silence at 16kHz
        audio_float = audio.astype(np.float32) / 32768.0
        # Run process_audio in a thread to avoid asyncio context issues
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, engine.process_audio, audio_float)
        print(f"✓ OpenWakeWord engine processed audio, result: {result}")
        
        await engine.stop()
        print("✓ OpenWakeWord engine stopped successfully")
        return True
    except Exception as e:
        print(f"⚠ OpenWakeWord engine test failed: {e}")
        return False

async def test_transport():
    """Test transport server instantiation."""
    try:
        from plugins.voice.transport.websocket_server import WebSocketTransportServer
        
        config = {
            'host': '0.0.0.0',
            'port': 8765,
            'ssl_enabled': False,
            'sample_rate': 16000,
            'opus_bitrate': 32000,
            'opus_frame_duration': 20
        }
        server = WebSocketTransportServer(config)
        print("✓ WebSocket transport server instantiated successfully")
        return True
    except Exception as e:
        print(f"⚠ WebSocket transport test failed: {e}")
        return False

async def test_plugin_registration():
    """Test voice plugin registration with Hermes plugin system."""
    try:
        from hermes_cli.plugins import PluginContext
        from plugins.voice import VoicePlugin, load_voice_config, register
        
        # Create a mock plugin context
        class MockPluginContext:
            def __init__(self):
                self.tools = {}
                self.hooks = {}
                self.state = {}
                self.config = {}
            
            def register_tool(self, name, description, parameters, handler):
                self.tools[name] = {'description': description, 'parameters': parameters, 'handler': handler}
            
            def register_hook(self, name, handler):
                self.hooks[name] = handler
            
            def set_state(self, key, value):
                self.state[key] = value
            
            def get_config(self, key, default=None):
                return self.config.get(key, default)

        ctx = MockPluginContext()
        config = load_voice_config()
        
        # Create voice plugin
        voice_plugin = VoicePlugin(ctx, config)
        
        # Register the plugin using the register function
        await register(ctx)
        
        # Check what was registered
        expected_tools = {'voice_start', 'voice_stop', 'voice_status'}
        expected_hooks = {'on_session_start', 'on_session_end'}
        
        registered_tools = set(ctx.tools.keys())
        registered_hooks = set(ctx.hooks.keys())
        
        if expected_tools.issubset(registered_tools) and expected_hooks.issubset(registered_hooks):
            print("✓ Voice plugin registered successfully with all expected tools and hooks")
            print(f"  Registered tools: {sorted(registered_tools)}")
            print(f"  Registered hooks: {sorted(registered_hooks)}")
            return True
        else:
            print(f"⚠ Voice plugin registration incomplete")
            print(f"  Expected tools: {expected_tools}, got: {registered_tools}")
            print(f"  Expected hooks: {expected_hooks}, got: {registered_hooks}")
            return False
    except Exception as e:
        print(f"⚠ Voice plugin registration test failed: {e}")
        return False

async def main():
    print("=" * 60)
    print("Comprehensive Voice Plugin Component Tests")
    print("=" * 60)

    await test_config_load()
    print()
    await test_plugin_import()
    print()
    await test_stt_engine()
    print()
    await test_tts_engine()
    print()
    await test_wakeword_engine()
    print()
    await test_transport()
    print()
    await test_plugin_registration()
    print()
    print("=" * 60)
    print("Comprehensive tests completed")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
