import sys
sys.path.insert(0, '.')
print('After sys.path', flush=True)
from plugins.voice.wakeword.openwakeword_engine import OpenWakeWordEngine
print('After import', flush=True)
import openwakeword
print('After openwakeword import', flush=True)
import numpy as np
print('After numpy import', flush=True)
import asyncio
print('After asyncio import', flush=True)

paths = openwakeword.get_pretrained_model_paths()
hey_jarvis_path = None
for p in paths:
    if 'hey_jarvis' in p:
        hey_jarvis_path = p
        break
print('Model path:', hey_jarvis_path, flush=True)
config = {'openwakeword_model': hey_jarvis_path, 'sensitivity': 0.5, 'keywords': ['エルメス', 'hermes']}
print('About to create engine', flush=True)
engine = OpenWakeWordEngine(config)
print('Engine created', flush=True)
print('About to start engine', flush=True)
async def test():
    await engine.start()
    print('Engine started', flush=True)
    print('About to process audio', flush=True)
    audio = np.zeros(1600, dtype=np.int16)
    audio_float = audio.astype(np.float32) / 32768.0
    result = engine.process_audio(audio_float)
    print('Processed audio result:', result, flush=True)
    print('About to stop engine', flush=True)
    await engine.stop()
    print('Engine stopped', flush=True)
print('About to run async test', flush=True)
asyncio.run(test())
print('Test completed', flush=True)
