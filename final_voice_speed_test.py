import asyncio
import time
import numpy as np
import json
from pathlib import Path
from scipy.io import wavfile
from scipy import signal

# STT
from plugins.voice.stt.faster_whisper_engine import FasterWhisperEngine
# TTS
from plugins.voice.tts.piper_engine import PiperEngine
# WakeWord
from plugins.voice.wakeword.openwakeword_engine import OpenWakeWordEngine
import openwakeword

TEST_DATA = Path('/opt/l-llm/voice-IO-hub/bench/test_data')
SAMPLES_STT = sorted(TEST_DATA.glob('*.wav'))[:3]  # first 3 wav files for speed
TARGET_SR = 16000

def load_and_resample_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        pass
    else:
        raise ValueError(f'Unsupported wav dtype: {data.dtype}')
    if data.ndim == 2:
        data = data[:, 0]
    if sr != TARGET_SR:
        num_samples = int(len(data) * TARGET_SR / sr)
        data = signal.resample_poly(data, TARGET_SR, sr)
    return data.astype(np.float32)

async def test_stt():
    print('=== STT Speed Test (faster-whisper tiny) ===')
    eng = FasterWhisperEngine({'model':'tiny','device':'cpu','compute_type':'int8','language':'ja'})
    await eng.load_model()
    times = []
    for wav_path in SAMPLES_STT:
        try:
            audio = load_and_resample_wav(wav_path)
        except Exception as e:
            print(f'    Failed to load {wav_path.name}: {e}')
            continue
        start = time.perf_counter()
        try:
            text = await eng.transcribe(audio)
        except Exception as e:
            text = f'ERROR: {e}'
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f'    {wav_path.name}: {elapsed:.3f}s -> {text[:30]}')
    if times:
        avg = sum(times)/len(times)
        print(f'  Average over {len(times)} samples: {avg:.3f}s')
        # compute average audio duration
        durations = []
        for wav_path in SAMPLES_STT:
            sr, data = wavfile.read(wav_path)
            dur = len(data) / sr
            durations.append(dur)
        avg_dur = sum(durations)/len(durations)
        print(f'  Average audio duration: {avg_dur:.3f}s')
        print(f'  Real-time factor: {avg_dur/avg:.2f}x')

async def test_tts():
    print('=== TTS Speed Test (Piper ja_JP-hi_fi_captain-medium) ===')
    model_path = '/tmp/piper_voices/ja_JP-hi_fi_captain-medium.onnx'
    eng = PiperEngine({'model': model_path, 'speaker_id': 0})
    await eng.load_model()
    # Get some Japanese text from json files
    texts = []
    for json_path in TEST_DATA.glob('*.json'):
        try:
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
                if 'text' in data and isinstance(data['text'], str):
                    texts.append(data['text'])
                    if len(texts) >= 3:
                        break
        except:
            pass
    if not texts:
        texts = ['こんにちは', 'テストです', '日本語の音声合成']
    times = []
    for txt in texts:
        start = time.perf_counter()
        try:
            audio = await eng.synthesize(txt)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f'    \"{txt[:10]}...\": {elapsed:.3f}s, audio samples {len(audio)}')
        except Exception as e:
            print(f'    \"{txt[:10]}...\": ERROR {e}')
    if times:
        avg = sum(times)/len(times)
        print(f'  Average over {len(times)} samples: {avg:.3f}s')

async def test_wakeword():
    print('=== WakeWord Processing Speed (OpenWakeWord hey_jarvis) ===')
    paths = openwakeword.get_pretrained_model_paths()
    hey_jarvis_path = None
    for p in paths:
        if 'hey_jarvis' in p:
            hey_jarvis_path = p
            break
    if hey_jarvis_path is None:
        print('  Could not find hey_jarvis model')
        return
    eng = OpenWakeWordEngine({'openwakeword_model': hey_jarvis_path, 'sensitivity': 0.5, 'keywords': ['エルメス', 'hermes']})
    await eng.start()
    # Process dummy audio frames of 1280 samples at 16kHz (0.08s)
    frame = np.zeros(1280, dtype=np.float32)
    times = []
    loop = asyncio.get_event_loop()
    for i in range(100):  # more iterations for stable avg
        start = time.perf_counter()
        result = await loop.run_in_executor(None, eng.process_audio, frame)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    await eng.stop()
    if times:
        avg = sum(times)/len(times)
        print(f'  Average processing time per frame (1280 samples @ 16kHz): {avg*1000:.2f} ms')
        print(f'  Corresponding audio duration per frame: {1280/16000*1000:.2f} ms')
        print(f'  Real-time factor: {(1280/16000)/avg:.2f}x')

async def main():
    await test_stt()
    await test_tts()
    await test_wakeword()

if __name__ == '__main__':
    asyncio.run(main())
