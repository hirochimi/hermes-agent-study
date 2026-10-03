import asyncio
import time
import numpy as np
import json
import os
from pathlib import Path
from scipy.io import wavfile
from scipy import signal

# STT engines
from plugins.voice.stt.faster_whisper_engine import FasterWhisperEngine
from plugins.voice.stt.whisper_cpp_engine import WhisperCppEngine
# from plugins.voice.stt.vosk_engine import VoskEngine  # skip due to model dependency

# TTS engines
from plugins.voice.tts.piper_engine import PiperEngine
# from plugins.voice.tts.coqui_engine import CoquiEngine
# from plugins.voice.tts.styletts2_engine import StyleTTS2Engine

TEST_DATA = Path('/opt/l-llm/voice-IO-hub/bench/test_data')
SAMPLES = sorted(TEST_DATA.glob('*.wav'))[:5]  # first 5 wav files
TARGET_SR = 16000

def load_and_resample_wav(path):
    sr, data = wavfile.read(path)
    # Convert to float32 [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        pass
    else:
        raise ValueError(f'Unsupported wav dtype: {data.dtype}')
    # If stereo, take first channel
    if data.ndim == 2:
        data = data[:, 0]
    # Resample if needed
    if sr != TARGET_SR:
        num_samples = int(len(data) * TARGET_SR / sr)
        data = signal.resample_poly(data, TARGET_SR, sr)
    return data.astype(np.float32)

async def test_stt():
    print('=== STT Speed Test ===')
    engines = {
        'faster-whisper': FasterWhisperEngine({'model':'tiny','device':'cpu','compute_type':'int8','language':'ja'}),
        'whisper-cpp': WhisperCppEngine({'model':'tiny','language':'ja'}),
    }
    for name, eng in engines.items():
        try:
            print(f'  Loading {name}...')
            await eng.load_model()
            print(f'  {name} loaded.')
        except Exception as e:
            print(f'  {name}: load failed: {e}')
            continue
        times = []
        for wav_path in SAMPLES:
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
            print(f'    {name} {wav_path.name}: {elapsed:.3f}s -> {text[:30]}')
        if times:
            avg = sum(times)/len(times)
            print(f'  {name} average over {len(times)} samples: {avg:.3f}s')

async def test_tts():
    print('=== TTS Speed Test ===')
    engines = {
        'piper': PiperEngine({'model':'ja_JP-naist-neutral-medium','speaker_id':0}),
    }
    for name, eng in engines.items():
        try:
            print(f'  Loading {name}...')
            await eng.load_model()
            print(f'  {name} loaded.')
        except Exception as e:
            print(f'  {name}: load failed: {e}')
            continue
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
                # audio is numpy array
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                print(f'    {name} "{txt[:10]}...": {elapsed:.3f}s, audio shape {getattr(audio,"shape","?")}')
            except Exception as e:
                print(f'    {name} "{txt[:10]}...": ERROR {e}')
        if times:
            avg = sum(times)/len(times)
            print(f'  {name} average over {len(times)} samples: {avg:.3f}s')

async def main():
    await test_stt()
    await test_tts()

if __name__ == '__main__':
    asyncio.run(main())
