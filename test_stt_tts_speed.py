import asyncio
import time
import numpy as np
import soundfile as sf
import json
import os
from pathlib import Path

# STT
from plugins.voice.stt.faster_whisper_engine import FasterWhisperEngine
from plugins.voice.stt.whisper_cpp_engine import WhisperCppEngine
from plugins.voice.stt.vosk_engine import VoskEngine

# TTS
from plugins.voice.tts.piper_engine import PiperEngine
from plugins.voice.tts.coqui_engine import CoquiEngine
from plugins.voice.tts.styletts2_engine import StyleTTS2Engine

TEST_DATA = Path('/opt/l-llm/voice-IO-hub/bench/test_data')
SAMPLES = sorted(TEST_DATA.glob('*.wav'))[:5]  # first 5 wav files

async def test_stt():
    print('=== STT Speed Test ===')
    engines = {
        'faster-whisper': FasterWhisperEngine({'model':'tiny','device':'cpu','compute_type':'int8','language':'ja'}),
        #'whisper-cpp': WhisperCppEngine({'model':'tiny','language':'ja'}),  # might be heavy
        #'vosk': VoskEngine({'model':'vosk-model-small-ja-0.22'}),  # may not be present
    }
    for name, eng in engines.items():
        try:
            await eng.load_model()
        except Exception as e:
            print(f'  {name}: load failed: {e}')
            continue
        times = []
        for wav_path in SAMPLES:
            wav, sr = sf.read(wav_path)
            if sr != 16000:
                # resample not implemented; skip if not 16k
                continue
            start = time.perf_counter()
            try:
                text = await eng.transcribe(wav.astype(np.float32))
            except Exception as e:
                text = f'ERROR: {e}'
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f'  {name} {wav_path.name}: {elapsed:.3f}s -> {text[:30]}')
        if times:
            avg = sum(times)/len(times)
            print(f'  {name} average over {len(times)} samples: {avg:.3f}s')

async def test_tts():
    print('=== TTS Speed Test ===')
    engines = {
        'piper': PiperEngine({'model':'ja_JP-naist-neutral-medium','speaker_id':0}),
        #'coqui': CoquiEngine({'model_path':'tts_models/ja/kokoro/tacotron2-DDC'}),  # placeholder
        #'styletts2': StyleTTS2Engine({'model_path':'styletts2'}),  # placeholder
    }
    for name, eng in engines.items():
        try:
            await eng.load_model()
        except Exception as e:
            print(f'  {name}: load failed: {e}')
            continue
        # Use some Japanese text from json files
        texts = []
        for json_path in TEST_DATA.glob('*.json'):
            try:
                with open(json_path, encoding='utf-8') as f:
                    data = json.load(f)
                    if 'text' in data:
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
                print(f'  {name} "{txt[:10]}...": {elapsed:.3f}s, audio shape {getattr(audio,"shape","?")}')
            except Exception as e:
                print(f'  {name} "{txt[:10]}...": ERROR {e}')
        if times:
            avg = sum(times)/len(times)
            print(f'  {name} average over {len(times)} samples: {avg:.3f}s')

async def main():
    await test_stt()
    await test_tts()

if __name__ == '__main__':
    asyncio.run(main())
