"""
VoiceSentinel AI - Unit Test for Audio Preprocessing Pipeline
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from ml_engine.preprocessing import (
    convert_to_mono,
    resample_audio,
    normalize_audio,
    chunk_or_pad_audio,
    compute_log_mel_spectrogram,
    decode_audio_bytes,
    preprocess_audio_pipeline,
    TARGET_SAMPLE_RATE,
    CHUNK_SAMPLES,
    N_MELS
)
import io
import pytest
from scipy.io import wavfile


def test_convert_to_mono():
    stereo = np.random.randn(1000, 2).astype(np.float32)
    mono = convert_to_mono(stereo)
    assert mono.ndim == 1
    assert len(mono) == 1000


def test_resample_audio():
    original = np.random.randn(44100).astype(np.float32)
    resampled = resample_audio(original, orig_sr=44100, target_sr=16000)
    assert len(resampled) == 16000


def test_normalize_audio():
    audio = np.array([0.5, -2.0, 1.0], dtype=np.float32)
    norm = normalize_audio(audio)
    assert np.max(np.abs(norm)) == 1.0


def test_chunk_or_pad_audio():
    short_audio = np.random.randn(8000).astype(np.float32)
    chunks_short = chunk_or_pad_audio(short_audio, target_samples=CHUNK_SAMPLES)
    assert len(chunks_short) == 1
    assert len(chunks_short[0]) == CHUNK_SAMPLES


def test_compute_log_mel_spectrogram():
    chunk = np.random.randn(CHUNK_SAMPLES).astype(np.float32)
    spec = compute_log_mel_spectrogram(chunk, sr=TARGET_SAMPLE_RATE)
    assert spec.ndim == 3
    assert spec.shape[0] == 1
    assert spec.shape[1] == N_MELS


def test_decode_audio_bytes_wav():
    sr = 16000
    signal = (np.random.randn(sr) * 10000).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, signal)
    wav_bytes = buf.getvalue()

    mono, decoded_sr, duration, channels = decode_audio_bytes(wav_bytes, filename="test.wav")
    assert decoded_sr == 16000
    assert abs(duration - 1.0) < 0.1
    assert channels == 1
    assert len(mono) == 16000


def test_decode_audio_bytes_invalid():
    invalid_bytes = b"CORRUPTED_NON_AUDIO_BYTES_12345"
    with pytest.raises(ValueError):
        decode_audio_bytes(invalid_bytes, filename="corrupt.aac")

