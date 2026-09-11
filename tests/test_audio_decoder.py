"""
VoiceSentinel AI - Unit Tests for FFmpeg Audio Decoder (backend/audio_decoder.py)
"""

import sys
import os
import io
import pytest
import numpy as np
from scipy.io import wavfile

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.audio_decoder import decode_audio_bytes, AudioDecodeError, find_ffmpeg


def test_find_ffmpeg_path():
    path = find_ffmpeg()
    assert path is not None
    assert len(path) > 0


def test_ffmpeg_decode_wav_bytes():
    # Create a 1-second 16kHz synthetic sine wave WAV buffer
    sr = 16000
    t = np.linspace(0, 1.0, sr, False)
    tone = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, sr, tone)
    wav_bytes = buf.getvalue()

    decoded = decode_audio_bytes(wav_bytes, filename="test_sine.wav")

    assert decoded.sample_rate == 16000
    assert decoded.channels == 1
    assert abs(decoded.duration_seconds - 1.0) < 0.1
    assert len(decoded.audio_data) == 16000
    assert isinstance(decoded.audio_data, np.ndarray)
    assert decoded.audio_data.dtype == np.float32
    assert decoded.original_format == "WAV"


def test_ffmpeg_decode_empty_bytes():
    with pytest.raises(AudioDecodeError):
        decode_audio_bytes(b"", filename="empty.aac")


def test_ffmpeg_decode_invalid_bytes():
    with pytest.raises(AudioDecodeError):
        decode_audio_bytes(b"CORRUPTED_NON_AUDIO_HEADER_12345", filename="corrupt.aac")
