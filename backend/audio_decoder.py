"""
VoiceSentinel AI - Robust Subprocess FFmpeg Audio Decoder Module
-----------------------------------------------------------------
Decodes uploaded audio files (WAV, FLAC, OGG, MP3, AAC, M4A) via FFmpeg subprocess
to 16000 Hz 1-channel mono PCM WAV bytes and converts to float32 numpy array.
"""

import os
import io
import sys
import shutil
import subprocess
import numpy as np
from scipy.io import wavfile

MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit
SUPPORTED_EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3', '.aac', '.m4a', '.mp4'}
SUPPORTED_MIME_TYPES = {
    'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/ogg',
    'audio/mpeg', 'audio/aac', 'audio/mp4', 'audio/x-m4a',
    'application/octet-stream'
}


class AudioDecodeError(ValueError):
    """Exception raised when audio file decoding fails."""
    def __init__(self, message="Unable to decode the uploaded audio file.", detail=None):
        super().__init__(message)
        self.message = message
        self.detail = detail or message


class DecodedAudio:
    """Dataclass wrapper for decoded audio array & metadata."""
    def __init__(self, audio_data: np.ndarray, sample_rate: int, channels: int, duration_seconds: float, original_format: str):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration_seconds = duration_seconds
        self.original_format = original_format

    def __iter__(self):
        # Support legacy tuple unpacking: float_pcm, sr, duration, channels = decode_audio_bytes(...)
        return iter((self.audio_data, self.sample_rate, self.duration_seconds, self.channels))


def decode_audio_bytes(raw_bytes: bytes, filename: str = "") -> DecodedAudio:
    """
    Decodes raw audio bytes via FFmpeg subprocess to a 16,000 Hz mono float32 PCM numpy array.

    Pipeline:
    raw_bytes -> FFmpeg subprocess (stdin -> stdout) -> WAV bytes -> float32 numpy array (16kHz mono)
    """
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    format_name = ext.lstrip('.').upper() if ext else "AUDIO"

    print(f"[AudioDecoder] Filename: {filename or 'unnamed'}")
    print(f"[AudioDecoder] Detected format: {format_name}")

    # 1. Size & Empty Payload Validation
    if not raw_bytes or len(raw_bytes) == 0:
        print("[AudioDecoder] Decode failed: Empty file payload")
        raise AudioDecodeError("Empty audio payload provided.")

    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        print(f"[AudioDecoder] Decode failed: File size exceeds {MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f}MB limit")
        raise AudioDecodeError("Audio file exceeds maximum size limit (50MB).")

    # 2. FFmpeg decoding via subprocess
    ffmpeg_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe") or "ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "wav",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1"
    ]

    print("[AudioDecoder] FFmpeg start")

    try:
        proc = subprocess.run(
            cmd,
            input=raw_bytes,
            capture_output=True,
            timeout=15,
            shell=False
        )
    except subprocess.TimeoutExpired:
        print("[AudioDecoder] Error: FFmpeg decoding process timed out (15s limit)")
        raise AudioDecodeError("Audio decoding timed out after 15 seconds.")
    except FileNotFoundError:
        print("[AudioDecoder] Error: FFmpeg executable not found in system PATH")
        raise AudioDecodeError("FFmpeg executable is not available on server PATH.")
    except Exception as e:
        print(f"[AudioDecoder] Error spawning FFmpeg process: {e}")
        raise AudioDecodeError(f"Failed to execute FFmpeg decoder: {str(e)}")

    if proc.returncode != 0 or not proc.stdout:
        stderr_msg = proc.stderr.decode('utf-8', errors='ignore').strip()
        print(f"[AudioDecoder] FFmpeg decode error (code {proc.returncode}): {stderr_msg}")
        raise AudioDecodeError(f"FFmpeg decode error: {stderr_msg or 'Invalid audio bytes'}", detail=stderr_msg)

    # 3. Parse decoded WAV bytes stdout using scipy.io.wavfile.read ONLY on FFmpeg output
    try:
        sr, wav_data = wavfile.read(io.BytesIO(proc.stdout))
    except Exception as e:
        print(f"[AudioDecoder] Error parsing FFmpeg output WAV bytes: {e}")
        raise AudioDecodeError("Unable to parse decoded WAV audio stream from FFmpeg.", detail=str(e))

    # Convert int16 / int32 / uint8 to normalized float32 [-1.0, 1.0]
    if wav_data.dtype == np.int16:
        float_pcm = (wav_data / 32768.0).astype(np.float32)
    elif wav_data.dtype == np.int32:
        float_pcm = (wav_data / 2147483648.0).astype(np.float32)
    elif wav_data.dtype == np.uint8:
        float_pcm = ((wav_data.astype(np.float32) - 128.0) / 128.0)
    else:
        float_pcm = wav_data.astype(np.float32)

    # Ensure 1D mono array
    if float_pcm.ndim > 1:
        float_pcm = np.mean(float_pcm, axis=1)

    duration = len(float_pcm) / float(sr) if sr > 0 else 0.0

    print("[AudioDecoder] FFmpeg decode success")
    print(f"[AudioDecoder] Sample rate: {sr}")
    print("[AudioDecoder] Channels: 1")
    print(f"[AudioDecoder] Duration: {duration:.2f}s ({len(float_pcm)} samples)")

    return DecodedAudio(
        audio_data=float_pcm,
        sample_rate=sr,
        channels=1,
        duration_seconds=duration,
        original_format=format_name
    )
