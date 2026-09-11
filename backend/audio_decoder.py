"""
VoiceSentinel AI - Robust Subprocess FFmpeg Audio Decoder Module
-----------------------------------------------------------------
Decodes uploaded audio files (WAV, FLAC, OGG, MP3, AAC, M4A, MP4) via FFmpeg subprocess
to 16000 Hz 1-channel mono PCM WAV bytes and converts to float32 numpy array.
"""

import os
import io
import sys
import shutil
import subprocess
from pathlib import Path
import numpy as np
from scipy.io import wavfile

MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit
SUPPORTED_EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3', '.aac', '.m4a', '.mp4'}
SUPPORTED_MIME_TYPES = {
    'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/ogg',
    'audio/mpeg', 'audio/aac', 'audio/mp4', 'audio/x-m4a',
    'application/octet-stream'
}


def find_ffmpeg() -> str:
    """
    Finds the full absolute path of the ffmpeg executable on Windows / Linux / macOS.
    Checks environment variables, system PATH, known WinGet links, and common install locations.
    """
    candidates = []

    # 1. Environment Variable
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path:
        candidates.append(env_path)

    # 2. System PATH via shutil.which
    which_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if which_path:
        candidates.append(which_path)

    # 3. Known Windows WinGet path for user 'D E L L'
    candidates.append(r"C:\Users\D E L L\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe")

    # 4. Dynamic WinGet / LocalAppData paths
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        candidates.append(os.path.join(local_appdata, r"Microsoft\WinGet\Links\ffmpeg.exe"))
        candidates.append(os.path.join(local_appdata, r"Microsoft\WinGet\Packages\GhanashyamPrasads.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.0.1-full_build\bin\ffmpeg.exe"))

    # 5. Program Files & Root C: candidates
    candidates.append(r"C:\ffmpeg\bin\ffmpeg.exe")
    candidates.append(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe")
    candidates.append(r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe")
    candidates.append("ffmpeg")  # Fallback name

    for candidate in candidates:
        if not candidate:
            continue
        if candidate == "ffmpeg":
            return "ffmpeg"
        try:
            p = Path(candidate)
            if p.exists() and p.is_file():
                return str(p.resolve())
        except Exception:
            pass

    raise RuntimeError("FFmpeg executable not found. Set FFMPEG_PATH to the full path of ffmpeg.exe.")


# Print startup detection message
try:
    _INIT_FFMPEG_PATH = find_ffmpeg()
    print(f"[AudioDecoder] FFmpeg detected:\n{_INIT_FFMPEG_PATH}")
except Exception as _err:
    print(f"[AudioDecoder] Warning on module init: {_err}")


class AudioDecodeError(ValueError):
    """Exception raised when audio file decoding fails."""
    def __init__(self, message="Unable to decode the uploaded audio file.", detail=None):
        super().__init__(message)
        self.message = message
        self.detail = detail or message


class AudioDecoderResult(dict):
    """
    Decoded audio dictionary with key indexing, property access, and tuple unpacking support.
    """
    def __init__(self, audio: np.ndarray, sample_rate: int, channels: int, duration_seconds: float, original_format: str):
        super().__init__({
            "audio": audio,
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_seconds": duration_seconds,
            "original_format": original_format
        })
        self.audio = audio
        self.audio_data = audio
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration_seconds = duration_seconds
        self.original_format = original_format

    def __iter__(self):
        # Enables tuple unpacking: float_pcm, sr, duration, channels = decode_audio_bytes(...)
        return iter((self.audio, self.sample_rate, self.duration_seconds, self.channels))


def decode_audio_bytes(raw_bytes: bytes, filename: str = "") -> AudioDecoderResult:
    """
    Decodes raw audio bytes via FFmpeg subprocess to a 16,000 Hz mono float32 PCM numpy array.

    Pipeline:
    raw_bytes -> FFmpeg subprocess (stdin -> stdout) -> WAV bytes -> float32 numpy array (16kHz mono)
    """
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    format_name = ext.lstrip('.').upper() if ext else "AUDIO"

    ffmpeg_path = find_ffmpeg()
    print(f"[AudioDecoder] FFmpeg executable:\n{ffmpeg_path}")
    print(f"[AudioDecoder] Received: {filename or 'unnamed'}")
    print(f"[AudioDecoder] Format: {format_name}")

    # 1. Size & Empty Payload Validation
    if not raw_bytes or len(raw_bytes) == 0:
        print("[AudioDecoder] Decode failed: Empty file payload")
        raise AudioDecodeError("Empty audio payload provided.")

    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        print(f"[AudioDecoder] Decode failed: File size exceeds {MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f}MB limit")
        raise AudioDecodeError("Audio file exceeds maximum size limit (50MB).")

    # 2. FFmpeg decoding via subprocess
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "wav",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1"
    ]

    print("[AudioDecoder] Starting FFmpeg decode")

    try:
        proc = subprocess.run(
            cmd,
            input=raw_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            shell=False
        )
    except subprocess.TimeoutExpired:
        print("[AudioDecoder] Error: FFmpeg decoding process timed out (30s limit)")
        raise AudioDecodeError("Audio decoding timed out after 30 seconds.")
    except FileNotFoundError:
        print(f"[AudioDecoder] Error: FFmpeg executable at '{ffmpeg_path}' not found")
        raise AudioDecodeError(f"FFmpeg executable not found at '{ffmpeg_path}'.")
    except Exception as e:
        print(f"[AudioDecoder] Error spawning FFmpeg process: {e}")
        raise AudioDecodeError(f"Failed to execute FFmpeg decoder: {str(e)}")

    if proc.returncode != 0 or not proc.stdout:
        stderr_msg = proc.stderr.decode('utf-8', errors='ignore').strip()
        print(f"[AudioDecoder] FFmpeg return code: {proc.returncode}")
        print(f"[AudioDecoder] stderr: {stderr_msg}")
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

    print("[AudioDecoder] Decode successful")
    print(f"[AudioDecoder] Sample Rate: {sr}")
    print("[AudioDecoder] Channels: 1")
    print(f"[AudioDecoder] Duration: {duration:.2f}s")

    return AudioDecoderResult(
        audio=float_pcm,
        sample_rate=sr,
        channels=1,
        duration_seconds=duration,
        original_format=format_name
    )
