"""
VoiceSentinel AI - Real Audio Dataset Generator Utility (Smoke-Test Baseline)
--------------------------------------------------------------------------------
Generates baseline 16kHz WAV audio samples in datasets/genuine, datasets/synthetic,
and datasets/human_mimicry across regional accent subfolders (Hindi, Tamil, Indian English).
"""

import sys
import os
import numpy as np
import scipy.io.wavfile as wavfile

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TARGET_SR = 16000
DURATION_SEC = 3.5
NUM_SAMPLES_PER_CATEGORY = 12


def generate_human_speech_signal(duration=DURATION_SEC, sr=TARGET_SR, f0=150.0):
    """
    Synthesizes a natural human-like speech signal with harmonic formants,
    pitch variation (vibrato/prosody), and natural amplitude envelope.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0_contour = f0 + 15.0 * np.sin(2 * np.pi * 1.5 * t) + 8.0 * np.cos(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(f0_contour) / sr

    voice = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    f1 = np.sin(2 * np.pi * 500 * t)
    f2 = np.sin(2 * np.pi * 1500 * t)

    signal_out = voice * (0.6 + 0.4 * f1) + 0.2 * f2
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 0.5 * t))
    signal_out = signal_out * envelope

    noise = np.random.normal(0, 0.005, len(t))
    signal_out = signal_out + noise

    signal_out = signal_out / (np.max(np.abs(signal_out)) + 1e-7) * 0.8
    return (signal_out * 32767).astype(np.int16)


def generate_ai_synthetic_signal(duration=DURATION_SEC, sr=TARGET_SR, f0=145.0):
    """
    Synthesizes a neural vocoder / AI clone signal with monotone pitch lock,
    steep 3.8kHz bandpass cutoff, phase discontinuity, and vocoder pulse buzz.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0_contour = f0 + 0.8 * np.sin(2 * np.pi * 0.2 * t)
    phase = 2 * np.pi * np.cumsum(f0_contour) / sr

    pulse = np.sign(np.sin(phase))
    harmonics = pulse + 0.3 * np.sin(3 * phase) + 0.15 * np.sin(5 * phase)

    phase_jitter = 0.05 * np.random.choice([-1, 1], size=len(t))
    signal_out = harmonics + phase_jitter

    window_len = 5
    signal_out = np.convolve(signal_out, np.ones(window_len)/window_len, mode='same')

    signal_out = signal_out / (np.max(np.abs(signal_out)) + 1e-7) * 0.8
    return (signal_out * 32767).astype(np.int16)


def generate_human_mimicry_signal(duration=DURATION_SEC, sr=TARGET_SR, f0=175.0):
    """
    Synthesizes a human mimicry signal (real human voice intentionally altering pitch/formants).
    Has high human-like speech naturalness but distinct pitch shift / formant mismatch.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Human voice with exaggerated vocal pitch variation
    f0_contour = f0 + 35.0 * np.sin(2 * np.pi * 2.2 * t)
    phase = 2 * np.pi * np.cumsum(f0_contour) / sr

    voice = np.sin(phase) + 0.6 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
    signal_out = voice * 0.7 + 0.008 * np.random.normal(0, 1, len(t))

    signal_out = signal_out / (np.max(np.abs(signal_out)) + 1e-7) * 0.8
    return (signal_out * 32767).astype(np.int16)


def generate_dataset_directory(base_dir="datasets"):
    """Populates genuine/, synthetic/, and human_mimicry/ dataset subfolders with WAV files."""
    categories = {
        "genuine": generate_human_speech_signal,
        "synthetic": generate_ai_synthetic_signal,
        "human_mimicry": generate_human_mimicry_signal
    }

    accents = ["hindi", "tamil", "indian_english", "default"]
    created_count = 0

    for category, generator_fn in categories.items():
        for accent in accents:
            folder_path = os.path.join(base_dir, category, accent)
            os.makedirs(folder_path, exist_ok=True)

            for i in range(1, NUM_SAMPLES_PER_CATEGORY + 1):
                file_path = os.path.join(folder_path, f"sample_{accent}_{i:02d}.wav")
                base_f0 = 150.0 + (i % 5) * 15.0 if category == "genuine" else (145.0 if category == "synthetic" else 175.0)
                audio_pcm = generator_fn(duration=DURATION_SEC, sr=TARGET_SR, f0=base_f0)
                wavfile.write(file_path, TARGET_SR, audio_pcm)
                created_count += 1

    print(f"[+] Successfully generated {created_count} baseline WAV audio dataset samples in '{base_dir}/'")


if __name__ == "__main__":
    generate_dataset_directory()
