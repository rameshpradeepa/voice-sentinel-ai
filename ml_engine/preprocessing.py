"""
VoiceSentinel AI - Audio Preprocessing & Multi-Format Audio Decoder
----------------------------------------------------------------------
Supports decoding AAC, MP3, OGG, FLAC, and WAV formats into float32 waveforms,
16kHz mono resampling, fixed 3-second chunking, and log-Mel spectrogram generation.
"""

import sys
import os
import io
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TARGET_SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 3.0
CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_SEC)  # 48,000 samples
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512


def decode_audio_bytes(raw_bytes, filename=""):
    """
    FFmpeg Audio Decoder for WAV, FLAC, OGG, MP3, AAC.
    Delegates to backend.audio_decoder.decode_audio_bytes.
    """
    try:
        from backend.audio_decoder import decode_audio_bytes as ffmpeg_decode
    except ImportError:
        from audio_decoder import decode_audio_bytes as ffmpeg_decode
    return ffmpeg_decode(raw_bytes, filename=filename)


def resample_audio(audio_data, orig_sr, target_sr=TARGET_SAMPLE_RATE):
    """Resamples audio array to target sample rate (default 16kHz)."""
    if orig_sr == target_sr or len(audio_data) == 0:
        return audio_data.astype(np.float32)
    num_output_samples = int(len(audio_data) * target_sr / orig_sr)
    resampled = signal.resample(audio_data, num_output_samples)
    return resampled.astype(np.float32)


def convert_to_mono(audio_data):
    """Converts multi-channel audio array to 1-channel mono."""
    if audio_data.ndim > 1:
        return np.mean(audio_data, axis=1).astype(np.float32)
    return audio_data.astype(np.float32)


def normalize_audio(audio_data):
    """Peak normalizes audio amplitude to range [-1.0, 1.0]."""
    max_val = np.max(np.abs(audio_data))
    if max_val > 1e-7:
        return (audio_data / max_val).astype(np.float32)
    return audio_data.astype(np.float32)


def trim_silence(audio_data, threshold=0.01):
    """Trims leading and trailing silence below energy threshold."""
    energy = np.abs(audio_data)
    non_silent = np.where(energy > threshold)[0]
    if len(non_silent) > 0:
        return audio_data[non_silent[0]:non_silent[-1] + 1]
    return audio_data


def chunk_or_pad_audio(audio_data, target_samples=CHUNK_SAMPLES, overlap_ratio=0.5):
    """
    Splits long audio into fixed target_samples chunks with overlap,
    or pads short audio with zeros if shorter than target_samples.
    """
    length = len(audio_data)
    if length == 0:
        return [np.zeros(target_samples, dtype=np.float32)]

    if length <= target_samples:
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[:length] = audio_data
        return [padded]

    stride = int(target_samples * (1.0 - overlap_ratio))
    chunks = []
    start = 0
    while start + target_samples <= length:
        chunk = audio_data[start:start + target_samples]
        chunks.append(chunk.astype(np.float32))
        start += stride

    if start < length and (length - start) > (target_samples * 0.2):
        tail = audio_data[start:]
        padded_tail = np.zeros(target_samples, dtype=np.float32)
        padded_tail[:len(tail)] = tail
        chunks.append(padded_tail)

    return chunks if len(chunks) > 0 else [np.zeros(target_samples, dtype=np.float32)]


def create_mel_filterbank(sr=TARGET_SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    """Computes a triangular Mel-filterbank matrix."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    f_min = 0.0
    f_max = sr / 2.0

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    num_fft_bins = n_fft // 2 + 1
    weights = np.zeros((n_mels, num_fft_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                weights[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                weights[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return weights


_MEL_FILTERBANK_CACHE = create_mel_filterbank()


def compute_log_mel_spectrogram(audio_chunk, sr=TARGET_SAMPLE_RATE):
    """
    Computes log-Mel Spectrogram array for a 1D audio waveform chunk.
    Output shape: (1, N_MELS, time_steps)
    """
    _, _, stft_matrix = signal.stft(
        audio_chunk,
        fs=sr,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        padded=True
    )
    power_spectrogram = np.abs(stft_matrix) ** 2
    mel_spectrogram = np.dot(_MEL_FILTERBANK_CACHE, power_spectrogram)

    log_mel = np.log(mel_spectrogram + 1e-6)

    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-7
    log_mel_norm = (log_mel - mean) / std

    return np.expand_dims(log_mel_norm.astype(np.float32), axis=0)


def extract_comprehensive_acoustic_features(audio_data, sr=TARGET_SAMPLE_RATE):
    """
    Extracts comprehensive Spectral, Prosodic, Voice Quality, Formants, and MFCC descriptors.
    """
    if len(audio_data) == 0:
        return {}

    freqs, _, stft_matrix = signal.stft(audio_data, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    power_spec = np.abs(stft_matrix) ** 2
    mag_spec = np.abs(stft_matrix)

    # Spectral Centroid
    spectral_centroids = np.sum(freqs[:, np.newaxis] * mag_spec, axis=0) / (np.sum(mag_spec, axis=0) + 1e-9)
    centroid_mean = float(np.mean(spectral_centroids))

    # Spectral Bandwidth
    bandwidths = np.sqrt(np.sum(((freqs[:, np.newaxis] - spectral_centroids) ** 2) * mag_spec, axis=0) / (np.sum(mag_spec, axis=0) + 1e-9))
    bandwidth_mean = float(np.mean(bandwidths))

    # Spectral Rolloff
    cum_power = np.cumsum(power_spec, axis=0)
    total_power = cum_power[-1, :]
    rolloff_bins = np.argmax(cum_power >= 0.85 * total_power, axis=0)
    rolloff_freqs = freqs[rolloff_bins]
    rolloff_mean = float(np.mean(rolloff_freqs))

    # Spectral Flatness
    log_mag = np.log(mag_spec + 1e-9)
    geo_mean = np.exp(np.mean(log_mag, axis=0))
    arith_mean = np.mean(mag_spec, axis=0) + 1e-9
    flatness = float(np.mean(geo_mean / arith_mean))

    # High Frequency Energy Ratio
    high_freq_idx = np.where(freqs > 3800)[0]
    high_freq_energy = np.sum(power_spec[high_freq_idx, :])
    total_energy = np.sum(power_spec) + 1e-9
    high_freq_ratio = float(high_freq_energy / total_energy)

    # Zero Crossing Rate (ZCR)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / 2.0
    zcr = float(zero_crossings / len(audio_data))

    # RMS Volume
    rms = float(np.sqrt(np.mean(audio_data ** 2)))

    # Voice Quality (Jitter & Shimmer)
    peaks, _ = signal.find_peaks(audio_data, distance=int(sr / 400.0), height=0.05)
    if len(peaks) > 3:
        periods = np.diff(peaks)
        amplitudes = np.abs(audio_data[peaks])
        jitter = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-7))
        shimmer = float(np.mean(np.abs(np.diff(amplitudes))) / (np.mean(amplitudes) + 1e-7))
    else:
        jitter = 0.012
        shimmer = 0.028

    # Formant Estimation
    mean_mag = np.mean(mag_spec, axis=1)
    spec_peaks, _ = signal.find_peaks(mean_mag, distance=5, height=np.max(mean_mag)*0.1)
    peak_freqs = freqs[spec_peaks]
    f1 = float(peak_freqs[0]) if len(peak_freqs) > 0 else 520.0
    f2 = float(peak_freqs[1]) if len(peak_freqs) > 1 else 1560.0

    # MFCC Statistics
    mel_spec = np.dot(_MEL_FILTERBANK_CACHE, power_spec)
    log_mel = np.log(mel_spec + 1e-6)
    n = np.arange(13)[:, np.newaxis]
    k = np.arange(N_MELS)[np.newaxis, :]
    dct_matrix = np.cos(np.pi * n * (k + 0.5) / N_MELS)
    mfcc = np.dot(dct_matrix, log_mel)
    mfcc_mean = [float(m) for m in np.mean(mfcc, axis=1)]
    mfcc_std = [float(s) for s in np.std(mfcc, axis=1)]

    return {
        "spectralCentroid": round(centroid_mean, 1),
        "spectralBandwidth": round(bandwidth_mean, 1),
        "spectralRolloff": round(rolloff_mean, 1),
        "spectralFlatness": round(flatness, 4),
        "highFreqEnergyRatio": round(high_freq_ratio, 4),
        "highFreqEnergyRatioPercent": round(high_freq_ratio * 100.0, 2),
        "zcr": round(zcr, 4),
        "volumeRms": round(rms, 4),
        "voiceQuality": {
            "jitter": round(jitter, 4),
            "shimmer": round(shimmer, 4),
            "formantF1Hz": round(f1, 1),
            "formantF2Hz": round(f2, 1)
        },
        "mfccStats": {
            "mean": [round(v, 2) for v in mfcc_mean],
            "stdDev": [round(v, 2) for v in mfcc_std]
        }
    }


def preprocess_audio_pipeline(audio_data, orig_sr):
    """
    Full Audio Preprocessing Pipeline:
    1. Mono conversion
    2. Resampling to 16kHz
    3. Peak normalization
    4. Silence trimming
    5. Fixed 3-second chunking
    6. Log-Mel spectrogram generation for each chunk
    """
    mono = convert_to_mono(audio_data)
    resampled = resample_audio(mono, orig_sr, TARGET_SAMPLE_RATE)
    normalized = normalize_audio(resampled)
    trimmed = trim_silence(normalized)
    chunks = chunk_or_pad_audio(trimmed, TARGET_SAMPLE_RATE)

    spectrograms = [compute_log_mel_spectrogram(chunk) for chunk in chunks]
    return spectrograms, chunks
