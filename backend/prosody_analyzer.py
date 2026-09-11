"""
VoiceSentinel AI - Real Prosody Analysis Module
------------------------------------------------
Extracts fundamental frequency (F0) mean, F0 variance, pitch range,
energy variation, pause ratio, and temporal rhythm variation from audio signals.
"""

import numpy as np
import scipy.signal as signal


def analyze_prosody_features(audio_data, sample_rate=16000):
    """
    Computes prosodic acoustic descriptors from a 1D float audio array.
    Returns prosody metrics dict and a prosody anomaly score (0-100).
    """
    if audio_data is None or len(audio_data) == 0:
        return {
            "f0MeanHz": 0.0,
            "f0VarianceHz": 0.0,
            "pitchRangeHz": 0.0,
            "energyVariationRms": 0.0,
            "pauseRatio": 0.0,
            "temporalVariation": 0.0,
            "prosodyAnomalyScore": 0.0
        }

    # 1. Energy Envelope & Pause Ratio
    frame_size = int(sample_rate * 0.03)  # 30ms frame
    hop_size = int(sample_rate * 0.01)    # 10ms hop

    num_frames = max(1, (len(audio_data) - frame_size) // hop_size + 1)
    frame_energies = []

    for i in range(num_frames):
        start = i * hop_size
        frame = audio_data[start:start + frame_size]
        energy = np.sqrt(np.mean(frame ** 2))
        frame_energies.append(energy)

    frame_energies = np.array(frame_energies)
    energy_mean = np.mean(frame_energies) if len(frame_energies) > 0 else 0.01
    energy_std = np.std(frame_energies) if len(frame_energies) > 0 else 0.0

    # Pause ratio: fraction of frames below 15% of average energy
    silence_threshold = 0.15 * energy_mean
    silent_frames = np.sum(frame_energies < silence_threshold)
    pause_ratio = silent_frames / max(1, len(frame_energies))

    # 2. Pitch (F0) Extraction via Autocorrelation across frames
    pitches = []
    for i in range(0, num_frames, 2):  # Step by 2 for speed
        start = i * hop_size
        frame = audio_data[start:start + frame_size]
        if len(frame) < frame_size:
            continue

        # Autocorrelation
        autocorr = signal.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        min_lag = int(sample_rate / 400.0)  # Max F0 = 400Hz
        max_lag = int(sample_rate / 60.0)   # Min F0 = 60Hz

        if len(autocorr) > max_lag:
            peak_lag = min_lag + np.argmax(autocorr[min_lag:max_lag])
            pitch = sample_rate / peak_lag if peak_lag > 0 else 0.0
            if 60.0 <= pitch <= 400.0:
                pitches.append(pitch)

    pitches = np.array(pitches)
    if len(pitches) > 0:
        f0_mean = float(np.mean(pitches))
        f0_variance = float(np.var(pitches))
        f0_std = float(np.std(pitches))
        pitch_range = float(np.max(pitches) - np.min(pitches))
    else:
        f0_mean = 145.0
        f0_variance = 10.0
        f0_std = 3.16
        pitch_range = 15.0

    # 3. Temporal Rhythm Variation (delta energy variation)
    if len(frame_energies) > 1:
        diffs = np.abs(np.diff(frame_energies))
        temporal_variation = float(np.mean(diffs))
    else:
        temporal_variation = 0.05

    # 4. Prosody Anomaly Score Computation
    # Unnatural neural TTS artifacts: F0 std dev < 5Hz (monotone lock) or low energy std dev
    anomaly_score = 0.0
    if f0_std < 5.0:
        anomaly_score += 45.0  # Monotone pitch constraint
    elif f0_std < 10.0:
        anomaly_score += 25.0

    if pause_ratio > 0.45 or pause_ratio < 0.05:
        anomaly_score += 25.0  # Unnatural pause ratio

    if energy_std < 0.01:
        anomaly_score += 30.0  # Constant unnatural amplitude

    anomaly_score = min(100.0, max(0.0, anomaly_score))

    return {
        "f0MeanHz": round(f0_mean, 1),
        "f0VarianceHz": round(f0_variance, 1),
        "f0StdDevHz": round(f0_std, 1),
        "pitchRangeHz": round(pitch_range, 1),
        "energyVariationRms": round(float(energy_std), 4),
        "pauseRatio": round(float(pause_ratio), 3),
        "temporalVariation": round(float(temporal_variation), 4),
        "prosodyAnomalyScore": round(anomaly_score, 1)
    }
