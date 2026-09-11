"""
VoiceSentinel AI - Unit Test for Backend Prosody Analysis Module
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from backend.prosody_analyzer import analyze_prosody_features


def test_analyze_prosody_features():
    dummy_audio = np.random.randn(16000 * 3).astype(np.float32)
    prosody = analyze_prosody_features(dummy_audio, sample_rate=16000)

    assert "f0MeanHz" in prosody
    assert "f0StdDevHz" in prosody
    assert "energyVariationRms" in prosody
    assert "pauseRatio" in prosody
    assert "prosodyAnomalyScore" in prosody
    assert 0.0 <= prosody["prosodyAnomalyScore"] <= 100.0
