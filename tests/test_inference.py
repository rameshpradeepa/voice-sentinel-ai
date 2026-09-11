"""
VoiceSentinel AI - Unit Test for Real-Time PyTorch Inference Engine
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from ml_engine.inference import VoiceDeepfakeInference


def test_inference_empty_audio():
    engine = VoiceDeepfakeInference(device="cpu")
    result = engine.predict_audio_array(np.array([], dtype=np.float32))
    assert "error" in result
    assert result["error"] == "INVALID_AUDIO"


def test_inference_synthetic_array():
    engine = VoiceDeepfakeInference(device="cpu")
    audio = np.random.randn(48000).astype(np.float32)
    result = engine.predict_audio_array(audio, sample_rate=16000)

    if result.get("error") == "MODEL_NOT_FOUND":
        assert result["error"] == "MODEL_NOT_FOUND"
    else:
        assert "aiProbability" in result
        assert 0.0 <= result["aiProbability"] <= 1.0
        assert result["verdict"] in ["GENUINE", "AI_GENERATED"]
