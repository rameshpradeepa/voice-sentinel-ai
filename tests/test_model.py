"""
VoiceSentinel AI - Unit Test for PyTorch Deepfake Model Architecture
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from ml_engine.model import VoiceDeepfakeDetector


def test_model_instantiation():
    model = VoiceDeepfakeDetector(in_channels=1, num_classes=1)
    assert model is not None
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0


def test_model_forward_pass():
    model = VoiceDeepfakeDetector(in_channels=1, num_classes=1)
    model.eval()

    dummy_input = torch.randn(2, 1, 64, 94)
    logits = model(dummy_input)

    assert logits.shape == (2, 1)


def test_model_predict_probability():
    model = VoiceDeepfakeDetector(in_channels=1, num_classes=1)
    dummy_input = torch.randn(1, 1, 64, 94)

    prob = model.predict_probability(dummy_input)
    assert prob.shape == (1, 1)
    assert 0.0 <= prob.item() <= 1.0
