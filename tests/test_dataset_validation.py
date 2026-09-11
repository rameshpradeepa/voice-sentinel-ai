"""
VoiceSentinel AI - Unit Test for Dataset Validation Engine
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_engine.validate_dataset import validate_dataset


def test_validate_dataset():
    report, is_valid = validate_dataset("datasets")
    assert isinstance(report, dict)
    assert "totalGenuine" in report
    assert "totalSynthetic" in report
    assert "totalHumanMimicry" in report
    assert "isValid" in report
