"""
VoiceSentinel AI - Unit Test for Multi-Factor Security Risk Engine & Interpretation Layer
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.risk_engine import SecurityRiskEngine


def test_genuine_classification():
    engine = SecurityRiskEngine()
    result = engine.compute_risk_score(ai_probability=0.10, prosody_anomaly_score=5.0, biometric_mismatch_score=0.0)

    assert result["verdict"] == "GENUINE HUMAN VOICE"
    assert result["riskLevel"] == "LOW"
    assert result["overallRiskScore"] < 30.0
    assert "ALLOW CALL" in result["availableActions"]


def test_synthetic_ai_classification():
    engine = SecurityRiskEngine()
    result = engine.compute_risk_score(
        ai_probability=0.92,
        prosody_anomaly_score=60.0,
        biometric_mismatch_score=80.0,
        call_metadata={"isSpoofedCallerID": True, "highRiskTransaction": True}
    )

    assert result["verdict"] == "AI VOICE CLONE SUSPECTED"
    assert result["riskLevel"] == "CRITICAL"
    assert result["overallRiskScore"] >= 80.0
    assert "BLOCK CALL" in result["availableActions"]


def test_human_mimicry_classification():
    engine = SecurityRiskEngine()
    # High human probability (low AI prob 0.18), but high biometric mismatch 75%
    result = engine.compute_risk_score(
        ai_probability=0.18,
        prosody_anomaly_score=35.0,
        biometric_mismatch_score=75.0
    )

    # Must NOT call human mimicry "AI VOICE CLONE SUSPECTED"!
    assert result["verdict"] == "HUMAN MIMICRY / POSSIBLE IMPERSONATION"
    assert result["verdict"] != "AI VOICE CLONE SUSPECTED"
    assert result["riskLevel"] in ["HIGH", "MEDIUM"]


def test_no_hardcoded_values():
    engine = SecurityRiskEngine()
    res1 = engine.compute_risk_score(ai_probability=0.1234)
    res2 = engine.compute_risk_score(ai_probability=0.8765)

    assert res1["overallRiskScore"] != res2["overallRiskScore"]
    assert abs(res1["aiProbabilityPercent"] - 12.3) < 0.2
    assert abs(res2["aiProbabilityPercent"] - 87.65) < 0.2
