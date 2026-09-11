"""
VoiceSentinel AI - Unit Test for SHA-256 Hashing and Privacy Cleanup
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.privacy_guard import compute_sha256_hash, create_privacy_preserving_log, purge_temporary_audio_buffer


def test_sha256_hashing():
    sample_bytes = b"VoiceSentinel AI Cyber Security Audio Frame"
    hash_str = compute_sha256_hash(sample_bytes)

    assert len(hash_str) == 64
    assert hash_str == compute_sha256_hash(sample_bytes)


def test_privacy_preserving_log():
    dummy_risk = {
        "overallRiskScore": 92.5,
        "verdict": "CRITICAL_RISK_AI_VOICE_CLONE",
        "alertLevel": "CRITICAL",
        "subScores": {"aiModelScore": 95.0, "prosodyAnomalyScore": 40.0}
    }
    log = create_privacy_preserving_log("inc_1234", "a" * 64, dummy_risk)

    assert log["audioHashSha256"] == "a" * 64
    assert log["privacyStatus"] == "RAW_AUDIO_PURGED_FEATURE_ONLY_LOGGED"
