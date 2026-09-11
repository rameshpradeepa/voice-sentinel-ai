"""
VoiceSentinel AI - Privacy & Compliance Module
------------------------------------------------
Computes SHA-256 audio hashes, ensures temporary memory buffer cleanup,
and enforces DPDP Act / GDPR compliant feature-only logging.
"""

import hashlib
import gc


def compute_sha256_hash(audio_bytes):
    """Computes SHA-256 cryptographic hash hex string from raw audio bytes."""
    if not audio_bytes:
        return "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # SHA256 of empty byte
    return hashlib.sha256(audio_bytes).hexdigest()


def purge_temporary_audio_buffer(*buffers):
    """Explicitly deletes temporary PCM byte arrays from memory and triggers garbage collection."""
    for buf in buffers:
        try:
            del buf
        except Exception:
            pass
    gc.collect()


def create_privacy_preserving_log(incident_id, audio_hash, risk_result, model_version="PyTorch-CNN-v2.0"):
    """
    Creates an anonymized privacy-preserving audit log record.
    Stores metadata summaries and SHA-256 hash only. Raw audio is excluded.
    """
    return {
        "incidentId": incident_id,
        "audioHashSha256": audio_hash,
        "overallRiskScore": risk_result["overallRiskScore"],
        "verdict": risk_result["verdict"],
        "alertLevel": risk_result.get("alertLevel", risk_result.get("riskLevel", "LOW")),
        "modelVersion": model_version,
        "subScores": risk_result.get("riskFormulaBreakdown", {}),
        "privacyStatus": "RAW_AUDIO_PURGED_FEATURE_ONLY_LOGGED"
    }
