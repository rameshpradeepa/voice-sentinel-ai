"""
VoiceSentinel AI - Acoustic Biometric Profile Engine
-----------------------------------------------------
Extracts acoustic biometric feature vectors (F0 pitch mean/std, spectral centroid,
high-frequency energy ratio, zero-crossing rate) and computes true Cosine Similarity
against enrolled authorized speaker profiles.
"""

import numpy as np


class AcousticBiometricEngine:
    """
    Acoustic Biometric Profile & Vector Distance Matching Engine.
    """
    def __init__(self):
        # Default enrolled target executive profiles
        self.enrolled_profiles = {
            "cxo_001": {
                "id": "cxo_001",
                "name": "Rajesh Sharma (CFO)",
                "organization": "Apex National Bank",
                "role": "Chief Financial Officer",
                "vector": np.array([142.5, 26.8, 2480.0, 0.085, 0.41], dtype=np.float32),
                "enrolledAt": "2026-08-15T10:30:00Z"
            },
            "govt_002": {
                "id": "govt_002",
                "name": "Dr. Anita Desai (Director)",
                "organization": "Cyber Security Cell, Ministry of Tech",
                "role": "Senior Security Director",
                "vector": np.array([210.3, 31.2, 2890.0, 0.092, 0.38], dtype=np.float32),
                "enrolledAt": "2026-08-20T14:15:00Z"
            }
        }

    def extract_biometric_vector(self, f0_mean, f0_std, spectral_centroid, high_freq_ratio, zcr):
        """Builds normalized 5D acoustic biometric vector."""
        return np.array([
            float(f0_mean),
            float(f0_std),
            float(spectral_centroid),
            float(high_freq_ratio * 100.0),
            float(zcr * 100.0)
        ], dtype=np.float32)

    def compute_cosine_similarity(self, vec_a, vec_b):
        """Computes true Cosine Similarity between two 1D feature vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-7 or norm_b < 1e-7:
            return 0.0
        similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        return float(np.clip(similarity, 0.0, 1.0))

    def evaluate_biometric_match(self, live_vector, target_profile_id=None):
        """
        Compares live_vector against target_profile_id (if provided).
        Returns match percentage, mismatch score, and profile details.
        """
        if not target_profile_id or target_profile_id not in self.enrolled_profiles:
            return {
                "profileEnrolled": False,
                "targetProfileId": target_profile_id or "NONE",
                "matchPercentage": 100.0,
                "biometricMismatchScore": 0.0,
                "label": "NO_TARGET_PROFILE_ENROLLED"
            }

        profile = self.enrolled_profiles[target_profile_id]
        ref_vector = profile["vector"]

        sim = self.compute_cosine_similarity(live_vector, ref_vector)
        match_pct = round(sim * 100.0, 1)
        mismatch_score = round((1.0 - sim) * 100.0, 1)

        return {
            "profileEnrolled": True,
            "targetProfileId": target_profile_id,
            "speakerName": profile["name"],
            "organization": profile["organization"],
            "cosineSimilarity": round(sim, 4),
            "matchPercentage": match_pct,
            "biometricMismatchScore": mismatch_score,
            "label": "ACOUSTIC_BIOMETRIC_PROFILE_MATCH" if match_pct >= 85.0 else "ACOUSTIC_BIOMETRIC_MISMATCH"
        }

    def enroll_new_profile(self, profile_id, name, organization, f0_mean, f0_std, spectral_centroid):
        """Registers a new acoustic biometric profile."""
        vec = np.array([f0_mean, f0_std, spectral_centroid, 8.5, 40.0], dtype=np.float32)
        profile_data = {
            "id": profile_id,
            "name": name,
            "organization": organization,
            "vector": vec,
            "enrolledAt": "2026-09-11T11:00:00Z"
        }
        self.enrolled_profiles[profile_id] = profile_data
        return profile_data


# Global Biometric Engine Instance
_BIOMETRIC_ENGINE = AcousticBiometricEngine()

def get_biometric_engine():
    return _BIOMETRIC_ENGINE
