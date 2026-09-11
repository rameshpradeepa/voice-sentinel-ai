"""
VoiceSentinel AI - Comprehensive Security Risk Engine & Security Interpretation Layer
----------------------------------------------------------------------------------------
Combines PyTorch neural AI probability with prosody anomalies, voice quality features,
acoustic biometric mismatch, and contextual transaction metadata into a 0-100 Security Risk Score.
Differentiates AI Voice Clones from Human Mimicry / Identity Mismatches.
"""

import numpy as np


class SecurityRiskEngine:
    """
    Multi-Factor Security Risk Engine & Interpretation Layer.
    """
    def __init__(self):
        self.weights = {
            "aiModel": 0.55,      # PyTorch AI Classifier Probability (55% weight)
            "prosody": 0.20,      # F0 pitch variance & pause cadence (20% weight)
            "biometric": 0.15,    # Acoustic Biometric Vector Cosine Distance (15% weight)
            "metadata": 0.10      # Caller ID spoofing & transaction context (10% weight)
        }

    def compute_risk_score(
        self,
        ai_probability,
        prosody_anomaly_score=0.0,
        biometric_mismatch_score=0.0,
        call_metadata=None,
        spectral_features=None,
        prosody_features=None,
        voice_quality=None,
        biometric_result=None
    ):
        """
        Calculates normalized 0-100 overall Security Risk Score, verdict,
        whyDetected explanations, and complete technical parameter breakdown.
        """
        metadata = call_metadata or {}
        spectral = spectral_features or {}
        prosody = prosody_features or {}
        quality = voice_quality or {}
        bio = biometric_result or {}

        ai_prob = float(np.clip(ai_probability, 0.0, 1.0))
        human_prob = float(np.clip(1.0 - ai_prob, 0.0, 1.0))

        # 1. AI Model Score (0 to 100)
        ai_score = ai_prob * 100.0

        # 2. Metadata Score (0 to 100)
        metadata_score = 0.0
        if metadata.get("isSpoofedCallerID", False):
            metadata_score += 40.0
        if metadata.get("highRiskTransaction", False):
            metadata_score += 35.0
        if metadata.get("voipProxyDetected", False):
            metadata_score += 25.0
        metadata_score = float(np.clip(metadata_score, 0.0, 100.0))

        bio_mismatch = float(bio.get("biometricMismatchScore", biometric_mismatch_score))

        # 3. Weighted Formula Sum: 55% AI + 20% Prosody + 15% Biometric + 10% Metadata
        total_risk = (
            self.weights["aiModel"] * ai_score +
            self.weights["prosody"] * float(prosody_anomaly_score) +
            self.weights["biometric"] * bio_mismatch +
            self.weights["metadata"] * metadata_score
        )

        # 4. Security Interpretation Layer (Verdict Determination)
        why_detected = []

        if ai_prob >= 0.50:
            verdict = "AI VOICE CLONE SUSPECTED"
            why_detected.append(f"High PyTorch neural deepfake probability ({round(ai_prob * 100, 1)}%)")
            if prosody_anomaly_score >= 30.0:
                why_detected.append("Unnatural pitch monotone lock & silence cadence")
            if spectral.get("highFreqEnergyRatio", 0) < 0.03:
                why_detected.append("Bandwidth suppression (< 3.8kHz vocoder limit)")
            if bio_mismatch >= 50.0:
                why_detected.append(f"Target speaker biometric mismatch ({round(bio_mismatch, 1)}%)")

        elif bio_mismatch >= 65.0 or (prosody.get("f0StdDevHz", 20.0) > 32.0 and bio_mismatch >= 50.0):
            verdict = "HUMAN MIMICRY / POSSIBLE IMPERSONATION"
            why_detected.append("High human speech probability confirmed by neural model")
            why_detected.append(f"Significant acoustic speaker mismatch ({round(bio_mismatch, 1)}%)")
            why_detected.append("Voice style / pitch imitation indicators detected")

            # Human Mimicry is a security identity mismatch; ensure risk score reflects MEDIUM/HIGH tier (at least 60.0)
            total_risk = max(total_risk, 62.5)

        elif ai_prob < 0.35 and bio_mismatch < 50.0:
            verdict = "GENUINE HUMAN VOICE"
            why_detected.append("High human speech confidence")
            why_detected.append("Natural pitch variations and formant transitions")
            why_detected.append("Acoustic biometric signature matched enrolled profile")

        else:
            verdict = "UNKNOWN / INCONCLUSIVE"
            why_detected.append("Inconclusive signal characteristics or high ambient noise")

        overall_risk = round(float(np.clip(total_risk, 0.0, 100.0)), 1)

        # Risk Level Tiers
        if overall_risk >= 80.0:
            risk_level = "CRITICAL"
            available_actions = ["BLOCK CALL", "BLOCK TRANSACTION", "REPORT INCIDENT", "REQUIRE MFA", "ESCALATE TO SOC"]
        elif overall_risk >= 60.0:
            risk_level = "HIGH"
            available_actions = ["REQUIRE MFA", "VERIFY IDENTITY", "REPORT INCIDENT", "BLOCK CALL"]
        elif overall_risk >= 30.0:
            risk_level = "MEDIUM"
            available_actions = ["VERIFY IDENTITY", "CHALLENGE PHRASE", "REPORT INCIDENT"]
        else:
            risk_level = "LOW"
            available_actions = ["ALLOW CALL", "REPORT INCIDENT"]

        return {
            "aiProbability": round(ai_prob, 4),
            "aiProbabilityPercent": round(ai_prob * 100.0, 1),
            "humanProbability": round(human_prob, 4),
            "humanProbabilityPercent": round(human_prob * 100.0, 1),
            "overallRiskScore": overall_risk,
            "riskLevel": risk_level,
            "verdict": verdict,
            "whyDetected": why_detected,
            "availableActions": available_actions,
            "riskFormulaBreakdown": {
                "aiModelContribution": round(self.weights["aiModel"] * ai_score, 1),
                "prosodyContribution": round(self.weights["prosody"] * float(prosody_anomaly_score), 1),
                "biometricContribution": round(self.weights["biometric"] * bio_mismatch, 1),
                "metadataContribution": round(self.weights["metadata"] * metadata_score, 1),
                "weights": self.weights
            },
            "technicalParameters": {
                "spectral": spectral,
                "prosody": prosody,
                "voiceQuality": quality,
                "biometric": bio,
                "context": metadata
            }
        }


# Global Instance
_RISK_ENGINE = SecurityRiskEngine()

def get_risk_engine():
    return _RISK_ENGINE
