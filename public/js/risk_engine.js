/**
 * VoiceSentinel AI - Dynamic Multi-Factor Voice Impersonation Risk Engine
 * Computes live weighted risk scores across acoustic, prosodic, biometric, and contextual layers.
 */

class RiskEngine {
  constructor() {
    this.weights = {
      spectral: 0.35,
      prosody: 0.25,
      biometric: 0.25,
      metadata: 0.15
    };
  }

  evaluateRisk(features, callMetadata = {}, enrolledProfile = null) {
    let spectralScore = 0;
    let prosodyScore = 0;
    let biometricMismatchScore = 0;
    let metadataScore = 0;

    if (features) {
      // 1. Spectral Artifacts
      // High-Frequency Energy Ratio cutoff (vocoder 16kHz bandpass limits)
      if (features.highFreqEnergyRatio < 0.02) {
        spectralScore += 45; // Severe high frequency suppression typical of Tacotron / MelGAN
      } else if (features.highFreqEnergyRatio < 0.04) {
        spectralScore += 25;
      }

      // Spectral Centroid Std Dev (robotic uniformity)
      if (features.spectralCentroidStdDev < 70) {
        spectralScore += 35;
      } else if (features.spectralCentroidStdDev < 120) {
        spectralScore += 20;
      }

      // 2. Prosody & Pitch Anomaly
      // Monotone F0 Pitch Std Dev (Neural TTS F0 constraint artifact)
      if (features.pitchStdDev < 4.5) {
        prosodyScore += 50; // Extremely robotic monotone
      } else if (features.pitchStdDev < 10.0) {
        prosodyScore += 30;
      }

      // ZCR Uniformity (unnatural fricative/silence transition)
      if (features.zcr > 0.85) {
        prosodyScore += 25;
      }

      // 3. Biometric Comparison if Enrolled Baseline Exists
      if (enrolledProfile && features.pitchMean) {
        const pitchDelta = Math.abs(features.pitchMean - enrolledProfile.baselinePitchMean);
        if (pitchDelta > 40) {
          biometricMismatchScore += 60;
        } else if (pitchDelta > 20) {
          biometricMismatchScore += 30;
        }
      }
    }

    // 4. Contextual Metadata Risk
    if (callMetadata.isSpoofedCallerID) metadataScore += 35;
    if (callMetadata.highRiskTransaction) metadataScore += 30;
    if (callMetadata.voipProxyDetected) metadataScore += 25;
    if (callMetadata.unusualTime) metadataScore += 10;

    // Clamp sub-scores
    spectralScore = Math.min(100, Math.max(0, spectralScore));
    prosodyScore = Math.min(100, Math.max(0, prosodyScore));
    biometricMismatchScore = Math.min(100, Math.max(0, biometricMismatchScore));
    metadataScore = Math.min(100, Math.max(0, metadataScore));

    // Weighted Risk Combination
    const overallRisk = (
      this.weights.spectral * spectralScore +
      this.weights.prosody * prosodyScore +
      this.weights.biometric * biometricMismatchScore +
      this.weights.metadata * metadataScore
    );

    const roundedRisk = parseFloat(overallRisk.toFixed(1));

    let alertLevel = 'LOW';
    let verdict = 'VERIFIED_GENUINE_HUMAN';
    let recommendation = 'Proceed with normal verification procedures.';

    if (roundedRisk >= 85.0) {
      alertLevel = 'CRITICAL';
      verdict = 'CRITICAL_RISK_AI_VOICE_CLONE';
      recommendation = 'IMMEDIATE BLOCK. High-probability synthetic speech impersonation attack detected. Require out-of-band supervisor verification.';
    } else if (roundedRisk >= 65.0) {
      alertLevel = 'HIGH';
      verdict = 'HIGH_RISK_SUSPECTED_IMPERSONATION';
      recommendation = 'Trigger dynamic challenge-response phrase test and initiate 2FA call-back.';
    } else if (roundedRisk >= 35.0) {
      alertLevel = 'MEDIUM';
      verdict = 'MODERATE_ANOMALY_INCONCLUSIVE';
      recommendation = 'Step-up security prompt required before proceeding with high-value approval.';
    }

    return {
      riskScore: roundedRisk,
      alertLevel: alertLevel,
      verdict: verdict,
      recommendation: recommendation,
      breakdown: {
        spectralScore: spectralScore,
        prosodyScore: prosodyScore,
        biometricMismatchScore: biometricMismatchScore,
        metadataScore: metadataScore
      },
      challengePhrases: [
        "Please recite: 'Alpha Bravo 794 Phoenix'",
        "Please repeat today's date backwards: 'Eleven September Twenty-Six'",
        "Say the current authorization phrase: 'VeriVoice 482'"
      ]
    };
  }
}

window.RiskEngine = RiskEngine;
