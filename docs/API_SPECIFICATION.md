# VoiceSentinel AI - Enterprise REST & gRPC API Specification

## 1. Overview
VoiceSentinel AI exposes REST and gRPC interfaces for integration with Core Banking Systems, Telephony SIP Trunks, Enterprise Contact Centers, and Call Verification workflows.

---

## 2. Endpoints

### 2.1 Audio Stream Analysis Endpoint
- **URL**: `/api/v1/analyze`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Body
```json
{
  "acousticFeatures": {
    "pitchMean": 144.2,
    "pitchStdDev": 3.1,
    "spectralCentroid": 1820.0,
    "spectralCentroidStdDev": 55.0,
    "highFreqEnergyRatio": 0.015,
    "zcrUniformity": 0.94
  },
  "callMetadata": {
    "callerNumber": "+91 98765 43210",
    "isSpoofedCallerID": true,
    "highRiskTransaction": true,
    "targetProfileId": "cxo_001",
    "voipProxyDetected": true
  }
}
```

#### Response (200 OK)
```json
{
  "success": true,
  "data": {
    "analysisId": "anl_9f2a81b3",
    "timestamp": "2026-09-11T10:55:00Z",
    "overallRiskScore": 94.2,
    "verdict": "CRITICAL_RISK_AI_VOICE_CLONE",
    "alertLevel": "CRITICAL",
    "recommendation": "IMMEDIATE BLOCK. Trigger out-of-band MFA and notify Cyber Security Operations.",
    "breakdown": {
      "spectralArtifactScore": 95.0,
      "prosodyAnomalyScore": 90.0,
      "biometricMismatchScore": 100.0,
      "metadataRiskScore": 90.0
    },
    "detectedArtifacts": [
      "Neural Vocoder Phase Discontinuity",
      "Micro-Pitch Monotone Constraint",
      "Speaker Identity Vector Drift",
      "VoIP Header Spoofing Signature"
    ]
  }
}
```

---

### 2.2 Telecom SIP Trunk Route Inspection
- **URL**: `/api/v1/telecom/sip-filter`
- **Method**: `POST`

#### Request Body
```json
{
  "sipHeader": "SIP/2.0 Via: Asterisk-Spoofed-Relay-Node",
  "callerNumber": "+91 98765 43210",
  "streamSampleRate": 16000
}
```

#### Response Body
```json
{
  "success": true,
  "telecomCarrier": "Airtel / Jio Enterprise SIP Trunk",
  "routeInspection": "FLAGGED_VOIP_RELAY",
  "sipHeaderTrustScore": 24,
  "action": "ROUTE_TO_VOICE_SENTINEL_INSPECTION"
}
```

---

### 2.3 Speaker Voiceprint Enrollment
- **URL**: `/api/v1/enroll`
- **Method**: `POST`

#### Request Body
```json
{
  "name": "Rajesh Sharma (CFO)",
  "organization": "Apex National Bank",
  "baselinePitchMean": 142.5,
  "baselineSpectralCentroid": 2150.8
}
```

---

### 2.4 Voice Cloning Datasets Query
- **URL**: `/api/v1/datasets`
- **Method**: `GET`
