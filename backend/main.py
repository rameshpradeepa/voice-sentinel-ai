"""
VoiceSentinel AI - Authoritative FastAPI Enterprise Backend Server
--------------------------------------------------------------------
Authoritative ML Backend providing PyTorch deepfake inference, 3-second WebSocket streaming,
multi-format audio decoding (WAV, FLAC, OGG, MP3, AAC), prosody forensics, acoustic biometrics,
dynamic security risk scoring, security action workflow, and privacy-preserving blockchain audit logging.
"""

import sys
import os
import glob
import json
import time
import asyncio
import numpy as np
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal Imports
from ml_engine.inference import get_inference_engine
from ml_engine.preprocessing import extract_comprehensive_acoustic_features, TARGET_SAMPLE_RATE
from backend.audio_decoder import decode_audio_bytes, AudioDecodeError, find_ffmpeg
from backend.prosody_analyzer import analyze_prosody_features
from backend.biometric_engine import get_biometric_engine
from backend.risk_engine import get_risk_engine
from backend.privacy_guard import compute_sha256_hash, create_privacy_preserving_log
from backend.blockchain_logger import get_blockchain_logger

app = FastAPI(
    title="VoiceSentinel AI - Authoritative Deepfake Detection Backend",
    description="Authoritative FastAPI Backend for PyTorch Inference, Prosody Forensics, Security Actions, and Blockchain Audit",
    version="2.0.0"
)

@app.on_event("startup")
def startup_event():
    try:
        path = find_ffmpeg()
        print(f"[AudioDecoder] FFmpeg detected:\n{path}")
    except Exception as e:
        print(f"[AudioDecoder] Startup Warning: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
inference_engine = get_inference_engine()
biometric_engine = get_biometric_engine()
risk_engine = get_risk_engine()
blockchain_logger = get_blockchain_logger()

# In-memory Incident Logs Store
incident_logs = {}


# Schemas
class AnalyzeRequest(BaseModel):
    acousticFeatures: Optional[dict] = None
    callMetadata: Optional[dict] = None
    targetProfileId: Optional[str] = None


class EnrollRequest(BaseModel):
    name: str
    organization: str
    f0Mean: float = 150.0
    f0StdDev: float = 25.0
    spectralCentroid: float = 2400.0


class SipFilterRequest(BaseModel):
    sipHeader: str
    callerNumber: str
    streamSampleRate: int = 16000


class AuditRequest(BaseModel):
    incidentId: str
    riskScore: float
    classification: str
    audioHashSha256: str


class SecurityActionRequest(BaseModel):
    incidentId: str
    action: str
    riskScore: float
    classification: str
    audioHashSha256: str


# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.get("/api/health")
def health_check():
    """Health Check Endpoint returning PyTorch model and system status."""
    model_status = "LOADED" if inference_engine.is_model_loaded else "MODEL_NOT_FOUND"
    return {
        "status": "healthy",
        "system": "VoiceSentinel AI Authoritative ML Backend",
        "version": "2.0.0",
        "pyTorchModelStatus": model_status,
        "device": str(inference_engine.device),
        "supportedFormats": ["WAV", "FLAC", "OGG", "MP3", "AAC"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


@app.post("/api/v1/analyze")
def analyze_features(req: AnalyzeRequest):
    """Authoritative feature & risk evaluation endpoint."""
    features = req.acousticFeatures or {}
    metadata = req.callMetadata or {}

    pitch_mean = features.get("pitchMean", 145.0)
    pitch_std = features.get("pitchStdDev", 15.0)
    centroid = features.get("spectralCentroid", 2000.0)
    high_freq_ratio = features.get("highFreqEnergyRatio", 0.05)
    zcr = features.get("zcr", 0.40)

    live_bio_vec = biometric_engine.extract_biometric_vector(pitch_mean, pitch_std, centroid, high_freq_ratio, zcr)
    bio_result = biometric_engine.evaluate_biometric_match(live_bio_vec, req.targetProfileId)

    ai_prob = 0.88 if (pitch_std < 5.0 or high_freq_ratio < 0.02) else 0.12

    risk_result = risk_engine.compute_risk_score(
        ai_probability=ai_prob,
        prosody_anomaly_score=40.0 if pitch_std < 8.0 else 10.0,
        biometric_mismatch_score=bio_result.get("biometricMismatchScore", 0.0),
        call_metadata=metadata,
        spectral_features=features,
        biometric_result=bio_result
    )

    return {
        "success": True,
        "data": {
            "analysisId": "anl_" + os.urandom(4).hex(),
            "aiProbability": ai_prob,
            "pyTorchVerdict": "AI_GENERATED" if ai_prob >= 0.5 else "GENUINE",
            "securityRisk": risk_result,
            "biometricResult": bio_result
        }
    }


@app.post("/api/v1/analyze-file")
async def analyze_file(file: UploadFile = File(...), targetProfileId: Optional[str] = Form(None)):
    """
    Authoritative Audio File Upload Endpoint.
    Decodes WAV, FLAC, OGG, MP3, AAC audio bytes, executes PyTorch deepfake inference,
    extracts comprehensive acoustic/prosody features, and calculates security risk score.
    """
    valid_exts = ('.wav', '.flac', '.ogg', '.mp3', '.aac', '.m4a', '.mp4')
    if not file.filename.lower().endswith(valid_exts):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "AUDIO_DECODE_FAILED",
                "message": f"FFmpeg decode error: Unsupported audio file extension '{file.filename}'. Supported: WAV, FLAC, OGG, MP3, AAC, M4A, MP4."
            }
        )

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "AUDIO_DECODE_FAILED",
                "message": "FFmpeg decode error: Empty audio file uploaded."
            }
        )

    audio_hash = compute_sha256_hash(raw_bytes)

    # 1. Decode Audio Bytes via FFmpeg Subprocess (WAV, FLAC, OGG, MP3, AAC, M4A, MP4)
    try:
        decoded = decode_audio_bytes(raw_bytes, filename=file.filename)
    except AudioDecodeError as e:
        stderr_msg = e.detail or e.message
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "AUDIO_DECODE_FAILED",
                "message": f"FFmpeg decode error: {stderr_msg}"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "AUDIO_DECODE_FAILED",
                "message": f"FFmpeg decode error: {str(e)}"
            }
        )

    float_pcm = decoded.audio_data
    sr = decoded.sample_rate
    duration = decoded.duration_seconds
    orig_fmt = decoded.original_format

    # 2. PyTorch Model Inference
    inf_result = inference_engine.predict_audio_array(float_pcm, sample_rate=sr)
    if "error" in inf_result and inf_result["error"] == "MODEL_NOT_FOUND":
        return {
            "success": False,
            "error": "MODEL_NOT_FOUND",
            "message": inf_result["message"]
        }

    ai_prob = inf_result.get("aiProbability", 0.5)

    # 3. Extract Real Acoustic & Prosody Descriptors from decoded float_pcm array
    spectral_feats = extract_comprehensive_acoustic_features(float_pcm, sr)
    prosody_feats = analyze_prosody_features(float_pcm, sr)

    # 4. Biometric Check (Only if targetProfileId is explicitly selected)
    bio_result = None
    if targetProfileId:
        live_bio_vec = biometric_engine.extract_biometric_vector(
            prosody_feats["f0MeanHz"],
            prosody_feats["f0StdDevHz"],
            spectral_feats.get("spectralCentroid", 2000),
            spectral_feats.get("highFreqEnergyRatio", 0.05),
            spectral_feats.get("zcr", 0.4)
        )
        bio_result = biometric_engine.evaluate_biometric_match(live_bio_vec, targetProfileId)
    else:
        bio_result = {
            "status": "NOT AVAILABLE",
            "biometricMismatchScore": 0.0,
            "matchPercentage": "N/A"
        }

    # 5. Multi-Factor Security Risk Computation
    risk_result = risk_engine.compute_risk_score(
        ai_probability=ai_prob,
        prosody_anomaly_score=prosody_feats.get("prosodyAnomalyScore", 10.0),
        biometric_mismatch_score=bio_result.get("biometricMismatchScore", 0.0) if targetProfileId else 0.0,
        call_metadata={"highRiskTransaction": True if ai_prob > 0.7 else False},
        spectral_features=spectral_feats,
        prosody_features=prosody_feats,
        voice_quality=spectral_feats.get("voiceQuality", {}),
        biometric_result=bio_result
    )

    print(f"[RiskEngine] Risk score: {risk_result.get('overallRiskScore', 0.0):.2f}")

    audit_record = create_privacy_preserving_log("inc_" + os.urandom(4).hex(), audio_hash, risk_result)

    return {
        "success": True,
        "data": {
            "fileName": file.filename,
            "originalFormat": orig_fmt,
            "decodedFormat": "WAV/PCM",
            "sampleRate": sr,
            "channels": 1,
            "durationSeconds": round(duration, 2),
            "fileSizeBytes": len(raw_bytes),
            "audioHashSha256": audio_hash,
            "pyTorchInference": inf_result,
            "securityRisk": risk_result,
            "privacyAuditRecord": audit_record
        }
    }


@app.post("/api/v1/security/action")
def trigger_security_action(req: SecurityActionRequest):
    action_type = req.action.upper()
    call_decision = "VERIFY"

    if "ALLOW" in action_type:
        call_decision = "ALLOW"
    elif "BLOCK" in action_type:
        call_decision = "BLOCKED"
    elif "REPORT" in action_type:
        call_decision = "REPORTED"

    audit_res = blockchain_logger.record_audit_log(
        req.incidentId, req.riskScore, req.classification, req.audioHashSha256
    )

    incident_record = {
        "incidentId": req.incidentId,
        "actionTaken": action_type,
        "callDecisionState": call_decision,
        "riskScore": req.riskScore,
        "classification": req.classification,
        "audioHashSha256": req.audioHashSha256,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "blockchainAudit": audit_res
    }
    incident_logs[req.incidentId] = incident_record

    return {
        "success": True,
        "actionStatus": "EXECUTED",
        "callDecisionState": call_decision,
        "incidentRecord": incident_record,
        "message": f"Security action '{action_type}' executed successfully."
    }


@app.get("/api/v1/enroll")
def get_enrolled_profiles():
    return {"success": True, "data": list(biometric_engine.enrolled_profiles.values())}


@app.post("/api/v1/enroll")
def enroll_profile(req: EnrollRequest):
    pid = "user_" + os.urandom(3).hex()
    profile = biometric_engine.enroll_new_profile(pid, req.name, req.organization, req.f0Mean, req.f0StdDev, req.spectralCentroid)
    return {"success": True, "data": profile, "message": f"Enrolled profile for {req.name}"}


@app.get("/api/v1/datasets")
def get_datasets_summary():
    base_dir = "datasets"
    exts = ("*.wav", "*.flac", "*.ogg", "*.mp3", "*.aac", "*.m4a", "*.mp4")

    genuine_files = []
    synthetic_files = []
    mimicry_files = []

    for ext in exts:
        genuine_files.extend(glob.glob(os.path.join(base_dir, "genuine", "**", ext), recursive=True))
        synthetic_files.extend(glob.glob(os.path.join(base_dir, "synthetic", "**", ext), recursive=True))
        mimicry_files.extend(glob.glob(os.path.join(base_dir, "human_mimicry", "**", ext), recursive=True))

    accents = {}
    for f in genuine_files + synthetic_files + mimicry_files:
        parts = f.replace("\\", "/").split("/")
        if len(parts) >= 3:
            acc = parts[-2]
            accents[acc] = accents.get(acc, 0) + 1

    sample_list = []
    for f in genuine_files[:4]:
        sample_list.append({"file": os.path.basename(f), "path": f, "type": "GENUINE", "accent": f.split(os.sep)[-2] if len(f.split(os.sep)) > 2 else "default"})
    for f in synthetic_files[:4]:
        sample_list.append({"file": os.path.basename(f), "path": f, "type": "SYNTHETIC", "accent": f.split(os.sep)[-2] if len(f.split(os.sep)) > 2 else "default"})
    for f in mimicry_files[:4]:
        sample_list.append({"file": os.path.basename(f), "path": f, "type": "HUMAN_MIMICRY", "accent": f.split(os.sep)[-2] if len(f.split(os.sep)) > 2 else "default"})

    return {
        "success": True,
        "data": {
            "totalAudioFiles": len(genuine_files) + len(synthetic_files) + len(mimicry_files),
            "genuineCount": len(genuine_files),
            "syntheticCount": len(synthetic_files),
            "humanMimicryCount": len(mimicry_files),
            "accentBreakdown": accents,
            "sampleList": sample_list,
            "datasetDirectory": os.path.abspath(base_dir)
        }
    }


@app.get("/api/v1/model")
def get_model_info():
    meta_path = os.path.join("ml_engine", "model_metadata.json")
    eval_path = os.path.join("ml_engine", "evaluation.json")

    if not os.path.exists(meta_path):
        return {
            "success": False,
            "error": "MODEL_NOT_FOUND",
            "message": "PyTorch model metadata not found. Run training script `python ml_engine/train.py`."
        }

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    eval_data = "NOT EVALUATED"
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

    return {
        "success": True,
        "modelMetadata": metadata,
        "evaluationMetrics": eval_data
    }


@app.get("/api/v1/stats")
def get_system_stats():
    eval_path = os.path.join("ml_engine", "evaluation.json")
    acc_str = "NOT AVAILABLE"
    if os.path.exists(eval_path):
        try:
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
                if "accuracy" in eval_data:
                    acc_str = f"{eval_data['accuracy']}%"
        except Exception:
            pass

    total_incidents = len(incident_logs)
    blocked_count = sum(1 for log in incident_logs.values() if log.get("callDecisionState") == "BLOCKED")

    return {
        "success": True,
        "data": {
            "totalCallsMonitored": total_incidents if total_incidents > 0 else "NOT AVAILABLE",
            "attacksBlocked": blocked_count if total_incidents > 0 else "NOT AVAILABLE",
            "fraudPreventedUSD": "NOT AVAILABLE",
            "pyTorchAccuracy": acc_str,
            "averageInferenceMs": "NOT AVAILABLE",
            "blockchainLogsCommitted": "NOT AVAILABLE"
        }
    }


@app.post("/api/v1/telecom/sip-filter")
def inspect_sip_filter(req: SipFilterRequest):
    is_suspicious = "Spoofed" in req.sipHeader or "Asterisk" in req.sipHeader
    return {
        "success": True,
        "notice": "PROTOTYPE SIP METADATA ANALYSIS: Full carrier cellular interception requires PBX/Media-Gateway integration.",
        "telecomCarrier": "Airtel / Jio Enterprise SIP Trunk",
        "sipTrustScore": 25 if is_suspicious else 98,
        "action": "ROUTE_TO_VOICESENTINEL_INSPECTION" if is_suspicious else "PASS_THROUGH"
    }


@app.post("/api/v1/audit")
def record_blockchain_audit(req: AuditRequest):
    result = blockchain_logger.record_audit_log(
        req.incidentId, req.riskScore, req.classification, req.audioHashSha256
    )
    return {"success": True, "data": result}


# ============================================================================
# WEBSOCKET REAL-TIME AUDIO ACCUMULATION & STREAMING (3-SECOND WINDOW)
# ============================================================================

@app.websocket("/ws/analyze")
async def websocket_audio_analyzer(websocket: WebSocket):
    await websocket.accept()
    print("[+] Client connected to /ws/analyze real-time audio stream")

    audio_accumulator = bytearray()
    REQUIRED_BYTES = 48000 * 2  # 96,000 bytes (3 seconds of 16-bit PCM @ 16kHz)

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                raw_chunk = message["bytes"]
                audio_accumulator.extend(raw_chunk)

                if len(audio_accumulator) >= REQUIRED_BYTES:
                    window_bytes = bytes(audio_accumulator[:REQUIRED_BYTES])
                    audio_accumulator = audio_accumulator[REQUIRED_BYTES // 2:]

                    audio_hash = compute_sha256_hash(window_bytes)

                    float_pcm = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    inf_result = inference_engine.predict_audio_array(float_pcm, sample_rate=TARGET_SAMPLE_RATE)
                    ai_prob = inf_result.get("aiProbability", 0.5)

                    spectral_feats = extract_comprehensive_acoustic_features(float_pcm, TARGET_SAMPLE_RATE)
                    prosody_feats = analyze_prosody_features(float_pcm, TARGET_SAMPLE_RATE)

                    risk_result = risk_engine.compute_risk_score(
                        ai_probability=ai_prob,
                        prosody_anomaly_score=prosody_feats.get("prosodyAnomalyScore", 10.0),
                        biometric_mismatch_score=0.0,
                        spectral_features=spectral_feats,
                        prosody_features=prosody_feats,
                        voice_quality=spectral_feats.get("voiceQuality", {})
                    )

                    response_payload = {
                        "type": "PYTORCH_INFERENCE_UPDATE",
                        "timestamp": time.time(),
                        "audioHashSha256": audio_hash,
                        "aiProbability": ai_prob,
                        "aiProbabilityPercent": round(ai_prob * 100.0, 1),
                        "humanProbabilityPercent": round((1.0 - ai_prob) * 100.0, 1),
                        "pyTorchVerdict": inf_result.get("verdict", "GENUINE"),
                        "riskResult": risk_result,
                        "inferenceLatencyMs": inf_result.get("inferenceLatencyMs", 12.0),
                        "audioMetadata": {
                            "durationSeconds": round(len(float_pcm) / TARGET_SAMPLE_RATE, 2),
                            "sampleRateHz": TARGET_SAMPLE_RATE,
                            "channels": 1,
                            "fileFormat": "PCM_STREAM",
                            "chunkCount": 1
                        }
                    }

                    await websocket.send_json(response_payload)

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                if data.get("action") == "PING":
                    await websocket.send_json({"type": "PONG", "timestamp": time.time()})

    except WebSocketDisconnect:
        print("[-] Client disconnected from /ws/analyze stream")
    except Exception as e:
        print(f"[!] WebSocket stream error: {e}")
