"""
VoiceSentinel AI - PyTorch Real-Time Inference Engine
------------------------------------------------------
Loads trained PyTorch VoiceDeepfakeDetector weights and performs real-time
inference on raw audio bytes (WAV, FLAC, OGG, MP3, AAC), 1D float arrays, or file paths.
Aggregates predictions across multiple 3-second chunks for long audio.
"""

import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from ml_engine.model import VoiceDeepfakeDetector
from ml_engine.preprocessing import preprocess_audio_pipeline, decode_audio_bytes, TARGET_SAMPLE_RATE


class VoiceDeepfakeInference:
    """
    Real-Time PyTorch Model Inference Engine.
    """
    def __init__(self, model_path=None, device="cpu"):
        self.device = device
        self.model = VoiceDeepfakeDetector(in_channels=1, num_classes=1).to(self.device)
        self.is_model_loaded = False
        self.model_path = model_path or os.path.join("ml_engine", "voice_deepfake_detector.pth")

        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_model_loaded = True
                print(f"[+] Loaded PyTorch deepfake detection weights from '{self.model_path}'")
            except Exception as e:
                print(f"[!] Failed loading PyTorch weights: {e}")
        else:
            print(f"[!] PyTorch model weights file '{self.model_path}' not found. Will report MODEL_NOT_FOUND until trained.")

    def reload_model(self):
        """Attempts to re-load model weights if newly trained."""
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_model_loaded = True
            return True
        return False

    def predict_audio_array(self, audio_data, sample_rate=TARGET_SAMPLE_RATE):
        """
        Runs PyTorch inference on a 1D float32 numpy audio array.
        Multi-chunk prediction aggregation for long audio.
        """
        if not self.is_model_loaded:
            if not self.reload_model():
                return {
                    "error": "MODEL_NOT_FOUND",
                    "message": "PyTorch model weights 'voice_deepfake_detector.pth' not found. Please run training pipeline.",
                    "aiProbability": 0.0,
                    "verdict": "UNKNOWN"
                }

        start_time = time.time()

        if audio_data is None or len(audio_data) == 0:
            return {
                "error": "INVALID_AUDIO",
                "message": "Input audio data is empty or invalid.",
                "aiProbability": 0.0,
                "verdict": "INVALID_AUDIO"
            }

        print("[PyTorch] Inference started")
        spectrograms, chunks = preprocess_audio_pipeline(audio_data, sample_rate)
        chunk_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for spec in spectrograms:
                spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits = self.model(spec_tensor)
                prob = torch.sigmoid(logits).item()
                chunk_probabilities.append(float(prob))

        aggregated_prob = float(np.mean(chunk_probabilities)) if len(chunk_probabilities) > 0 else 0.0
        elapsed_ms = round((time.time() - start_time) * 1000.0, 2)

        verdict = "AI_GENERATED" if aggregated_prob >= 0.50 else "GENUINE"
        print("[PyTorch] Inference completed")

        return {
            "status": "SUCCESS",
            "aiProbability": round(aggregated_prob, 4),
            "aiProbabilityPercent": round(aggregated_prob * 100.0, 1),
            "humanProbability": round(1.0 - aggregated_prob, 4),
            "humanProbabilityPercent": round((1.0 - aggregated_prob) * 100.0, 1),
            "verdict": verdict,
            "totalChunksAnalyzed": len(chunk_probabilities),
            "chunkProbabilities": [round(p, 4) for p in chunk_probabilities],
            "inferenceLatencyMs": elapsed_ms,
            "modelVersion": "PyTorch-CNN-v2.0"
        }

    def predict_audio_bytes(self, raw_bytes, filename=""):
        """
        Decodes raw audio file bytes (WAV, FLAC, OGG, MP3, AAC) via FFmpeg decoder
        and passes through PyTorch inference pipeline.
        """
        try:
            decoded = decode_audio_bytes(raw_bytes, filename=filename)
            res = self.predict_audio_array(
                decoded["audio"],
                decoded["sample_rate"]
            )
            if "status" in res and res["status"] == "SUCCESS":
                res["audioDurationSec"] = round(decoded["duration_seconds"], 2)
                res["originalSampleRateHz"] = decoded["sample_rate"]
                res["originalChannels"] = "Mono" if decoded["channels"] == 1 else f"{decoded['channels']} Channels"
                res["originalFormat"] = decoded["original_format"]
            return res
        except Exception as e:
            stderr_msg = getattr(e, 'detail', None) or str(e)
            return {
                "error": "AUDIO_DECODE_FAILED",
                "message": f"FFmpeg decode error: {stderr_msg}",
                "verdict": "UNKNOWN"
            }

    def predict_file(self, file_path):
        """Reads audio file path and runs PyTorch inference."""
        if not os.path.exists(file_path):
            return {"error": "FILE_NOT_FOUND", "message": f"File '{file_path}' does not exist."}
        with open(file_path, "rb") as f:
            return self.predict_audio_bytes(f.read(), filename=os.path.basename(file_path))


# Global Inference Engine Instance
_INFERENCE_ENGINE = None

def get_inference_engine():
    global _INFERENCE_ENGINE
    if _INFERENCE_ENGINE is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _INFERENCE_ENGINE = VoiceDeepfakeInference(device=device)
    return _INFERENCE_ENGINE
