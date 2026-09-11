# VoiceSentinel AI: PyTorch Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

Developed for **AICTE Cyber Security Cell** Problem Statement: *AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks*.

---

## 🌟 Solution Overview

**VoiceSentinel AI** is an authoritative cybersecurity framework designed to detect, flag, and mitigate synthetic voice cloning and deepfake impersonation attacks across telephonic, VoIP, and banking authorization channels.

The system uses an **Authoritative FastAPI Machine Learning Backend** with a **PyTorch 2D Convolutional Neural Network (CNN) Classifier**, **3-Second WebSocket Stream Accumulator**, **Real Prosody Analysis**, **Expanded Acoustic Feature Extraction (Jitter, Shimmer, Formants, MFCCs)**, **Acoustic Biometric Vector Similarity**, **Human Mimicry Security Interpretation Layer**, **Interactive Security Actions Panel**, and **Privacy-Preserving Blockchain Audit Logging**.

---

## 📁 Repository Structure

```
voicecloning/
├── package.json                   # Node.js configuration & run scripts
├── requirements.txt               # Python dependencies (PyTorch, FastAPI, Librosa, Web3)
├── README.md                      # Comprehensive system documentation
├── DATASET_SETUP.md               # Guide for audio dataset setup, validation, training
├── IMPLEMENTATION_STATUS.md       # Status matrix of all system components
├── conftest.py                    # Pytest configuration file
├── backend/                       # Authoritative FastAPI Enterprise Server
│   ├── main.py                    # REST APIs & 3-second WebSocket /ws/analyze handler
│   ├── prosody_analyzer.py        # F0 pitch mean/std, pitch range, pause ratio
│   ├── biometric_engine.py        # 5D Acoustic Biometric Profile & Cosine Similarity
│   ├── risk_engine.py             # Multi-factor security risk & interpretation layer
│   ├── privacy_guard.py           # Memory buffer cleanup & SHA-256 audio hashing
│   └── blockchain_logger.py       # Local Web3 / Ethereum EVM audit logger
├── ml_engine/                     # PyTorch Machine Learning Engine
│   ├── model.py                   # VoiceDeepfakeDetector (2D-CNN architecture)
│   ├── preprocessing.py           # 16kHz resampling, mono, silence trim, Log-Mel Spectrogram
│   ├── dataset_loader.py          # PyTorch Dataset for genuine, synthetic, human_mimicry
│   ├── validate_dataset.py        # Dataset integrity & duplicate SHA-256 validator
│   ├── generate_sample_dataset.py # Baseline smoke-test WAV sample generator
│   ├── train.py                   # Model training script saving voice_deepfake_detector.pth
│   ├── evaluate.py                # Model evaluation (Accuracy, F1, Precision, Recall, ROC-AUC)
│   └── inference.py               # Real-time PyTorch inference engine with chunk aggregation
├── datasets/                      # Real Audio Datasets Directory
│   ├── genuine/                   # Genuine human speech WAV files (tamil, hindi, indian_english)
│   ├── synthetic/                 # Synthetic AI clone WAV files (tamil, hindi, indian_english)
│   └── human_mimicry/             # Real human mimicry WAV files (tamil, hindi, indian_english)
├── blockchain/                    # Solidity Audit Logging Module
│   └── contracts/
│       └── VoiceSentinelAudit.sol # Smart Contract for logging SHA-256 audio hashes on-chain
├── public/                        # Web Application Dashboard
│   ├── index.html                 # Preserved Glassmorphic Dark UI (Single Unified View)
│   ├── styles.css                 # Cyber security dark glassmorphism stylesheet
│   ├── js/
│   │   ├── audio_engine.js        # Web Audio API DSP + 3s WebSocket binary audio streamer
│   │   ├── spectrogram_renderer.js# HTML5 2D Canvas Waterfall Spectrogram & Waveform
│   │   ├── app.js                 # Authoritative Frontend Controller & Security Actions
│   │   ├── call_simulator.js      # Social Engineering Scenario Simulator
│   │   └── api_client.js          # REST & WebSocket API client
│   └── assets/
│       └── dataset_samples.json   # Browser dataset samples index
└── tests/                         # Automated Unit & Integration Test Suite (8 Test Files)
    ├── test_preprocessing.py      # Audio resampling, chunking, and Mel-spectrogram shape
    ├── test_model.py              # PyTorch model forward pass tensor shapes
    ├── test_inference.py          # Real-time PyTorch inference test
    ├── test_backend.py            # FastAPI REST endpoints & security actions test
    ├── test_prosody.py            # Prosody feature analyzer test
    ├── test_risk_engine.py        # Multi-factor security risk score & human mimicry test
    ├── test_dataset_validation.py # Dataset validation tool test
    └── test_sha256.py             # SHA-256 hashing & privacy buffer cleanup test
```

---

## ✨ Core System Capabilities

### 1. PyTorch Deepfake Classifier Model (`ml_engine/`)
- **Input Specs**: Supports WAV, FLAC, OGG, MP3, and AAC (including WhatsApp voice notes). Automatically decoded, converted to 1-channel mono at 16,000 Hz, and split into fixed 3-second chunks ($16,000 \times 3 = 48,000$ samples).
- **Log-Mel Spectrogram**: $N_{\text{mels}} = 64$, $N_{\text{fft}} = 1024$, $\text{hop\_length} = 512$.
- **Architecture**: 2D Convolutional Neural Network with BatchNorm2d, ReLU activation, MaxPool2d, AdaptiveAvgPool2d, Dropout (0.4), and Linear classifier outputting logits.

### 2. Security Interpretation Layer & Human Mimicry Distinction (`backend/risk_engine.py`)
Differentiates between:
- `GENUINE HUMAN VOICE` ($P_{\text{AI}} < 0.35$ & Biometric Mismatch $< 50\%$)
- `AI VOICE CLONE SUSPECTED` ($P_{\text{AI}} \ge 0.50$)
- `HUMAN MIMICRY / POSSIBLE IMPERSONATION` (High human probability but Biometric Mismatch $\ge 65\%$)
- `UNKNOWN / INCONCLUSIVE`

### 3. Dynamic Multi-Factor Security Risk Score (`backend/risk_engine.py`)
Normalized 0 to 100 overall Security Risk Score:
$$\text{Risk Score} = 0.55 \times (P_{\text{AI}} \times 100) + 0.20 \times S_{\text{prosody}} + 0.15 \times S_{\text{biometric\_mismatch}} + 0.10 \times S_{\text{metadata}}$$

### 4. Interactive Security Actions & Decision States (`POST /api/v1/security/action`)
Supports interactive actions: `ALLOW CALL`, `VERIFY IDENTITY`, `CHALLENGE PHRASE`, `REQUIRE MFA`, `BLOCK CALL`, `BLOCK TRANSACTION`, `REPORT INCIDENT`, `ESCALATE TO SOC`.
Updates decision states: `ALLOW`, `VERIFY`, `REPORTED`, `BLOCKED`.

---

## 🚀 Execution Commands

### 1. Install Dependencies
```bash
pip install -r requirements.txt
npm install
```

### 2. Validate Dataset Integrity
```bash
python ml_engine/validate_dataset.py
```

### 3. Generate Baseline Dataset (Optional)
```bash
python ml_engine/generate_sample_dataset.py
```

### 4. Train PyTorch Deepfake Model
```bash
python ml_engine/train.py
```

### 5. Evaluate Model Metrics
```bash
python ml_engine/evaluate.py
```

### 6. Run Automated Test Suite
```bash
pytest tests/
```

### 7. Start FastAPI Authoritative ML Backend Server
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
- Open Swagger API Documentation: `http://localhost:8000/docs`
- Open UI Dashboard: `http://localhost:3000` or open [`public/index.html`](file:///c:/Users/D%20E%20L%20L/OneDrive/Desktop/voicecloning/public/index.html) in your browser.

---

## 📜 License
Developed for academic and cybersecurity research under the MIT License.
