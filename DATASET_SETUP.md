# VoiceSentinel AI - Audio Dataset Setup & Validation Guide

This guide explains dataset directory layout, validation requirements, training pipelines, and dataset inspection commands.

---

## 📁 1. Dataset Directory Structure

Place raw `.wav` or `.flac` audio files inside the `datasets/` directory structured as follows:

```
datasets/
├── genuine/                     # Natural Human Speech Audio (.wav / .flac)
│   ├── tamil/
│   ├── hindi/
│   └── indian_english/
├── synthetic/                   # AI-Generated / Cloned Audio (.wav / .flac)
│   ├── tamil/
│   ├── hindi/
│   └── indian_english/
└── human_mimicry/               # Real Human Speech Intentionally Imitating Voice
    ├── tamil/
    ├── hindi/
    └── indian_english/
```

### Audio File Requirements:
- **Formats**: WAV (16-bit PCM recommended) or FLAC.
- **Sample Rate**: Any sample rate (automatically resampled to 16,000 Hz mono during preprocessing).
- **Duration**: Minimum 3 seconds per file.

---

## 🛠️ 2. Step-by-Step Execution Commands

### Step A: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step B: Validate Audio Dataset
Before training, run the dataset validator tool:
```bash
python ml_engine/validate_dataset.py
```
**Validation Checks**:
- Total genuine, synthetic, and human mimicry counts.
- Duration statistics (min, mean, max) and sample rate distributions.
- Duplicate SHA-256 file hashes.
- Cross-category file overlap (rejects identical genuine and synthetic files).
- Displays `"REAL TRAINING DATA REQUIRED"` if rejected.

### Step C: Generate Baseline Smoke-Test Dataset (Optional)
```bash
python ml_engine/generate_sample_dataset.py
```

### Step D: Train the PyTorch Deepfake Classifier Model
```bash
python ml_engine/train.py
```

### Step E: Evaluate Model Performance
```bash
python ml_engine/evaluate.py
```

### Step F: Run Automated Unit & Integration Tests
```bash
pytest tests/
```

### Step G: Launch FastAPI Enterprise Server
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
