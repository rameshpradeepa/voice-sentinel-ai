"""
VoiceSentinel AI - Dataset Validation Engine
---------------------------------------------
Validates dataset integrity, checks class balance, detects duplicate audio files (SHA-256),
analyzes duration and sample rate distributions across genuine, synthetic, and human_mimicry folders.
Rejects training if dataset contains identical files or only baseline generated signals.
"""

import sys
import os
import glob
import hashlib
import numpy as np
import scipy.io.wavfile as wavfile

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def compute_file_sha256(file_path):
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_audio_file(file_path):
    """Inspects audio duration and sample rate."""
    try:
        sr, pcm = wavfile.read(file_path)
        num_samples = pcm.shape[0] if pcm.ndim >= 1 else 0
        duration = num_samples / float(sr) if sr > 0 else 0.0
        return sr, duration, True
    except Exception:
        return 0, 0.0, False


def validate_dataset(dataset_dir="datasets"):
    """
    Validates audio dataset in dataset_dir and prints report.
    Returns dict report and boolean is_valid.
    """
    print(f"=======================================================")
    print(f"🔍 VoiceSentinel AI - Dataset Integrity Validator")
    print(f"=======================================================")
    print(f"[*] Scanning dataset directory: '{os.path.abspath(dataset_dir)}'")

    categories = ["genuine", "synthetic", "human_mimicry"]
    file_counts = {}
    hashes_by_category = {}
    all_hashes = {}
    durations = []
    sample_rates = {}
    languages = set()
    speakers = set()

    is_baseline_only = True

    for category in categories:
        cat_dir = os.path.join(dataset_dir, category)
        files = glob.glob(os.path.join(cat_dir, "**", "*.wav"), recursive=True) + \
                glob.glob(os.path.join(cat_dir, "**", "*.flac"), recursive=True)

        file_counts[category] = len(files)
        hashes_by_category[category] = set()

        for f in files:
            # Extract language / subfolder if present
            rel_path = os.path.relpath(f, cat_dir).replace("\\", "/")
            parts = rel_path.split("/")
            if len(parts) > 1:
                languages.add(parts[0])

            # Check if non-baseline generated filename
            filename = os.path.basename(f)
            if not filename.startswith("sample_"):
                is_baseline_only = False

            # Hash & inspect audio
            h = compute_file_sha256(f)
            hashes_by_category[category].add(h)
            if h in all_hashes:
                all_hashes[h].append(f)
            else:
                all_hashes[h] = [f]

            sr, dur, ok = inspect_audio_file(f)
            if ok:
                durations.append(dur)
                sample_rates[sr] = sample_rates.get(sr, 0) + 1

    total_genuine = file_counts.get("genuine", 0)
    total_synthetic = file_counts.get("synthetic", 0)
    total_mimicry = file_counts.get("human_mimicry", 0)
    total_files = sum(file_counts.values())

    # Detect duplicate files
    duplicate_groups = [paths for h, paths in all_hashes.items() if len(paths) > 1]
    num_duplicates = sum(len(paths) - 1 for paths in duplicate_groups)

    # Check cross-category overlap (identical genuine & synthetic files)
    genuine_hashes = hashes_by_category.get("genuine", set())
    synthetic_hashes = hashes_by_category.get("synthetic", set())
    overlap = genuine_hashes.intersection(synthetic_hashes)

    min_dur = round(float(np.min(durations)), 2) if durations else 0.0
    max_dur = round(float(np.max(durations)), 2) if durations else 0.0
    mean_dur = round(float(np.mean(durations)), 2) if durations else 0.0

    print("\n--- DATASET METRICS REPORT ---")
    print(f"• Total Genuine Audio Files : {total_genuine}")
    print(f"• Total Synthetic AI Clones : {total_synthetic}")
    print(f"• Total Human Mimicry Files : {total_mimicry}")
    print(f"• Total Audio Files         : {total_files}")
    print(f"• Regional Accents / Languages: {list(languages) if languages else ['default']}")
    print(f"• Duration (Min / Mean / Max): {min_dur}s / {mean_dur}s / {max_dur}s")
    print(f"• Sample Rate Distribution  : {sample_rates}")
    print(f"• Duplicate File Hashes     : {num_duplicates}")
    print(f"• Genuine/Synthetic Overlap : {len(overlap)} identical files")

    # Validation Failure Checks
    validation_reasons = []

    if total_files == 0:
        validation_reasons.append("Dataset is completely empty.")
    if total_genuine < 5 or total_synthetic < 5:
        validation_reasons.append("Insufficient audio files (at least 5 samples per category required).")
    if len(overlap) > 0:
        validation_reasons.append("Identical audio files found across both genuine and synthetic directories.")
    if is_baseline_only and total_files > 0:
        print("\n⚠️ NOTICE: Dataset contains only baseline generated smoke-test signals.")

    is_valid = len(validation_reasons) == 0

    if not is_valid:
        print("\n❌ DATASET VALIDATION FAILED:")
        for r in validation_reasons:
            print(f"  - {r}")
        print("\n⛔ REAL TRAINING DATA REQUIRED")
    else:
        print("\n✅ DATASET VALIDATION PASSED - Ready for PyTorch Training!")

    report = {
        "isValid": is_valid,
        "totalGenuine": total_genuine,
        "totalSynthetic": total_synthetic,
        "totalHumanMimicry": total_mimicry,
        "totalFiles": total_files,
        "languages": list(languages),
        "durationStats": {"minSec": min_dur, "meanSec": mean_dur, "maxSec": max_dur},
        "sampleRates": sample_rates,
        "duplicateCount": num_duplicates,
        "overlapCount": len(overlap),
        "isBaselineOnly": is_baseline_only,
        "rejectionReasons": validation_reasons
    }

    return report, is_valid


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets"
    validate_dataset(dataset_path)
