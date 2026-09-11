"""
VoiceSentinel AI - PyTorch Training Pipeline
---------------------------------------------
Trains the PyTorch VoiceDeepfakeDetector CNN model on genuine vs. synthetic audio samples,
evaluates performance metrics, and saves voice_deepfake_detector.pth and model_metadata.json.
"""

import sys
import os
import json
import time

# Ensure project root is in sys.path for direct python script invocation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from ml_engine.model import VoiceDeepfakeDetector
from ml_engine.dataset_loader import create_dataloaders
from ml_engine.generate_sample_dataset import generate_dataset_directory
from ml_engine.evaluate import evaluate_model_on_dataloader


def train_model(dataset_dir="datasets", epochs=8, batch_size=8, lr=0.001, device="cpu"):
    """
    Executes complete training loop for PyTorch deepfake voice detector.
    """
    print(f"[*] Starting PyTorch Voice Deepfake Detector Training Pipeline")
    print(f"[*] Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")

    # Ensure dataset exists
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print("[!] Datasets directory empty. Generating baseline audio samples...")
        generate_dataset_directory(dataset_dir)

    train_loader, val_loader, test_loader = create_dataloaders(dataset_dir, batch_size=batch_size)

    if train_loader is None or len(train_loader.dataset) == 0:
        print("[!] Generating WAV audio dataset...")
        generate_dataset_directory(dataset_dir)
        train_loader, val_loader, test_loader = create_dataloaders(dataset_dir, batch_size=batch_size)

    model = VoiceDeepfakeDetector(in_channels=1, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for specs, labels, _ in train_loader:
            specs = specs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * specs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = (correct / max(1, total)) * 100.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            if val_loader is not None:
                for specs, labels, _ in val_loader:
                    specs = specs.to(device)
                    labels = labels.to(device)
                    outputs = model(specs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * specs.size(0)
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

        val_epoch_loss = val_loss / max(1, val_total)
        val_epoch_acc = (val_correct / max(1, val_total)) * 100.0 if val_total > 0 else epoch_acc

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1f}% | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.1f}%")

    training_time_sec = round(time.time() - start_time, 2)

    # Save Model Weights
    output_model_path = os.path.join("ml_engine", "voice_deepfake_detector.pth")
    torch.save(model.state_dict(), output_model_path)
    print(f"[+] Saved trained PyTorch model weights to: '{output_model_path}'")

    # Evaluate on test loader
    eval_metrics = evaluate_model_on_dataloader(model, test_loader if test_loader else val_loader, device=device)
    
    # Save Evaluation JSON
    eval_path = os.path.join("ml_engine", "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"[+] Saved evaluation metrics to: '{eval_path}'")

    # Save Model Metadata JSON
    metadata = {
        "modelName": "VoiceDeepfakeDetector (2D-CNN)",
        "framework": "PyTorch",
        "version": "2.0.0",
        "sampleRateHz": 16000,
        "inputChannels": 1,
        "nMels": 64,
        "nFFT": 1024,
        "hopLength": 512,
        "chunkDurationSec": 3.0,
        "totalParameters": sum(p.numel() for p in model.parameters()),
        "trainedEpochs": epochs,
        "trainingDurationSec": training_time_sec,
        "datasetSamples": len(train_loader.dataset) + (len(val_loader.dataset) if val_loader else 0),
        "targetClasses": ["GENUINE", "AI_GENERATED"],
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "TRAINED_AND_ACTIVE"
    }

    metadata_path = os.path.join("ml_engine", "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved model metadata to: '{metadata_path}'")

    return model, metadata, eval_metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(epochs=6, device=device)
