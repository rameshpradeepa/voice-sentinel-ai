"""
VoiceSentinel AI - PyTorch Audio Dataset Loader
------------------------------------------------
Recursively scans real audio files in datasets/genuine (label 0), datasets/synthetic (label 1),
and datasets/human_mimicry (label 0), preprocessing audio signals into Log-Mel Spectrogram tensors.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from torch.utils.data import Dataset, DataLoader
from ml_engine.preprocessing import preprocess_audio_pipeline


class VoiceDataset(Dataset):
    """
    PyTorch Dataset for Deepfake Voice Audio Files.
    Label 0: GENUINE (Natural human speech & human mimicry)
    Label 1: SYNTHETIC (AI-generated voice clone)
    """
    def __init__(self, dataset_dir="datasets", split="train", val_ratio=0.2, test_ratio=0.1, seed=42):
        super(VoiceDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.samples = []

        genuine_dir = os.path.join(dataset_dir, "genuine")
        synthetic_dir = os.path.join(dataset_dir, "synthetic")
        mimicry_dir = os.path.join(dataset_dir, "human_mimicry")

        genuine_files = glob.glob(os.path.join(genuine_dir, "**", "*.wav"), recursive=True) + \
                        glob.glob(os.path.join(genuine_dir, "**", "*.flac"), recursive=True)

        synthetic_files = glob.glob(os.path.join(synthetic_dir, "**", "*.wav"), recursive=True) + \
                          glob.glob(os.path.join(synthetic_dir, "**", "*.flac"), recursive=True)

        mimicry_files = glob.glob(os.path.join(mimicry_dir, "**", "*.wav"), recursive=True) + \
                        glob.glob(os.path.join(mimicry_dir, "**", "*.flac"), recursive=True)

        # Label 0 for genuine human speech and human mimicry, Label 1 for AI clones
        all_samples = [(f, 0.0, "genuine") for f in genuine_files] + \
                      [(f, 1.0, "synthetic") for f in synthetic_files] + \
                      [(f, 0.0, "human_mimicry") for f in mimicry_files]

        rng = np.random.RandomState(seed)
        rng.shuffle(all_samples)

        total = len(all_samples)
        if total > 0:
            n_test = int(total * test_ratio)
            n_val = int(total * val_ratio)
            n_train = total - n_test - n_val

            if split == "train":
                self.samples = all_samples[:n_train]
            elif split == "val":
                self.samples = all_samples[n_train:n_train + n_val]
            elif split == "test":
                self.samples = all_samples[n_train + n_val:]
            else:
                self.samples = all_samples
        else:
            self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, category = self.samples[idx]
        try:
            sr, raw_data = wavfile.read(file_path)
            if raw_data.dtype == np.int16:
                raw_data = raw_data.astype(np.float32) / 32768.0
            elif raw_data.dtype == np.int32:
                raw_data = raw_data.astype(np.float32) / 2147483648.0
            else:
                raw_data = raw_data.astype(np.float32)

            spectrograms, _ = preprocess_audio_pipeline(raw_data, sr)
            spec_tensor = torch.tensor(spectrograms[0], dtype=torch.float32)
            label_tensor = torch.tensor([label], dtype=torch.float32)
            return spec_tensor, label_tensor, file_path

        except Exception as e:
            zero_spec = torch.zeros((1, 64, 94), dtype=torch.float32)
            return zero_spec, torch.tensor([label], dtype=torch.float32), file_path


def create_dataloaders(dataset_dir="datasets", batch_size=8):
    """Creates PyTorch DataLoaders for train, val, and test splits."""
    train_dataset = VoiceDataset(dataset_dir, split="train")
    val_dataset = VoiceDataset(dataset_dir, split="val")
    test_dataset = VoiceDataset(dataset_dir, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if len(test_dataset) > 0 else None

    return train_loader, val_loader, test_loader
