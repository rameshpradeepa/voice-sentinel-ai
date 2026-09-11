"""
VoiceSentinel AI - PyTorch Deepfake Voice Detector Model Architecture
----------------------------------------------------------------------
Lightweight 2D Convolutional Neural Network (CNN) operating on Log-Mel
Spectrogram tensors to classify speech into GENUINE vs AI_GENERATED.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceDeepfakeDetector(nn.Module):
    """
    PyTorch 2D CNN Deepfake Voice Detector.
    Input Shape: (Batch_Size, 1, N_MELS, Time_Steps) -> e.g. (B, 1, 64, 94)
    Output Shape: (Batch_Size, 1) raw logit output (sigmoid yields probability of AI_GENERATED).
    """
    def __init__(self, in_channels=1, num_classes=1):
        super(VoiceDeepfakeDetector, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Dense Layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (B, 1, 64, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, p=0.3, training=self.training)

        # Global Average Pooling -> (B, 128, 1, 1)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # (B, 128)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # (B, 1)
        return logits

    def predict_probability(self, x):
        """Helper method returning sigmoid probability [0.0, 1.0] of being AI_GENERATED."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            prob = torch.sigmoid(logits)
        return prob
