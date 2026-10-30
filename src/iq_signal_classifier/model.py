"""Small 1D CNN for classifying two-channel I/Q baseband signals."""

from __future__ import annotations

import torch
from torch import nn


class IQSignalCNN(nn.Module):
    """Classify BPSK, QPSK, AM, and noise-only I/Q signals.

    Input shape: (batch, 2, signal_length). Output: raw class logits.
    CrossEntropyLoss expects logits, so softmax is applied only at inference.
    """

    def __init__(self, num_classes: int = 4, input_channels: int = 2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if input_channels != 2:
            raise ValueError("I/Q signals require exactly two input channels")

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits for a batch of I/Q signals."""
        if signals.ndim != 3:
            raise ValueError("signals must have shape (batch, 2, signal_length)")
        if signals.shape[1] != 2:
            raise ValueError("signals must contain I and Q channels")
        if signals.shape[2] < 4:
            raise ValueError("signal_length must be at least 4")
        return self.classifier(self.features(signals))

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        """Convert model logits to class probabilities for inference."""
        return torch.softmax(logits, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
