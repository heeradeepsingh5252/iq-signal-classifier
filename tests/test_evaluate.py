"""Fast tests for held-out I/Q CNN evaluation and report generation."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from iq_signal_classifier.dataset import build_dataset_splits
from iq_signal_classifier.evaluate import (
    evaluate_checkpoint,
    evaluate_loader,
)
from iq_signal_classifier.model import IQSignalCNN
from iq_signal_classifier.signal_gen import CLASS_NAMES


class AlwaysClassZero(nn.Module):
    """Tiny deterministic model used to test metric calculations."""

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((signals.shape[0], len(CLASS_NAMES)), device=signals.device)
        logits[:, 0] = 1.0
        return logits


def test_evaluate_loader_calculates_accuracy_and_class_metrics() -> None:
    signals = torch.zeros(4, 2, 16)
    labels = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    loader = DataLoader(TensorDataset(signals, labels), batch_size=2)

    result = evaluate_loader(AlwaysClassZero(), loader)

    assert result.sample_count == 4
    assert result.accuracy == pytest.approx(0.5)
    assert result.confusion == [
        [2, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert result.per_class["BPSK"].precision == pytest.approx(0.5)
    assert result.per_class["BPSK"].recall == pytest.approx(1.0)
    assert result.per_class["QPSK"].recall == pytest.approx(0.0)
    assert result.per_class["NOISE"].support == 0


def test_evaluate_loader_rejects_wrong_number_of_logits() -> None:
    signals = torch.zeros(2, 2, 16)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(signals, labels), batch_size=2)

    class WrongOutput(nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.zeros(values.shape[0], 3)

    with pytest.raises(ValueError, match="logits"):
        evaluate_loader(WrongOutput(), loader)


def test_evaluate_checkpoint_writes_real_metrics_and_artifacts(tmp_path: Path) -> None:
    # Build a tiny checkpoint compatible with the real Phase 4 format.
    config = {
        "n_per_class": 10,
        "low_snr_n_per_class": 10,
        "signal_length": 64,
        "matched_snr_range": (10.0, 20.0),
        "low_snr_range": (-5.0, 5.0),
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "seed": 42,
    }
    splits = build_dataset_splits(
        n_per_class=config["n_per_class"],
        low_snr_n_per_class=config["low_snr_n_per_class"],
        signal_length=config["signal_length"],
        matched_snr_range=config["matched_snr_range"],
        low_snr_range=config["low_snr_range"],
        validation_fraction=config["validation_fraction"],
        test_fraction=config["test_fraction"],
        seed=config["seed"],
    )
    model = IQSignalCNN()
    checkpoint_path = tmp_path / "models" / "iq_classifier.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"num_classes": 4, "input_channels": 2},
            "class_names": list(CLASS_NAMES),
            "training_config": config,
            "normalizer": {
                "mean": splits.normalizer.mean.tolist(),
                "std": splits.normalizer.std.tolist(),
            },
            "best_epoch": 1,
            "best_val_loss": 0.5,
            "best_val_accuracy": 0.5,
            "history": [],
        },
        checkpoint_path,
    )

    result = evaluate_checkpoint(checkpoint_path, output_dir=tmp_path, batch_size=8)

    assert result.matched.sample_count == 6
    assert result.low_snr.sample_count == 40
    assert 0.0 <= result.matched.accuracy <= 1.0
    assert 0.0 <= result.low_snr.accuracy <= 1.0
    assert result.results_path.is_file()
    assert result.summary_path.is_file()
    assert result.chart_path.is_file()
    assert result.chart_path.stat().st_size > 0

    payload = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert payload["class_names"] == list(CLASS_NAMES)
    assert payload["matched_snr_test"]["sample_count"] == 6
    assert payload["low_snr_evaluation"]["sample_count"] == 40
    assert "noise-only" in payload["noise_class_snr_note"]


def test_evaluate_checkpoint_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        evaluate_checkpoint(tmp_path / "missing.pt")


def test_evaluate_checkpoint_rejects_normalizer_mismatch(tmp_path: Path) -> None:
    config = {
        "n_per_class": 10,
        "low_snr_n_per_class": 10,
        "signal_length": 64,
        "matched_snr_range": (10.0, 20.0),
        "low_snr_range": (-5.0, 5.0),
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "seed": 42,
    }
    splits = build_dataset_splits(
        n_per_class=10,
        low_snr_n_per_class=10,
        signal_length=64,
        seed=42,
    )
    model = IQSignalCNN()
    checkpoint_path = tmp_path / "bad-normalizer.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"num_classes": 4, "input_channels": 2},
            "class_names": list(CLASS_NAMES),
            "training_config": config,
            "normalizer": {
                "mean": (splits.normalizer.mean + 1.0).tolist(),
                "std": splits.normalizer.std.tolist(),
            },
            "best_epoch": 1,
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="mean differs"):
        evaluate_checkpoint(checkpoint_path, output_dir=tmp_path)
