"""Small CPU tests for the I/Q CNN training workflow."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from iq_signal_classifier.dataset import IQSignalDataset
from iq_signal_classifier.model import IQSignalCNN
from iq_signal_classifier.train import TrainingConfig, run_epoch, train_model


def test_run_epoch_returns_finite_training_metrics() -> None:
    signals = torch.randn(12, 2, 64)
    labels = torch.tensor([0, 1, 2, 3] * 3, dtype=torch.long)
    loader = DataLoader(IQSignalDataset(signals.numpy(), labels.numpy()), batch_size=4)
    model = IQSignalCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer)

    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.loss >= 0.0
    assert torch.isfinite(torch.tensor(metrics.loss))


def test_one_epoch_training_saves_checkpoint_history_and_plot(tmp_path: Path) -> None:
    config = TrainingConfig(
        n_per_class=10,
        low_snr_n_per_class=10,
        signal_length=64,
        batch_size=8,
        epochs=1,
        patience=1,
        num_threads=1,
        seed=7,
    )
    result = train_model(config=config, output_dir=tmp_path, verbose=False)

    assert result.epochs_completed == 1
    assert result.best_epoch == 1
    assert result.checkpoint_path.is_file()
    assert result.history_path.is_file()
    assert result.curve_path.is_file()
    assert result.curve_path.stat().st_size > 0

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["class_names"] == ["BPSK", "QPSK", "AM", "NOISE"]
    assert checkpoint["normalizer"]["mean"]
    assert checkpoint["normalizer"]["std"]
    assert checkpoint["best_epoch"] == 1

    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert len(history["history"]) == 1
    assert 0.0 <= history["history"][0]["val_accuracy"] <= 1.0


def test_training_config_rejects_invalid_epochs() -> None:
    config = TrainingConfig(epochs=0)
    try:
        config.validate()
    except ValueError as exc:
        assert "epochs" in str(exc)
    else:
        raise AssertionError("Invalid epoch count should be rejected")
