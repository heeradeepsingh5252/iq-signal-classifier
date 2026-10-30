"""Train the small I/Q CNN using matched-SNR data only.

The matched-SNR test set and independent low-SNR set are deliberately not
used here. They are reserved for the Phase 5 robustness evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from iq_signal_classifier.dataset import build_dataset_splits, create_dataloaders
from iq_signal_classifier.model import IQSignalCNN, count_parameters
from iq_signal_classifier.signal_gen import CLASS_NAMES


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a reproducible, CPU-friendly training run."""

    n_per_class: int = 100
    low_snr_n_per_class: int = 100
    signal_length: int = 256
    matched_snr_range: tuple[float, float] = (10.0, 20.0)
    low_snr_range: tuple[float, float] = (-5.0, 5.0)
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42
    num_threads: int = 1

    def validate(self) -> None:
        """Reject invalid values before dataset generation or training."""
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.num_threads < 1:
            raise ValueError("num_threads must be at least 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class EpochMetrics:
    """Average loss and classification accuracy for one data pass."""

    loss: float
    accuracy: float


@dataclass(frozen=True)
class TrainingResult:
    """Paths and summary values produced by a completed training run."""

    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    epochs_completed: int
    checkpoint_path: Path
    history_path: Path
    curve_path: Path
    history: list[dict[str, float | int]]


def set_reproducible_seed(seed: int, num_threads: int = 1) -> None:
    """Set Python, NumPy and PyTorch seeds for repeatable CPU experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    """Run one training epoch or evaluation pass and return sample averages."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(signals)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("DataLoader produced no examples")

    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


def _save_training_curve(
    history: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Save train/validation loss and accuracy plots to one PNG."""
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_accuracy = [float(row["train_accuracy"]) for row in history]
    val_accuracy = [float(row["val_accuracy"]) for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, train_loss, marker="o", label="Train")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, train_accuracy, marker="o", label="Train")
    axes[1].plot(epochs, val_accuracy, marker="o", label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.suptitle("I/Q CNN training history")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _atomic_torch_save(payload: dict[str, Any], output_path: Path) -> None:
    """Write a checkpoint via a temporary file to avoid partial outputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)


def train_model(
    config: TrainingConfig | None = None,
    output_dir: str | Path = ".",
    verbose: bool = True,
) -> TrainingResult:
    """Train on matched-SNR data and save the best validation checkpoint.

    The matched-SNR test and low-SNR evaluation loaders are constructed by
    the shared dataset helper but never iterated during training or model
    selection. Phase 5 uses those held-out datasets for final reporting.
    """
    config = config or TrainingConfig()
    config.validate()
    set_reproducible_seed(config.seed, config.num_threads)

    output_root = Path(output_dir)
    checkpoint_path = output_root / "models" / "iq_classifier.pt"
    history_path = output_root / "reports" / "training_history.json"
    curve_path = output_root / "docs" / "screenshots" / "training-curve.png"

    splits = build_dataset_splits(
        n_per_class=config.n_per_class,
        low_snr_n_per_class=config.low_snr_n_per_class,
        signal_length=config.signal_length,
        matched_snr_range=config.matched_snr_range,
        low_snr_range=config.low_snr_range,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    loaders = create_dataloaders(splits, batch_size=config.batch_size, seed=config.seed)

    device = torch.device("cpu")
    model = IQSignalCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if verbose:
        print("I/Q CNN TRAINING")
        print(f"Device: {device}")
        print(f"Trainable parameters: {count_parameters(model):,}")
        print(f"Training examples: {len(splits.train)}")
        print(f"Validation examples: {len(splits.validation)}")
        print(f"Matched-SNR test examples held out: {len(splits.test)}")
        print(f"Low-SNR examples held out: {len(splits.low_snr_eval)}")
        print(f"Matched SNR range: {splits.matched_snr_range} dB")
        print(f"Low-SNR range (reserved for Phase 5): {splits.low_snr_range} dB")

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["validation"], criterion, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
        }
        history.append(row)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{config.epochs:02d} | "
                f"train_loss={train_metrics.loss:.4f} "
                f"train_acc={train_metrics.accuracy:.3f} | "
                f"val_loss={val_metrics.loss:.4f} "
                f"val_acc={val_metrics.accuracy:.3f}"
            )

        # Select the checkpoint by validation loss, not by test performance.
        if val_metrics.loss < best_val_loss - 1e-8:
            best_val_loss = val_metrics.loss
            best_val_accuracy = val_metrics.accuracy
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            if verbose:
                print(f"Early stopping: no validation-loss improvement for {config.patience} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training ended without producing a best model checkpoint")

    model.load_state_dict(best_state)
    checkpoint: dict[str, Any] = {
        "model_state_dict": best_state,
        "model_config": {"num_classes": len(CLASS_NAMES), "input_channels": 2},
        "class_names": list(CLASS_NAMES),
        "training_config": asdict(config),
        "normalizer": {
            "mean": splits.normalizer.mean.tolist(),
            "std": splits.normalizer.std.tolist(),
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "history": history,
    }
    _atomic_torch_save(checkpoint, checkpoint_path)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            {
                "training_config": asdict(config),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _save_training_curve(history, curve_path)

    if verbose:
        print("\nTRAINING COMPLETE")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation accuracy: {best_val_accuracy:.3f}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Training curve: {curve_path}")

    return TrainingResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_accuracy,
        epochs_completed=len(history),
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        curve_path=curve_path,
        history=history,
    )


def parse_args() -> argparse.Namespace:
    """Parse the small set of useful command-line training options."""
    parser = argparse.ArgumentParser(description="Train the I/Q signal CNN on CPU.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    """Run training from the command line."""
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        n_per_class=args.n_per_class,
        low_snr_n_per_class=args.n_per_class,
        seed=args.seed,
    )
    train_model(config=config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
