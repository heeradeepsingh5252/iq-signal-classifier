"""Generate the default Phase 2 datasets and print a concise summary."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from iq_signal_classifier.dataset import build_dataset_splits, create_dataloaders
from iq_signal_classifier.signal_gen import CLASS_NAMES


def main() -> None:
    splits = build_dataset_splits(
        n_per_class=100,
        signal_length=256,
        matched_snr_range=(10.0, 20.0),
        low_snr_range=(-5.0, 5.0),
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=42,
    )

    print("I/Q DATASET SUMMARY")
    print(f"Matched SNR range: {splits.matched_snr_range} dB")
    print(f"Low-SNR evaluation range: {splits.low_snr_range} dB")
    print("Normalization: per-channel mean/std fitted on training data only")
    print()

    for name in ("train", "validation", "test", "low_snr_eval"):
        dataset = getattr(splits, name)
        counts = dataset.labels.bincount(minlength=len(CLASS_NAMES)).tolist()
        class_counts = dict(zip(CLASS_NAMES, counts, strict=True))
        print(
            f"{name:14s}: {len(dataset):4d} samples | "
            f"shape per sample={tuple(dataset.signals.shape[1:])} | "
            f"classes={class_counts}"
        )

    loaders = create_dataloaders(splits, batch_size=32, seed=42)
    batch_signals, batch_labels = next(iter(loaders["train"]))
    print()
    print(f"Training batch signals: {tuple(batch_signals.shape)}")
    print(f"Training batch labels:  {tuple(batch_labels.shape)}")


if __name__ == "__main__":
    main()
