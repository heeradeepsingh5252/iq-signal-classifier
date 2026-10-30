"""Dataset preparation for the synthetic I/Q modulation classifier.

The matched-SNR pool is split into train, validation, and test subsets.
A second pool is generated independently at a lower SNR and is never used
for fitting the normalizer or selecting model settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from iq_signal_classifier.signal_gen import CLASS_NAMES, generate_dataset

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
DEFAULT_MATCHED_SNR_RANGE: Final[tuple[float, float]] = (10.0, 20.0)
DEFAULT_LOW_SNR_RANGE: Final[tuple[float, float]] = (-5.0, 5.0)


@dataclass(frozen=True)
class ChannelNormalizer:
    """Per-channel standardization fitted only on the training split.

    Means and standard deviations are calculated over examples and time
    positions for each I/Q channel separately. The same values are then
    applied unchanged to validation, test, and low-SNR evaluation data.
    """

    mean: FloatArray
    std: FloatArray

    @classmethod
    def fit(cls, signals: NDArray[np.floating]) -> ChannelNormalizer:
        _validate_signal_array(signals)
        mean = np.asarray(signals.mean(axis=(0, 2), dtype=np.float64), dtype=np.float32)
        std = np.asarray(signals.std(axis=(0, 2), dtype=np.float64), dtype=np.float32)
        std = np.maximum(std, np.float32(1e-6))
        return cls(mean=mean, std=std)

    def transform(self, signals: NDArray[np.floating]) -> FloatArray:
        """Standardize I and Q with training-only channel statistics."""
        _validate_signal_array(signals)
        normalized = (np.asarray(signals, dtype=np.float32) - self.mean[None, :, None])
        normalized = normalized / self.std[None, :, None]
        return np.ascontiguousarray(normalized, dtype=np.float32)


class IQSignalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A small PyTorch dataset returning one I/Q tensor and one class label."""

    def __init__(self, signals: NDArray[np.floating], labels: NDArray[np.integer]) -> None:
        _validate_signal_array(signals)
        label_array = np.asarray(labels)
        if label_array.ndim != 1 or label_array.shape[0] != signals.shape[0]:
            raise ValueError("labels must have shape (N,) and match signals")
        if not np.issubdtype(label_array.dtype, np.integer):
            raise ValueError("labels must contain integers")
        if not np.all(np.isfinite(signals)):
            raise ValueError("signals must contain only finite values")

        self.signals = torch.from_numpy(np.ascontiguousarray(signals, dtype=np.float32))
        self.labels = torch.from_numpy(np.ascontiguousarray(label_array, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signals[index], self.labels[index]


@dataclass(frozen=True)
class DatasetSplits:
    """Four normalized datasets and the training-only normalization values."""

    train: IQSignalDataset
    validation: IQSignalDataset
    test: IQSignalDataset
    low_snr_eval: IQSignalDataset
    normalizer: ChannelNormalizer
    matched_snr_range: tuple[float, float]
    low_snr_range: tuple[float, float]


def _validate_signal_array(signals: NDArray[np.floating]) -> None:
    values = np.asarray(signals)
    if values.ndim != 3 or values.shape[1] != 2 or values.shape[2] == 0:
        raise ValueError("signals must have shape (N, 2, signal_length)")
    if values.shape[0] == 0:
        raise ValueError("signals must contain at least one example")
    if not np.all(np.isfinite(values)):
        raise ValueError("signals must contain only finite values")


def _validate_snr_range(name: str, snr_range: tuple[float, float]) -> tuple[float, float]:
    if len(snr_range) != 2:
        raise ValueError(f"{name} must be a pair (minimum_snr_db, maximum_snr_db)")
    low, high = map(float, snr_range)
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError(f"{name} must contain finite values with minimum <= maximum")
    return low, high


def build_dataset_splits(
    n_per_class: int = 100,
    signal_length: int = 256,
    matched_snr_range: tuple[float, float] = DEFAULT_MATCHED_SNR_RANGE,
    low_snr_range: tuple[float, float] = DEFAULT_LOW_SNR_RANGE,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    low_snr_n_per_class: int | None = None,
    seed: int = 42,
) -> DatasetSplits:
    """Build train/validation/test data and an independent low-SNR test set.

    ``n_per_class`` examples per class are generated for the matched-SNR
    pool. That pool is stratified into train, validation, and test subsets.
    A separate low-SNR pool is generated after the split and is never used
    to fit the channel normalizer.
    """
    if not isinstance(n_per_class, int) or n_per_class < 10:
        raise ValueError("n_per_class must be an integer of at least 10 for stratified splits")
    if not isinstance(signal_length, int) or signal_length <= 0:
        raise ValueError("signal_length must be a positive integer")
    if low_snr_n_per_class is None:
        low_snr_n_per_class = n_per_class
    if not isinstance(low_snr_n_per_class, int) or low_snr_n_per_class <= 0:
        raise ValueError("low_snr_n_per_class must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    matched_range = _validate_snr_range("matched_snr_range", matched_snr_range)
    low_range = _validate_snr_range("low_snr_range", low_snr_range)
    if low_range[1] >= matched_range[0]:
        raise ValueError("low_snr_range must be strictly below matched_snr_range")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1")

    matched_rng_seed, low_rng_seed = np.random.SeedSequence(seed).spawn(2)
    matched_rng = np.random.default_rng(matched_rng_seed)
    low_rng = np.random.default_rng(low_rng_seed)

    matched_x, matched_y = generate_dataset(
        n_per_class=n_per_class,
        snr_db_range=matched_range,
        signal_length=signal_length,
        rng=matched_rng,
    )

    # Convert proportions to explicit sample counts. This avoids floating-point
    # rounding making a nominal 15% validation split contain one extra sample.
    test_count = round(len(matched_y) * test_fraction)
    validation_count = round(len(matched_y) * validation_fraction)
    minimum_split_count = len(CLASS_NAMES)
    if test_count < minimum_split_count or validation_count < minimum_split_count:
        raise ValueError(
            "The requested validation/test fractions produce too few examples; "
            "increase n_per_class or increase the split fractions"
        )

    try:
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            matched_x,
            matched_y,
            test_size=test_count,
            random_state=seed,
            stratify=matched_y,
        )
        x_train, x_validation, y_train, y_validation = train_test_split(
            x_train_val,
            y_train_val,
            test_size=validation_count,
            random_state=seed + 1,
            stratify=y_train_val,
        )
    except ValueError as exc:
        raise ValueError(
            "Not enough examples for the requested stratified split; "
            "increase n_per_class or reduce validation_fraction/test_fraction"
        ) from exc

    # Fit only on training signals to prevent validation/test information leak.
    normalizer = ChannelNormalizer.fit(x_train)
    train_dataset = IQSignalDataset(normalizer.transform(x_train), y_train)
    validation_dataset = IQSignalDataset(normalizer.transform(x_validation), y_validation)
    test_dataset = IQSignalDataset(normalizer.transform(x_test), y_test)

    low_x, low_y = generate_dataset(
        n_per_class=low_snr_n_per_class,
        snr_db_range=low_range,
        signal_length=signal_length,
        rng=low_rng,
    )
    low_snr_dataset = IQSignalDataset(normalizer.transform(low_x), low_y)

    return DatasetSplits(
        train=train_dataset,
        validation=validation_dataset,
        test=test_dataset,
        low_snr_eval=low_snr_dataset,
        normalizer=normalizer,
        matched_snr_range=matched_range,
        low_snr_range=low_range,
    )


def create_dataloaders(
    splits: DatasetSplits,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create CPU-friendly DataLoaders; shuffle training data only."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    return {
        "train": DataLoader(
            splits.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=train_generator,
        ),
        "validation": DataLoader(
            splits.validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            splits.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "low_snr_eval": DataLoader(
            splits.low_snr_eval,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }
