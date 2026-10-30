import numpy as np
import pytest
import torch

from iq_signal_classifier.dataset import build_dataset_splits, create_dataloaders
from iq_signal_classifier.signal_gen import CLASS_NAMES


def _small_splits():
    return build_dataset_splits(
        n_per_class=12,
        low_snr_n_per_class=12,
        signal_length=128,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=17,
    )


def test_split_sizes_shapes_and_types() -> None:
    splits = _small_splits()

    assert len(splits.train) == 24
    assert len(splits.validation) == 12
    assert len(splits.test) == 12
    assert len(splits.low_snr_eval) == 48

    for dataset in (splits.train, splits.validation, splits.test, splits.low_snr_eval):
        assert dataset.signals.shape == (len(dataset), 2, 128)
        assert dataset.signals.dtype == torch.float32
        assert dataset.labels.shape == (len(dataset),)
        assert dataset.labels.dtype == torch.int64


def test_each_split_preserves_all_classes() -> None:
    splits = _small_splits()
    expected_counts = {
        "train": 6,
        "validation": 3,
        "test": 3,
        "low_snr_eval": 12,
    }

    for split_name, expected_count in expected_counts.items():
        dataset = getattr(splits, split_name)
        counts = torch.bincount(dataset.labels, minlength=len(CLASS_NAMES))
        assert counts.tolist() == [expected_count] * len(CLASS_NAMES)


def test_normalization_is_fitted_on_train_and_applied_to_all_splits() -> None:
    splits = _small_splits()
    train_signals = splits.train.signals.numpy()

    # The training channels should be approximately zero-mean/unit-std.
    np.testing.assert_allclose(train_signals.mean(axis=(0, 2)), [0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(train_signals.std(axis=(0, 2)), [1.0, 1.0], atol=1e-5)

    assert splits.normalizer.mean.shape == (2,)
    assert splits.normalizer.std.shape == (2,)
    assert np.all(splits.normalizer.std > 0)


def test_dataloaders_return_expected_batches_and_only_four_splits() -> None:
    splits = _small_splits()
    loaders = create_dataloaders(splits, batch_size=8, seed=23)

    assert set(loaders) == {"train", "validation", "test", "low_snr_eval"}
    signals, labels = next(iter(loaders["train"]))
    assert signals.shape == (8, 2, 128)
    assert labels.shape == (8,)
    assert signals.dtype == torch.float32
    assert labels.dtype == torch.int64

    low_signals, low_labels = next(iter(loaders["low_snr_eval"]))
    assert low_signals.shape == (8, 2, 128)
    assert low_labels.shape == (8,)


def test_build_is_reproducible_for_same_seed() -> None:
    first = _small_splits()
    second = _small_splits()

    torch.testing.assert_close(first.train.signals, second.train.signals)
    torch.testing.assert_close(first.low_snr_eval.signals, second.low_snr_eval.signals)
    torch.testing.assert_close(first.train.labels, second.train.labels)


def test_invalid_ranges_and_sizes_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        build_dataset_splits(n_per_class=5)

    with pytest.raises(ValueError, match="strictly below"):
        build_dataset_splits(low_snr_range=(5.0, 12.0))

    with pytest.raises(ValueError, match="less than 1"):
        build_dataset_splits(validation_fraction=0.6, test_fraction=0.4)

    with pytest.raises(ValueError, match="batch_size"):
        create_dataloaders(_small_splits(), batch_size=0)
