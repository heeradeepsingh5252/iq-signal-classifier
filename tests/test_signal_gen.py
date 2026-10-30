import numpy as np
import pytest

from iq_signal_classifier.signal_gen import (
    CLASS_NAMES,
    CLASS_TO_LABEL,
    add_noise,
    generate_dataset,
    generate_signal,
)


def test_each_signal_has_two_iq_channels_and_expected_length() -> None:
    rng = np.random.default_rng(7)
    for name in CLASS_NAMES:
        signal = generate_signal(name, signal_length=128, rng=rng)
        assert signal.shape == (2, 128)
        assert signal.dtype == np.float32
        assert np.isfinite(signal).all()


def test_dataset_shape_and_balanced_label_counts() -> None:
    x, y = generate_dataset(
        n_per_class=5,
        snr_db_range=(5.0, 15.0),
        signal_length=128,
        rng=np.random.default_rng(11),
    )
    assert x.shape == (20, 2, 128)
    assert y.shape == (20,)
    assert x.dtype == np.float32
    assert y.dtype == np.int64
    for label in CLASS_TO_LABEL.values():
        assert int(np.sum(y == label)) == 5


def test_higher_snr_adds_less_noise_than_lower_snr() -> None:
    clean = np.ones((2, 4096), dtype=np.float32)
    high_snr = add_noise(clean, 20.0, rng=np.random.default_rng(123))
    low_snr = add_noise(clean, 0.0, rng=np.random.default_rng(123))
    high_noise_power = float(np.mean((high_snr - clean) ** 2))
    low_noise_power = float(np.mean((low_snr - clean) ** 2))
    assert low_noise_power > high_noise_power * 50.0


def test_invalid_modulation_and_inputs_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="Unsupported modulation"):
        generate_signal("FM", 128)
    with pytest.raises(ValueError, match="positive integer"):
        generate_signal("BPSK", 0)
    with pytest.raises(ValueError, match="snr_db_range"):
        generate_dataset(2, (10.0, -2.0), 128)
    with pytest.raises(ValueError, match="positive integer"):
        generate_dataset(0, (0.0, 10.0), 128)


def test_add_noise_accepts_a_batch() -> None:
    clean = np.ones((3, 2, 64), dtype=np.float32)
    noisy = add_noise(clean, 10.0, rng=np.random.default_rng(9))
    assert noisy.shape == clean.shape
    assert np.isfinite(noisy).all()
