"""Generate small synthetic complex-baseband I/Q signals.

This module is a teaching simulation, not a hardware-validated RF simulator.
Each sample is represented as two channels: I (in-phase) and Q (quadrature).
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]

CLASS_NAMES: Final[tuple[str, ...]] = ("BPSK", "QPSK", "AM", "NOISE")
CLASS_TO_LABEL: Final[dict[str, int]] = {name: index for index, name in enumerate(CLASS_NAMES)}
LABEL_TO_CLASS: Final[dict[int, str]] = {index: name for name, index in CLASS_TO_LABEL.items()}
DEFAULT_SAMPLES_PER_SYMBOL: Final[int] = 8


def _validate_signal_length(signal_length: int) -> None:
    if not isinstance(signal_length, int) or signal_length <= 0:
        raise ValueError("signal_length must be a positive integer")


def _symbols_to_samples(symbols: NDArray[np.float64], signal_length: int) -> NDArray[np.float64]:
    """Use a simple rectangular pulse: repeat each symbol for 8 samples."""
    samples = np.repeat(symbols, DEFAULT_SAMPLES_PER_SYMBOL)
    if samples.size < signal_length:
        samples = np.pad(samples, (0, signal_length - samples.size), mode="edge")
    return samples[:signal_length]


def generate_signal(
    modulation: str,
    signal_length: int,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Generate one clean I/Q signal with shape ``(2, signal_length)``.

    BPSK uses real symbols -1/+1. QPSK uses four equally spaced points
    (±1 ± j)/sqrt(2). AM is represented by a real baseband envelope
    1 + 0.5*cos(2*pi*f_m*t), with Q=0. The NOISE class is complex white
    Gaussian noise with unit average I/Q power; it has no underlying signal.
    """
    _validate_signal_length(signal_length)
    name = modulation.upper()
    if name not in CLASS_TO_LABEL:
        raise ValueError(f"Unsupported modulation {modulation!r}; choose from {CLASS_NAMES}")

    generator = rng if rng is not None else np.random.default_rng()

    if name == "BPSK":
        # Binary phase-shift keying: each bit selects phase 0 or pi,
        # represented by a real baseband symbol of +1 or -1.
        count = int(np.ceil(signal_length / DEFAULT_SAMPLES_PER_SYMBOL))
        symbols = generator.choice(np.array([-1.0, 1.0]), size=count)
        in_phase = _symbols_to_samples(symbols, signal_length)
        quadrature = np.zeros(signal_length, dtype=np.float64)

    elif name == "QPSK":
        # Quadrature phase-shift keying: two bits choose one of four
        # phases. The I and Q components are each +/-1/sqrt(2).
        count = int(np.ceil(signal_length / DEFAULT_SAMPLES_PER_SYMBOL))
        i_symbols = generator.choice(np.array([-1.0, 1.0]), size=count) / np.sqrt(2.0)
        q_symbols = generator.choice(np.array([-1.0, 1.0]), size=count) / np.sqrt(2.0)
        in_phase = _symbols_to_samples(i_symbols, signal_length)
        quadrature = _symbols_to_samples(q_symbols, signal_length)

    elif name == "AM":
        # Amplitude modulation varies the carrier envelope with the message.
        # After ideal down-conversion, this simple real envelope is on I and
        # the quadrature channel is zero. This deliberately omits RF hardware
        # effects, carrier offsets, filtering, and channel impairments.
        time = np.arange(signal_length, dtype=np.float64) / signal_length
        message_cycles = float(generator.uniform(2.0, 5.0))
        phase = float(generator.uniform(0.0, 2.0 * np.pi))
        in_phase = 1.0 + 0.5 * np.cos(2.0 * np.pi * message_cycles * time + phase)
        quadrature = np.zeros(signal_length, dtype=np.float64)

    else:  # NOISE
        # No signal is present: independent I and Q Gaussian components.
        # Each component has variance 1/2, giving total complex power 1.
        in_phase = generator.normal(0.0, np.sqrt(0.5), size=signal_length)
        quadrature = generator.normal(0.0, np.sqrt(0.5), size=signal_length)

    return np.stack((in_phase, quadrature)).astype(np.float32, copy=False)


def add_noise(
    iq: NDArray[np.floating],
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Add complex Gaussian noise to achieve the requested average I/Q SNR.

    ``iq`` must have shape ``(2, samples)`` or ``(batch, 2, samples)``.
    Signal power is mean(I**2 + Q**2). Noise power is signal_power divided
    by 10**(snr_db/10). The target noise power is split equally between I
    and Q, so each real component has standard deviation sqrt(noise_power/2).
    """
    values = np.asarray(iq, dtype=np.float32)
    if values.ndim not in (2, 3) or values.shape[-2] != 2 or values.shape[-1] == 0:
        raise ValueError("iq must have shape (2, samples) or (batch, 2, samples)")
    if not np.all(np.isfinite(values)):
        raise ValueError("iq must contain only finite values")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")

    generator = rng if rng is not None else np.random.default_rng()
    signal_power = np.mean(np.sum(values.astype(np.float64) ** 2, axis=-2), axis=-1)
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    component_std = np.sqrt(noise_power / 2.0)
    noise_shape = values.shape
    if values.ndim == 3:
        component_std = component_std[:, np.newaxis, np.newaxis]
    else:
        component_std = np.asarray(component_std).reshape(1, 1)

    noise = generator.normal(0.0, component_std, size=noise_shape)
    return (values.astype(np.float64) + noise).astype(np.float32)


def generate_dataset(
    n_per_class: int,
    snr_db_range: tuple[float, float],
    signal_length: int,
    rng: np.random.Generator | None = None,
) -> tuple[FloatArray, IntArray]:
    """Generate a balanced dataset with shape ``(N, 2, signal_length)``.

    For BPSK, QPSK, and AM, an SNR is sampled uniformly from ``snr_db_range``
    for each generated example. The NOISE class contains noise-only examples
    with unit average I/Q power, independent of the SNR range.
    Labels follow ``CLASS_NAMES``: BPSK=0, QPSK=1, AM=2, NOISE=3.
    """
    _validate_signal_length(signal_length)
    if not isinstance(n_per_class, int) or n_per_class <= 0:
        raise ValueError("n_per_class must be a positive integer")
    if len(snr_db_range) != 2:
        raise ValueError("snr_db_range must contain (minimum_snr_db, maximum_snr_db)")
    low_snr, high_snr = map(float, snr_db_range)
    if not np.isfinite(low_snr) or not np.isfinite(high_snr) or low_snr > high_snr:
        raise ValueError("snr_db_range must contain finite values with minimum <= maximum")

    generator = rng if rng is not None else np.random.default_rng()
    examples: list[FloatArray] = []
    labels: list[int] = []

    for class_name in CLASS_NAMES:
        for _ in range(n_per_class):
            clean_or_noise = generate_signal(class_name, signal_length, generator)
            if class_name != "NOISE":
                sample_snr = float(generator.uniform(low_snr, high_snr))
                sample = add_noise(clean_or_noise, sample_snr, generator)
            else:
                sample = clean_or_noise
            examples.append(sample)
            labels.append(CLASS_TO_LABEL[class_name])

    order = generator.permutation(len(examples))
    x = np.stack(examples, axis=0)[order].astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)[order]
    return x, y
