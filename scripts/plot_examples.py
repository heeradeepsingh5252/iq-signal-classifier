"""Save one example I/Q time-series plot for each synthetic class."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from iq_signal_classifier.signal_gen import CLASS_NAMES, generate_signal


def main() -> None:
    output_dir = ROOT / "docs" / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    time = np.arange(256)

    fig, axes = plt.subplots(len(CLASS_NAMES), 1, figsize=(11, 9), sharex=True)
    for axis, class_name in zip(axes, CLASS_NAMES, strict=True):
        iq = generate_signal(class_name, signal_length=256, rng=rng)
        axis.plot(time, iq[0], label="I (in-phase)", linewidth=1.0)
        axis.plot(time, iq[1], label="Q (quadrature)", linewidth=1.0, alpha=0.8)
        axis.set_title(class_name)
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=2)

    axes[-1].set_xlabel("Sample index")
    fig.suptitle("Synthetic baseband I/Q examples (no added channel noise)", fontsize=13)
    fig.tight_layout()
    destination = output_dir / "signal-examples.png"
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved signal examples to: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
