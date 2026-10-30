"""Small synthetic I/Q signal classification project."""

from .signal_gen import (
    CLASS_NAMES,
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    add_noise,
    generate_dataset,
    generate_signal,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_LABEL",
    "LABEL_TO_CLASS",
    "add_noise",
    "generate_dataset",
    "generate_signal",
]
