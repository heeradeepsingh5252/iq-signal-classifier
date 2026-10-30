"""Evaluate a saved I/Q CNN on matched- and low-SNR held-out datasets.

This module deliberately performs inference only. It never trains the model,
selects a checkpoint, or fits normalization statistics on evaluation data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from iq_signal_classifier.dataset import build_dataset_splits, create_dataloaders
from iq_signal_classifier.model import IQSignalCNN
from iq_signal_classifier.signal_gen import CLASS_NAMES


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall, F1 and support for one signal class."""

    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class EvaluationMetrics:
    """Metrics computed from one untouched evaluation loader."""

    accuracy: float
    sample_count: int
    confusion: list[list[int]]
    per_class: dict[str, ClassMetrics]


@dataclass(frozen=True)
class EvaluationResult:
    """Complete matched-SNR and low-SNR evaluation outputs."""

    matched_snr_range: tuple[float, float]
    low_snr_range: tuple[float, float]
    matched: EvaluationMetrics
    low_snr: EvaluationMetrics
    accuracy_difference_percentage_points: float
    results_path: Path
    summary_path: Path
    chart_path: Path


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    class_names: tuple[str, ...] = CLASS_NAMES,
    device: torch.device | None = None,
) -> EvaluationMetrics:
    """Run inference and calculate accuracy, confusion and per-class metrics."""
    if not class_names:
        raise ValueError("class_names must not be empty")
    target_device = device or torch.device("cpu")
    model = model.to(target_device)
    model.eval()

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for signals, labels in loader:
            logits = model(signals.to(target_device))
            if logits.ndim != 2 or logits.shape[1] != len(class_names):
                raise ValueError(
                    f"model must return logits with shape (batch, {len(class_names)})"
                )
            predictions = logits.argmax(dim=1)
            all_targets.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))
            all_predictions.append(predictions.detach().cpu().numpy().astype(np.int64, copy=False))

    if not all_targets:
        raise ValueError("evaluation DataLoader produced no batches")

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    label_ids = np.arange(len(class_names), dtype=np.int64)
    if np.any(y_true < 0) or np.any(y_true >= len(class_names)):
        raise ValueError("evaluation labels contain an unknown class index")

    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    per_class = {
        name: ClassMetrics(
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1_score=float(f1_score[index]),
            support=int(support[index]),
        )
        for index, name in enumerate(class_names)
    }
    return EvaluationMetrics(
        accuracy=float(np.mean(y_true == y_pred)),
        sample_count=int(y_true.size),
        confusion=matrix.astype(int).tolist(),
        per_class=per_class,
    )


def _load_checkpoint(checkpoint_path: Path) -> tuple[IQSignalCNN, dict[str, Any]]:
    """Load and validate the Phase 4 checkpoint on CPU."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # Older PyTorch releases without weights_only.
        raise RuntimeError(
            "This evaluation script requires a PyTorch version supporting "
            "torch.load(weights_only=True). Upgrade PyTorch and retry."
        ) from exc

    required = {"model_state_dict", "model_config", "class_names", "training_config", "normalizer"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    if list(checkpoint["class_names"]) != list(CLASS_NAMES):
        raise ValueError("Checkpoint class order does not match the current signal generator")

    model_config = checkpoint["model_config"]
    model = IQSignalCNN(
        num_classes=int(model_config["num_classes"]),
        input_channels=int(model_config["input_channels"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _rebuild_splits(checkpoint: dict[str, Any]):
    """Recreate the exact deterministic splits used for Phase 4 training."""
    config = checkpoint["training_config"]
    required = {
        "n_per_class",
        "low_snr_n_per_class",
        "signal_length",
        "matched_snr_range",
        "low_snr_range",
        "validation_fraction",
        "test_fraction",
        "seed",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Checkpoint training_config is missing: {sorted(missing)}")

    splits = build_dataset_splits(
        n_per_class=int(config["n_per_class"]),
        low_snr_n_per_class=int(config["low_snr_n_per_class"]),
        signal_length=int(config["signal_length"]),
        matched_snr_range=tuple(config["matched_snr_range"]),
        low_snr_range=tuple(config["low_snr_range"]),
        validation_fraction=float(config["validation_fraction"]),
        test_fraction=float(config["test_fraction"]),
        seed=int(config["seed"]),
    )

    saved_normalizer = checkpoint["normalizer"]
    saved_mean = np.asarray(saved_normalizer["mean"], dtype=np.float32)
    saved_std = np.asarray(saved_normalizer["std"], dtype=np.float32)
    if not np.allclose(saved_mean, splits.normalizer.mean, rtol=1e-6, atol=1e-7):
        raise ValueError("Recreated training mean differs from the saved checkpoint")
    if not np.allclose(saved_std, splits.normalizer.std, rtol=1e-6, atol=1e-7):
        raise ValueError("Recreated training standard deviation differs from the checkpoint")
    return splits


def _metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, Any]:
    return {
        "accuracy": metrics.accuracy,
        "sample_count": metrics.sample_count,
        "confusion_matrix": metrics.confusion,
        "per_class": {name: asdict(value) for name, value in metrics.per_class.items()},
    }


def _save_comparison_chart(
    matched: EvaluationMetrics,
    low_snr: EvaluationMetrics,
    class_names: tuple[str, ...],
    matched_range: tuple[float, float],
    low_range: tuple[float, float],
    output_path: Path,
) -> None:
    """Save accuracy comparison and both confusion matrices in one PNG."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    accuracy_values = [matched.accuracy, low_snr.accuracy]
    accuracy_bars = axes[0].bar(
        [f"Matched\n{matched_range[0]:g} to {matched_range[1]:g} dB", f"Low SNR\n{low_range[0]:g} to {low_range[1]:g} dB"],
        accuracy_values,
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Held-out accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(accuracy_bars, accuracy_values, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center")

    for axis, metrics, title in (
        (axes[1], matched, "Matched-SNR confusion"),
        (axes[2], low_snr, "Low-SNR confusion"),
    ):
        matrix = np.asarray(metrics.confusion, dtype=np.int64)
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
        axis.set_yticks(range(len(class_names)), class_names)
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                color = "white" if matrix[row, column] > threshold else "black"
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("I/Q CNN robustness evaluation")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_summary(
    matched: EvaluationMetrics,
    low_snr: EvaluationMetrics,
    matched_range: tuple[float, float],
    low_range: tuple[float, float],
    output_path: Path,
) -> None:
    """Write an interpretation grounded in the measured aggregate metrics."""
    difference = (matched.accuracy - low_snr.accuracy) * 100.0
    if difference > 0:
        comparison = (
            f"Low-SNR accuracy is {difference:.2f} percentage points lower than "
            "matched-SNR accuracy in this synthetic experiment."
        )
    elif difference < 0:
        comparison = (
            f"Low-SNR accuracy is {abs(difference):.2f} percentage points higher "
            "than matched-SNR accuracy in this synthetic experiment."
        )
    else:
        comparison = "Matched-SNR and low-SNR accuracy are equal in this run."

    lines = [
        "# I/Q CNN robustness evaluation",
        "",
        "## Experimental setup",
        "",
        f"- Matched-SNR test range: {matched_range[0]:g} to {matched_range[1]:g} dB ({matched.sample_count} samples).",
        f"- Independent low-SNR range: {low_range[0]:g} to {low_range[1]:g} dB ({low_snr.sample_count} samples).",
        "- The checkpoint and training-only channel normalization statistics were loaded from the Phase 4 artifact.",
        "- The low-SNR set was generated independently and was not used for training, validation, or checkpoint selection.",
        "",
        "## Measured results",
        "",
        "| Evaluation | Accuracy | Samples |",
        "|---|---:|---:|",
        f"| Matched SNR | {matched.accuracy:.4f} ({matched.accuracy:.1%}) | {matched.sample_count} |",
        f"| Low SNR | {low_snr.accuracy:.4f} ({low_snr.accuracy:.1%}) | {low_snr.sample_count} |",
        "",
        "## Per-class precision and recall",
        "",
        "| Class | Matched precision | Matched recall | Low-SNR precision | Low-SNR recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CLASS_NAMES:
        first = matched.per_class[name]
        second = low_snr.per_class[name]
        lines.append(
            f"| {name} | {first.precision:.4f} | {first.recall:.4f} | "
            f"{second.precision:.4f} | {second.recall:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            comparison,
            "",
            "Class-level differences should be read from the table and confusion matrices above; do not infer a general RF capability from this synthetic experiment.",
            "",
            "**Simulation limitation:** the Phase 1 NOISE class is generated as noise-only data and is not altered by the requested SNR range. Therefore, the low-SNR experiment changes the added-noise condition for BPSK, QPSK, and AM, but does not create an SNR-shifted NOISE class.",
            "",
            "This is a simplified synthetic baseband simulation, not a hardware-validated RF pipeline. Real systems would require captured signals and evaluation across hardware impairments, channel effects, and broader operating conditions.",
            "",
            "## Confusion matrices",
            "",
            "Rows are true labels and columns are predicted labels. Class order: " + ", ".join(CLASS_NAMES) + ".",
            "",
            "### Matched-SNR test",
            "",
            "```text",
            json.dumps(matched.confusion),
            "```",
            "",
            "### Low-SNR evaluation",
            "",
            "```text",
            json.dumps(low_snr.confusion),
            "```",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_checkpoint(
    checkpoint_path: str | Path = "models/iq_classifier.pt",
    output_dir: str | Path = ".",
    batch_size: int = 32,
) -> EvaluationResult:
    """Evaluate the Phase 4 checkpoint and save real metrics and plots."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    checkpoint_file = Path(checkpoint_path)
    output_root = Path(output_dir)
    model, checkpoint = _load_checkpoint(checkpoint_file)
    splits = _rebuild_splits(checkpoint)
    loaders = create_dataloaders(splits, batch_size=batch_size, seed=0)

    # Evaluate only the two reserved loaders. No training or validation data is used.
    matched = evaluate_loader(model, loaders["test"])
    low_snr = evaluate_loader(model, loaders["low_snr_eval"])
    difference = (matched.accuracy - low_snr.accuracy) * 100.0

    results_path = output_root / "reports" / "robustness_results.json"
    summary_path = output_root / "reports" / "robustness_summary.md"
    chart_path = output_root / "docs" / "screenshots" / "robustness-comparison.png"
    matched_range = splits.matched_snr_range
    low_range = splits.low_snr_range

    payload = {
        "checkpoint": str(checkpoint_file),
        "best_epoch": int(checkpoint["best_epoch"]),
        "class_names": list(CLASS_NAMES),
        "matched_snr_range_db": list(matched_range),
        "low_snr_range_db": list(low_range),
        "matched_snr_test": _metrics_to_dict(matched),
        "low_snr_evaluation": _metrics_to_dict(low_snr),
        "accuracy_difference_percentage_points_matched_minus_low": difference,
        "noise_class_snr_note": (
            "The Phase 1 NOISE class is generated as noise-only data independently of the SNR range; "
            "the low-SNR range changes added noise for BPSK, QPSK, and AM only."
        ),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_summary(matched, low_snr, matched_range, low_range, summary_path)
    _save_comparison_chart(matched, low_snr, CLASS_NAMES, matched_range, low_range, chart_path)

    return EvaluationResult(
        matched_snr_range=matched_range,
        low_snr_range=low_range,
        matched=matched,
        low_snr=low_snr,
        accuracy_difference_percentage_points=difference,
        results_path=results_path,
        summary_path=summary_path,
        chart_path=chart_path,
    )


def parse_args() -> argparse.Namespace:
    """Parse evaluation command-line options."""
    parser = argparse.ArgumentParser(description="Evaluate the trained I/Q CNN robustness.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/iq_classifier.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Run matched-SNR and low-SNR evaluation from the command line."""
    args = parse_args()
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print("I/Q CNN ROBUSTNESS EVALUATION")
    print(f"Matched-SNR range: {result.matched_snr_range} dB")
    print(f"Low-SNR range: {result.low_snr_range} dB")
    print("\nAccuracy comparison")
    print(f"Matched-SNR test: {result.matched.accuracy:.4f} ({result.matched.accuracy:.1%})")
    print(f"Low-SNR evaluation: {result.low_snr.accuracy:.4f} ({result.low_snr.accuracy:.1%})")
    print(
        "Matched minus low-SNR difference: "
        f"{result.accuracy_difference_percentage_points:+.2f} percentage points"
    )
    print("\nPer-class metrics")
    print("Class    Matched P/R       Low-SNR P/R")
    for name in CLASS_NAMES:
        first = result.matched.per_class[name]
        second = result.low_snr.per_class[name]
        print(
            f"{name:<7}  {first.precision:.3f}/{first.recall:.3f}       "
            f"{second.precision:.3f}/{second.recall:.3f}"
        )
    print("\nConfusion matrix order:", ", ".join(CLASS_NAMES))
    print("Matched-SNR confusion matrix:")
    print(np.asarray(result.matched.confusion))
    print("Low-SNR confusion matrix:")
    print(np.asarray(result.low_snr.confusion))
    print(f"\nJSON results: {result.results_path}")
    print(f"Written interpretation: {result.summary_path}")
    print(f"Comparison chart: {result.chart_path}")


if __name__ == "__main__":
    main()
