"""Fast shape, parameter-count, and input-validation tests for the CNN."""

import pytest
import torch

from iq_signal_classifier.model import IQSignalCNN, count_parameters


def test_forward_output_shape() -> None:
    model = IQSignalCNN(num_classes=4)
    batch = torch.randn(8, 2, 256)
    logits = model(batch)
    assert logits.shape == (8, 4)


def test_model_has_fewer_than_100k_parameters() -> None:
    model = IQSignalCNN()
    assert count_parameters(model) < 100_000


def test_probabilities_sum_to_one() -> None:
    model = IQSignalCNN()
    logits = model(torch.randn(3, 2, 128))
    probabilities = model.probabilities(logits)
    assert probabilities.shape == (3, 4)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-6)


def test_rejects_wrong_channel_count() -> None:
    model = IQSignalCNN()
    with pytest.raises(ValueError, match="I and Q"):
        model(torch.randn(4, 1, 128))


def test_rejects_wrong_input_rank() -> None:
    model = IQSignalCNN()
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 256))


def test_rejects_signal_too_short() -> None:
    model = IQSignalCNN()
    with pytest.raises(ValueError, match="at least 4"):
        model(torch.randn(2, 2, 3))


def test_num_classes_validation() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        IQSignalCNN(num_classes=1)
