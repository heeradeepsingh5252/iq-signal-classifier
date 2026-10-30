# Phase 3 files

- `src/iq_signal_classifier/model.py`: small three-block 1D CNN for 2-channel I/Q input. Returns logits for use with `torch.nn.CrossEntropyLoss`; `probabilities()` applies softmax for inference.
- `tests/test_model.py`: checks output shape, parameter budget, probability normalization, and invalid inputs.

Expected trainable parameters with the default four classes: 11,364.
