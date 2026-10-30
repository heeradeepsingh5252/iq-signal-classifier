Phase 2 adds a reproducible, stratified dataset pipeline for the synthetic I/Q project.

Files:
- src/iq_signal_classifier/dataset.py: channel normalization, PyTorch Dataset, split builder, and DataLoaders.
- tests/test_dataset.py: fast tests for sizes, shapes, class balance, normalization, reproducibility, and validation.
- scripts/inspect_dataset.py: prints the default dataset sizes and one DataLoader batch shape.

The normalizer is fitted only on the training subset and applied unchanged to validation, matched-SNR test, and low-SNR evaluation data. The low-SNR range must be strictly below the matched-SNR range.
