# Phase 4 files: CPU-friendly CNN training

- `src/iq_signal_classifier/train.py`: reproducible CPU training, per-epoch train/validation loss and accuracy, validation-loss checkpoint selection, early stopping, checkpoint metadata (including training-only normalization parameters), JSON history, and a training-curve PNG.
- `tests/test_train.py`: fast smoke tests for an optimization epoch, a one-epoch end-to-end training run, output artifacts, and invalid configuration.

Default run: 20 epochs maximum, batch size 32, 100 examples per class in the matched-SNR pool, AdamW learning rate 0.001, weight decay 0.0001, patience 5, CPU with one PyTorch thread. The training routine only iterates the train and validation loaders; matched-SNR test and low-SNR evaluation sets are kept for Phase 5.

Outputs:
- `models/iq_classifier.pt`
- `reports/training_history.json`
- `docs/screenshots/training-curve.png`

The best checkpoint is selected by the lowest validation loss. No test or low-SNR metric is computed in this phase.
