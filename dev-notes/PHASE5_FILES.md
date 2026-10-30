# Phase 5 files: held-out robustness evaluation

- `src/iq_signal_classifier/evaluate.py`: loads the Phase 4 checkpoint safely on CPU, rebuilds the deterministic Phase 2 datasets from the checkpoint's saved training configuration, verifies the saved training-only normalizer, evaluates only the matched-SNR test and independent low-SNR evaluation loaders, and reports accuracy, confusion matrices, per-class precision/recall/F1, and the accuracy difference.
- `tests/test_evaluate.py`: fast metric, output-validation, checkpoint, normalizer-integrity, and artifact tests using small synthetic data.

Run from the repository root:

```bash
PYTHONPATH=src python -m iq_signal_classifier.evaluate
```

Artifacts generated from the user's actual checkpoint and local run:

- `reports/robustness_results.json`
- `reports/robustness_summary.md`
- `docs/screenshots/robustness-comparison.png`

The Phase 1 NOISE class is independent of SNR and remains noise-only; only BPSK, QPSK, and AM receive SNR-controlled additive Gaussian noise. The report explicitly documents this simulation limitation.
