# I/Q CNN robustness evaluation

## Experimental setup

- Matched-SNR test range: 10 to 20 dB (60 samples).
- Independent low-SNR range: -5 to 5 dB (400 samples).
- The checkpoint and training-only channel normalization statistics were loaded from the Phase 4 artifact.
- The low-SNR set was generated independently and was not used for training, validation, or checkpoint selection.

## Measured results

| Evaluation | Accuracy | Samples |
|---|---:|---:|
| Matched SNR | 1.0000 (100.0%) | 60 |
| Low SNR | 0.4475 (44.8%) | 400 |

## Per-class precision and recall

| Class | Matched precision | Matched recall | Low-SNR precision | Low-SNR recall |
|---|---:|---:|---:|---:|
| BPSK | 1.0000 | 1.0000 | 1.0000 | 0.2400 |
| QPSK | 1.0000 | 1.0000 | 1.0000 | 0.2700 |
| AM | 1.0000 | 1.0000 | 1.0000 | 0.2800 |
| NOISE | 1.0000 | 1.0000 | 0.3115 | 1.0000 |

## Interpretation

Low-SNR accuracy is 55.25 percentage points lower than matched-SNR accuracy in this synthetic experiment.

Class-level differences should be read from the table and confusion matrices above; do not infer a general RF capability from this synthetic experiment.

**Simulation limitation:** the Phase 1 NOISE class is generated as noise-only data and is not altered by the requested SNR range. Therefore, the low-SNR experiment changes the added-noise condition for BPSK, QPSK, and AM, but does not create an SNR-shifted NOISE class.

This is a simplified synthetic baseband simulation, not a hardware-validated RF pipeline. Real systems would require captured signals and evaluation across hardware impairments, channel effects, and broader operating conditions.

## Confusion matrices

Rows are true labels and columns are predicted labels. Class order: BPSK, QPSK, AM, NOISE.

### Matched-SNR test

```text
[[15, 0, 0, 0], [0, 15, 0, 0], [0, 0, 15, 0], [0, 0, 0, 15]]
```

### Low-SNR evaluation

```text
[[24, 0, 0, 76], [0, 27, 0, 73], [0, 0, 28, 72], [0, 0, 0, 100]]
```
