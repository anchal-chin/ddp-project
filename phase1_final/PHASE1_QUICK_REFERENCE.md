# Phase 1 — Quick Reference

*One-page cheat sheet. All numbers verified against `results/metrics/experiment_log.csv`, `final_test_results.json`, and loaded model checkpoints.*

## Dataset
| | |
|---|---|
| Train | 13,932 images |
| Validation | 2,452 images |
| Test | 8,827 images |
| Resolution | 64×64, grayscale (1 channel) |
| Classes | 11 (integer-encoded, 0–10) |
| Imbalance ratio | 5.64× (class 6 largest 3,464; class 2 smallest 614) |
| Dataset name | **Not independently verified** — do not assert a specific name as confirmed fact |

## Baseline Model A
| | |
|---|---|
| Architecture | 3 conv blocks (16→32→64), Flatten, Dense(64), Dropout(0.5), Dense(11, softmax) |
| Parameters | **286,219** |
| Optimizer | Adam, lr = 1e-3 |
| Loss | `sparse_categorical_crossentropy` |
| Batch size | 64 |
| Early stopping | patience 5, monitor `val_loss` |

## Results — all 6 experiments (validation set)

| Experiment | Accuracy | Macro F1 |
|---|---|---|
| **Baseline Model A** | **0.8907** | **0.8053** ← best |
| Class Weighting | 0.8732 | 0.7833 |
| Augmentation | 0.8426 | 0.7763 |
| Model B (602,347 params) | 0.8858 | 0.7918 |
| Model C (250,635 params) | 0.8825 | 0.7796 |
| Focal Loss (γ=2.0) | 0.8760 | 0.7925 |

**Every alternative reduced Macro F1 relative to the baseline.**

## Final Test Result (Model A, evaluated once, after selection)
| | |
|---|---|
| Test Accuracy | **0.7339** |
| Test Macro Precision | 0.7007 |
| Test Macro Recall | 0.6857 |
| **Test Macro F1** | **0.6840** |
| Validation→Test gap | −0.1213 Macro F1 (larger than the ~0.01-0.02 reproducibility margin; not fully explained) |

## Important Findings
- **Class 1/2 confusion is the dominant bottleneck**: 94 of 95 true class-2 validation samples predicted as class 1 — the single largest confusion pair in the entire matrix.
- Model confidence on these errors (~0.57) is *lower* than on correct predictions (~0.94) — the model is uncertain, not confidently wrong.
- Classes 1 and 2 have nearly identical raw pixel statistics (mean intensity 185.1 vs 183.1).
- **Model B overfit**: more capacity (602K params) reduced Macro F1 vs. the 286K-param baseline.
- **Validation/test gap** (0.8053 → 0.6840) is real, reported honestly, not tuned away.
- Reproducibility: seeds fixed, oneDNN disabled, `enable_op_determinism()` used — but bit-exact reproducibility is **not** claimed; ~0.01-0.02 Macro F1 run-to-run variation was directly observed.
