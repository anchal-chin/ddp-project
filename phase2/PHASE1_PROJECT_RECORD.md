# Phase 1 Project Record
### (Reference document for Phase 2 — Phase 1 itself is frozen and untouched)

This document is a new file created for Phase 2 planning purposes. It does **not** modify, replace, or supersede `organ_cnn_project.ipynb` or `phase1_final/PHASE1_COMPLETE_TECHNICAL_REPORT.md` — it summarizes them so Phase 2 work can proceed without repeatedly re-opening the original notebook. All numbers below were re-verified against `results/metrics/experiment_log.csv`, `results/metrics/final_test_results.json`, and `organ_cnn_project.ipynb`'s actual cell outputs as of this writing — none are invented or estimated.

---

## 1. Original Project Objective

The professor provided a pre-split, 64×64 grayscale image dataset (six `.npy` files) and asked for **custom CNN models** — not pretrained/transfer-learning backbones — to be developed and evaluated for an 11-class classification task. No further requirements beyond this are documented anywhere in the project workspace; this record does not invent additional scope.

**Dataset provenance note (carried forward from Phase 1):** the class structure (11 classes, one dominant class, ~5.6× imbalance) is consistent with the publicly documented "OrganSMNIST" dataset layout, but this was never independently confirmed from any file in the workspace (no README/label-map exists). Phase 1 treated this as an unverified hypothesis, and this record does the same.

---

## 2. Dataset

Verified directly from the `.npy` files:

| Split | Images | Image shape | dtype | Pixel range |
|---|---|---|---|---|
| Train | 13,932 | (13932, 64, 64) | uint8 | 0–255 |
| Validation | 2,452 | (2452, 64, 64) | uint8 | 0–255 |
| Test | 8,827 | (8827, 64, 64) | uint8 | 0–255 |

- **Channels:** 1 (grayscale); expanded to `(N,64,64,1)` in preprocessing.
- **Classes:** 11, integer-encoded (0–10).
- **Label format:** stored as `(N,1)` column vectors, flattened to `(N,)`; used with `sparse_categorical_crossentropy` (not one-hot).
- **Available files:** `train_images.npy`, `train_labels.npy`, `val_images.npy`, `val_labels.npy`, `test_images.npy`, `test_labels.npy` — all six present, no others.

---

## 3. Dataset Analysis (EDA — already completed in Phase 1)

**Training-set class distribution:**

| Class | Count | Percent |
|---|---|---|
| 0 | 1,148 | 8.24% |
| 1 | 630 | 4.52% |
| 2 | 614 | 4.41% |
| 3 | 721 | 5.18% |
| 4 | 1,132 | 8.13% |
| 5 | 1,119 | 8.03% |
| 6 | 3,464 | 24.86% |
| 7 | 741 | 5.32% |
| 8 | 803 | 5.76% |
| 9 | 2,004 | 14.38% |
| 10 | 1,556 | 11.17% |

**OBSERVED:** imbalance ratio (largest/smallest) = 5.64×. Zero duplicate images found within any split; zero exact-duplicate images shared across splits (MD5-hash checked).

**OBSERVED (pixel statistics, raw uint8):** train mean=126.29/std=69.74; val mean=127.06/std=70.82; test mean=122.39/std=67.21. No NaN/Inf in any split.

**OBSERVED (class 1 vs. class 2 pixel statistics, training set):** mean intensity 185.14 vs. 183.05; std intensity 57.27 vs. 58.28; mean variance 1926.2 vs. 1893.4 — nearly identical.

**HYPOTHESIS (not proven):** this near-identical pixel-statistic profile, combined with the model's confusion pattern (Section 7 below), is *consistent with* classes 1 and 2 being visually difficult to distinguish at 64×64 resolution — not confirmed as the definite cause.

---

## 4. Preprocessing (already implemented in Phase 1)

```python
def preprocess_images(images):
    images = images.astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)  # (N,64,64) -> (N,64,64,1)
    return images
```
- Normalization: uint8 [0,255] → float32 [0,1] via fixed division by 255 (no leakage risk — not a computed statistic).
- Identical function applied to train/val/test — verified no branching/special-casing.
- Labels kept as integers, not one-hot.
- **Augmentation** (rotation ±10°, translation ±5%, zoom ±10%, no horizontal flip) was used **only** in Experiment 3, applied to training data only via a dedicated `tf.data` pipeline — validation/test were never augmented in any experiment.

---

## 5. Model A (Phase 1 Baseline Architecture)

Verified by loading `models/model_a_baseline_best.keras` directly (read-only inspection):

| Layer | Output Shape | Parameters |
|---|---|---|
| Input | (64,64,1) | 0 |
| Conv2D(16, 3×3, same) + ReLU | (64,64,16) | 160 |
| MaxPooling2D(2×2) | (32,32,16) | 0 |
| Conv2D(32, 3×3, same) + ReLU | (32,32,32) | 4,640 |
| MaxPooling2D(2×2) | (16,16,32) | 0 |
| Conv2D(64, 3×3, same) + ReLU | (16,16,64) | 18,496 |
| MaxPooling2D(2×2) | (8,8,64) | 0 |
| Flatten | (4096,) | 0 |
| Dense(64) + ReLU | (64,) | 262,208 |
| Dropout(0.5) | (64,) | 0 |
| Dense(11) + Softmax | (11,) | 715 |

**Total trainable parameters: 286,219** (checkpoint-verified). Note: 262,208 of these (92%) live in the single Flatten→Dense(64) step.

**Training configuration (Model A, Phase 1):**

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 (constant, no schedule) |
| Loss | `sparse_categorical_crossentropy` |
| Batch size | 64 |
| Max epochs | 30 |
| Early stopping | `monitor="val_loss"`, patience 5, `restore_best_weights=True` |
| Checkpointing | `ModelCheckpoint(monitor="val_loss", save_best_only=True)` |
| Random seed | 42 (`keras.utils.set_random_seed`) |
| Determinism controls | `TF_ENABLE_ONEDNN_OPTS=0`, `tf.config.experimental.enable_op_determinism()` |

**Final logged run:** 17 epochs, best epoch 12 (best `val_loss` = confirmed from training log text).

---

## 6. Experiments (all six, Phase 1 — source: `results/metrics/experiment_log.csv`)

| # | Experiment | Change vs. Model A | Params | Val Accuracy | Val Macro F1 | Result |
|---|---|---|---|---|---|---|
| 1 | Baseline Model A | — | 286,219 | 0.8907 | **0.8053** | Best of all 6 |
| 2 | + Class Weighting | `class_weight='balanced'` (from `y_train` only) in `.fit()` | 286,219 | 0.8732 | 0.7833 | Negative |
| 3 | + Augmentation | Rotation ±10°/translation ±5%/zoom ±10%, train-only, no flip | 286,219 | 0.8426 | 0.7763 | Negative (worst) |
| 4 | Model B | 4 blocks ×2 conv (32-32-64-64-128-128-128-128), BatchNorm, GAP, no L2 | 602,347 | 0.8858 | 0.7918 | Negative (overfit) |
| 5 | Model C | 4 blocks ×1 conv (32-64-128-128), BatchNorm, **L2=1e-4**, GAP | 250,635 | 0.8825 | 0.7796 | Negative |
| 6 | + Focal Loss | γ=2.0, no α term, Model A architecture unchanged | 286,219 | 0.8760 | 0.7925 | Negative |

**Every one of the five alternatives reduced Macro F1 relative to the plain baseline.** This is the central Phase 1 finding and the direct motivation for Phase 2's cautious, evidence-first approach.

Per-experiment purpose/hypothesis/conclusion detail is in `phase1_final/PHASE1_COMPLETE_TECHNICAL_REPORT.md`, Sections 10–14 — not repeated in full here to avoid duplication drift; this table is the authoritative number summary.

---

## 7. Error Analysis (Phase 1 findings)

**OBSERVED:** Class 1↔2 is the single largest confusion pair in the entire 11×11 confusion matrix — in the final documented run, 94 of 95 true class-2 validation samples were predicted as class 1 (next-largest confusion pair anywhere in the matrix: 53).

**OBSERVED (confidence analysis):** model confidence on class-2→1 errors ≈0.57, vs. ≈0.94 on correct predictions — the model was genuinely uncertain on these errors, not confidently wrong.

**OBSERVED (data-quality check):** no blank/near-blank images or gross statistical anomalies found in classes 1 or 2; outlier flagging found only the statistically-expected small number of low-variance images per class (7 each), consistent with normal distribution tails, not corruption.

**HYPOTHESIS (explicitly not proven):** classes 1 and 2 may be difficult to visually distinguish at 64×64 resolution. This is **not** established as fact — it is the leading explanation given that five mechanistically distinct interventions (frequency reweighting, augmentation, more capacity, regularized capacity, difficulty-aware reweighting) all failed to resolve it.

---

## 8. Model Selection

**Model A (baseline) was selected as the final Phase 1 model**, based on:
- Highest validation Macro F1 (0.8053) of all six candidates — primary criterion.
- Fewest parameters (286,219 vs. up to 602,347 for Model B).
- Least overfitting of any architecture tested.
- **Not** selected on accuracy alone — Model B's accuracy (0.8858) was close to Model A's (0.8907), but its Macro F1 (0.7918) was clearly worse, which would have been hidden by accuracy alone.

Alternatives were not selected because every one of them scored lower on the pre-specified primary metric (Macro F1) — documented as negative results, not discarded or hidden.

---

## 9. Final Test Evaluation (already completed in Phase 1 — NOT re-run for this document)

Source: `results/metrics/final_test_results.json`.

| Metric | Value |
|---|---|
| Test Accuracy | **0.7339** |
| Test Macro Precision | 0.7007 |
| Test Macro Recall | 0.6857 |
| **Test Macro F1** | **0.6840** |

Full per-class test results and confusion matrix are documented in `phase1_final/PHASE1_COMPLETE_TECHNICAL_REPORT.md`, Section 18.

**Validation → test gap:** Macro F1 dropped 0.8053 → 0.6840 (−0.1213), larger than the ~0.01–0.02 run-to-run reproducibility margin observed in Phase 1. Two hypotheses were offered (model-selection bias from comparing six candidates on one validation set; a possible validation/test distributional difference) — **neither confirmed.** This evaluation happened exactly once, after model selection, and the model was not modified afterward.

---

## 10. Phase 1 Conclusion

**The original CNN project (Phase 1) was completed.** Dataset analysis, preprocessing, a custom baseline CNN, five additional controlled experiments, error analysis, model selection, and a single final test evaluation were all performed and documented. The final selected model (Model A) achieved **Test Accuracy ≈ 73.39%, Test Macro F1 ≈ 0.6840**. This is the **historical benchmark** Phase 2 will be measured against — not a target to reproduce by re-running the test set, and not a number Phase 2 is permitted to tune toward (see `phase2/README.md`).

**This Phase 1 record does not draw any Phase 2 conclusions.** Phase 2 has not started as of this document.
