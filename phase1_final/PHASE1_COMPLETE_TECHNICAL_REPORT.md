# Phase 1 — Complete Technical Report
## Custom CNN-Based 11-Class Image Classification

**Status:** Phase 1 complete. This document is the permanent technical record of Phase 1, written for future reference. All numbers in this report were re-verified directly against the project's saved artifacts (notebook cell outputs, `results/metrics/experiment_log.csv`, `results/metrics/final_test_results.json`, and the saved `.keras` checkpoints) as part of producing this document — none were retyped from memory or assumed.

Source of truth used: `organ_cnn_project.ipynb`, `results/metrics/experiment_log.csv`, `results/metrics/final_test_results.json`, `results/metrics/experiment_config.json`, and the six saved model checkpoints in `models/`.

---

## Section 1 — Project Background

The professor provided a pre-split image dataset as six `.npy` files (`train_images.npy`, `train_labels.npy`, `val_images.npy`, `val_labels.npy`, `test_images.npy`, `test_labels.npy`) and asked for **custom CNN models** to be developed — not pretrained/transfer-learning backbones.

Verified properties of the data (re-checked directly from the files for this report):
- Image resolution: **64×64**
- **Grayscale** (single channel)
- **Multi-class classification**, **11 classes**
- Pixel dtype `uint8`, range 0–255

**Objective:** build and evaluate custom CNN architectures on this dataset, using a scientifically defensible process — not simply train one model and report a number.

**On dataset naming:** the workspace contains no README, label-map, or metadata file. The class structure (11 classes, one dominant class, ~5.6× imbalance ratio) is consistent with the publicly documented "OrganSMNIST" dataset structure, but **this was never independently confirmed from any file in this project**. This report and the accompanying presentation refer to the data only by its verified properties (resolution, channel count, class count) and do not assert a specific dataset name as fact.

---

## Section 2 — Dataset

Directly re-verified by loading the `.npy` files for this report:

| Split | Images | Image shape | Label shape | dtype | Pixel range |
|---|---|---|---|---|---|
| Train | **13,932** | (13932, 64, 64) | (13932, 1) → flattened (13932,) | uint8 | 0–255 |
| Validation | **2,452** | (2452, 64, 64) | (2452, 1) → flattened (2452,) | uint8 | 0–255 |
| Test | **8,827** | (8827, 64, 64) | (8827, 1) → flattened (8827,) | uint8 | 0–255 |

- **Classes:** 11, integer-encoded (0–10), **not one-hot**.
- **Label encoding:** stored as `(N, 1)` column vectors; flattened to `(N,)` in preprocessing for use with `sparse_categorical_crossentropy`.

**Training-set class distribution** (from notebook Section 5, computed on `y_train` only):

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

- **Imbalance ratio (largest/smallest):** 5.64× (class 6: 3,464 vs. class 2: 614)
- **Why imbalance matters:** a classifier can achieve high overall accuracy while completely failing minority classes, because errors on small classes barely move the accuracy needle. This is the direct motivation for using **Macro F1** (Section 8) instead of accuracy as the primary metric.

Validation/test class distributions were not separately tabulated as their own notebook cell (train distribution was the only split explicitly tabulated in the deliverable notebook, per Section 5's "training set only" scope). *Not available beyond what is stated above / not independently verifiable as a separate tabulated artifact for val/test splits.*

---

## Section 3 — Exploratory Data Analysis

Actually performed and present in the notebook (Section 2, Section 3, Section 3b, Section 5):

- **Sample image visualization** — grid of real training images per class (`baseline_sample_images.png`).
- **Class distribution** — table and bar chart (`baseline_train_class_distribution.png`), see Section 2 above.
- **Pixel statistics** (raw uint8, all three splits, re-verified for this report):

| Split | Min | Max | Mean | Std | NaN | Inf |
|---|---|---|---|---|---|---|
| Train | 0 | 255 | 126.286 | 69.738 | 0 | 0 |
| Val | 0 | 255 | 127.064 | 70.818 | 0 | 0 |
| Test | 0 | 255 | 122.387 | 67.206 | 0 | 0 |

- **Duplicate check** (MD5 hash of raw pixels, within each split): train 13,932/13,932 unique, val 2,452/2,452 unique, test 8,827/8,827 unique — **zero duplicates in any split**.
- **Cross-split leakage check** (identical images shared between splits): train↔val 0, train↔test 0, val↔test 0 — **no exact-duplicate leakage found**.

**OBSERVED FACT:** classes 1 and 2 have nearly identical raw per-class pixel statistics (mean intensity 185.1 vs. 183.1, similar variance) — see Section 16 for the full data-quality table.

**HYPOTHESIS (not proven):** this pixel-statistic similarity, combined with the model's confusion pattern, is *consistent with* classes 1 and 2 being visually difficult to distinguish at 64×64 resolution. This is not confirmed as the definite cause — see Section 16 for the careful distinction between what was observed and what is inferred.

---

## Section 4 — Preprocessing

Implemented exactly as follows (verified directly from the `preprocess_images()` function in the notebook):

```python
def preprocess_images(images):
    images = images.astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)  # (N, 64, 64) -> (N, 64, 64, 1)
    return images
```

1. **Pixel normalization:** uint8 [0, 255] → float32 [0, 1] via division by the fixed constant 255 (not a computed statistic — avoids any leakage between splits).
2. **dtype handling:** explicit cast to `float32` for numerical stability with the Adam optimizer.
3. **Channel dimension:** `np.expand_dims(..., axis=-1)` — (N,64,64) → (N,64,64,1), as required by `Conv2D`.
4. **Label handling:** kept as plain integers (0–10), not one-hot encoded, because `sparse_categorical_crossentropy` accepts integer labels directly.
5. **Training preprocessing:** identical function applied to `train_images`.
6. **Validation preprocessing:** identical function applied to `val_images` — **no special-casing**, confirmed by direct code inspection (single shared function, no branching).
7. **Test preprocessing:** identical function applied to `test_images` — same guarantee.
8. **Augmentation pipeline:** used **only** in Experiment 3 (Section 11) — applied via a `tf.data` `.map()` call on the training dataset only; validation used a separate, unaugmented `tf.data` pipeline (`val_ds_plain`) with zero augmentation calls in its construction. No augmentation was ever applied to validation or test data in any experiment.

---

## Section 5 — Why CNN?

Images have local spatial structure: nearby pixels are correlated and meaningful patterns (edges, shapes, textures) are position-local before they become position-global. A CNN's convolutional layers exploit this via:

- **Parameter sharing** — the same small filter (e.g. 3×3) slides across the whole image, so the network doesn't need a separate weight for every pixel position, unlike a fully-connected layer.
- **Local receptive fields** — early layers detect small, local patterns (edges, corners); as layers stack, receptive fields grow, allowing later layers to detect larger, more abstract structures.
- **Hierarchical feature learning** — the network learns simple-to-complex features automatically through training, rather than requiring hand-engineered features.
- **Translation-limited invariance** via pooling — small shifts in where a pattern appears don't require entirely new learned features.

For a 64×64 grayscale input with a modest training set (~14K images), a small custom CNN is well-matched: enough capacity to learn the task, few enough parameters to avoid needing a massive dataset or a pretrained backbone.

---

## Section 6 — Baseline Model A

**Verified directly by loading `models/model_a_baseline_best.keras` and inspecting it (no retraining performed for this verification).**

| Layer | Output Shape | Parameters | Purpose |
|---|---|---|---|
| Input | (64, 64, 1) | 0 | Grayscale image input |
| Conv2D(16, 3×3, same) + ReLU | (64, 64, 16) | 160 | Low-level feature extraction |
| MaxPooling2D(2×2) | (32, 32, 16) | 0 | Downsampling |
| Conv2D(32, 3×3, same) + ReLU | (32, 32, 32) | 4,640 | Mid-level features |
| MaxPooling2D(2×2) | (16, 16, 32) | 0 | Downsampling |
| Conv2D(64, 3×3, same) + ReLU | (16, 16, 64) | 18,496 | Higher-level features |
| MaxPooling2D(2×2) | (8, 8, 64) | 0 | Downsampling |
| Flatten | (4,096,) | 0 | Vectorize feature map |
| Dense(64) + ReLU | (64,) | 262,208 | Learned combination of features |
| Dropout(0.5) | (64,) | 0 | Regularization |
| Dense(11) + Softmax | (11,) | 715 | Class probabilities |

**Total trainable parameters (verified from loaded checkpoint): 286,219** — matches the value reported throughout the project.

Architecture was checkpoint-verified to be an **exact match** to this table (layer types, filter counts, kernel sizes, activations, output shapes all confirmed by direct inspection of the saved model object).

**Why this was a reasonable starting architecture:** 64×64 grayscale images are far simpler than natural RGB photos (single-channel, low texture complexity relative to e.g. ImageNet-scale images). A shallow 3-block CNN is a legitimate, non-underpowered baseline for this input scale — it's meant to prove the pipeline works and give later, more complex models something concrete to beat, not to be claimed as optimal.

---

## Section 7 — Baseline Training

Verified from the training cell and its logged output:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss | `sparse_categorical_crossentropy` |
| Batch size | 64 |
| Max epochs | 30 |
| Early stopping | `monitor="val_loss"`, patience 5, `restore_best_weights=True` |
| Checkpointing | `ModelCheckpoint(monitor="val_loss", save_best_only=True)` |
| Validation strategy | `validation_data=(X_val, y_val)` — held-out validation set, never test |
| Random seed | 42 (`keras.utils.set_random_seed(SEED)`) |

**Actual training run (final logged run):** 17 epochs before early stopping fired; best `val_loss` improvement occurred at epoch 12 and did not improve again for 5 subsequent epochs, triggering early stopping at epoch 17. This is directly confirmed in the raw training log text (`Epoch 12: val_loss improved... Epoch 17: early stopping`), matching `experiment_log.csv`'s `epochs_run=17, best_epoch=12`.

**Why early stopping:** prevents training past the point where the model starts overfitting — training loss can keep falling indefinitely, but validation loss is the honest signal of generalization. **Why validation, not training, performance matters:** training performance only tells you the model can fit the data it has already seen; validation performance (on unseen-during-training data) is the actual proxy for how the model will perform on new data.

---

## Section 8 — Why Macro F1?

- **Accuracy** = fraction of correct predictions. Simple, but blind to which classes are wrong.
- **Precision** (per class) = of predicted-as-class-X, how many actually are class X.
- **Recall** (per class) = of actually-class-X, how many were correctly found.
- **F1** (per class) = harmonic mean of precision and recall for that class.
- **Macro F1** = the *unweighted* average of per-class F1 across all 11 classes.

**Why accuracy alone is insufficient here:** with a 5.64× class imbalance, a model can score ~89% accuracy while completely failing a minority class (this is exactly what happened — see Section 9: baseline accuracy 89.07% but class 2 F1 only 0.021). Because large classes dominate the accuracy calculation, a model that is excellent on the majority classes and terrible on minority classes still looks good on accuracy alone.

**Why Macro F1 was chosen as the primary comparison metric:** because it weights every class *equally* regardless of size, failing a small class (like class 2, with only 614 training examples) is directly visible in the score — it cannot be hidden by good performance on class 6 (3,464 examples). This was the deciding factor in evaluating every experiment in this project.

---

## Section 9 — Baseline Results

**Source: `experiment_log.csv` row "Exp1: Baseline Model A", and notebook Section 10 output — re-verified identical for this report.**

| Metric | Value |
|---|---|
| Accuracy | **0.8907** |
| Macro Precision | 0.8866 |
| Macro Recall | 0.8353 |
| **Macro F1** | **0.8053** |

**Per-class results (validation set):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 | 0.960 | 0.899 | 0.929 | 188 |
| 1 | 0.523 | 0.990 | 0.684 | 104 |
| 2 | 1.000 | 0.011 | **0.021** | 95 |
| 3 | 0.996 | 0.976 | 0.986 | 246 |
| 4 | 0.651 | 0.971 | 0.779 | 140 |
| 5 | 0.922 | 0.667 | 0.774 | 159 |
| 6 | 0.970 | 0.986 | 0.978 | 491 |
| 7 | 0.920 | 0.923 | 0.922 | 261 |
| 8 | 0.925 | 0.898 | 0.911 | 275 |
| 9 | 0.936 | 0.886 | 0.910 | 280 |
| 10 | 0.950 | 0.981 | 0.965 | 213 |

**Key finding:** overall accuracy (89%) is strong, but class 2's F1 (0.021) reveals the model essentially never correctly identifies class 2 — it has precision 1.000 (on the rare occasions it does predict class 2, it's right) but recall 0.011 (it almost never predicts class 2 at all). Figures: `baseline_confusion_matrix.png`, `baseline_training_curves.png`.

---

## Section 10 — Experiment 2: Class Weighting

**Why attempted:** class 2's near-zero recall in the baseline suggested the model might simply need stronger incentive to predict minority classes.

**Implementation (verified in code):** `sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=classes_sorted, y=y_train)` — **weights calculated from `y_train` only**, confirmed by direct code inspection (no validation or test labels used). Class 2 (smallest, 614 samples) received the highest weight (2.0628); class 6 (largest, 3,464 samples) received the lowest (0.3656) — confirmed minority classes get higher weights, as expected. Weights were passed via `class_weight=class_weight_dict` directly to `model.fit()` — confirmed present in the actual `.fit()` call.

**Hypothesis:** upweighting rare-class errors would push the model to predict class 2 more often.

**Actual result:**

| Metric | Baseline | Class Weighting | Change |
|---|---|---|---|
| Accuracy | 0.8907 | 0.8732 | −0.0175 |
| **Macro F1** | **0.8053** | **0.7833** | **−0.0220** |
| Class 1 F1 | 0.684 | 0.676 | −0.008 |
| Class 2 F1 | 0.021 | 0.000 | −0.021 |

**Class 1/2 behavior:** class 2's recall/F1 got *worse*, not better — it dropped to 0.000. Classes 1 and 2 have almost identical training counts (630 vs. 614), so "balanced" weighting assigned them nearly equal weights — this does nothing to resolve a confusion driven by two classes looking alike rather than one being statistically rarer.

**Conclusion — NEGATIVE RESULT:** class weighting reduced Macro F1 and made the target problem (class 2) worse, not better. This was not hidden or discarded — it is preserved as an important, informative negative result: the class 1/2 confusion is not a frequency-imbalance problem.

---

## Section 11 — Experiment 3: Augmentation

**Augmentation used** (verified from code — `keras.Sequential([RandomRotation, RandomTranslation, RandomZoom])`):
- Rotation: ±10° (`factor=10/360`)
- Translation: ±5% (`height_factor=0.05, width_factor=0.05`)
- Zoom: ±10% (`height_factor=0.1, width_factor=0.1`)
- Fill mode: constant, value 0.0 (black — matches the dark image background observed in EDA)
- **No `RandomFlip` layer present** — directly confirmed by inspecting the `keras.Sequential` definition; the notebook's claim of "no horizontal flipping" is verified true in the actual code, not just stated.

**Why attempted:** to test whether more training-image variation would help the model generalize better, particularly for minority/confusable classes.

**Training-only application (verified):** augmentation is applied inside `train_ds_aug`'s `.map()` call; the separate `val_ds_plain` pipeline has **zero** augmentation calls in its construction — confirmed by direct code inspection.

**Actual result:**

| Metric | Baseline | Augmentation | Change |
|---|---|---|---|
| Accuracy | 0.8907 | 0.8426 | −0.0481 |
| **Macro F1** | **0.8053** | **0.7763** | **−0.0290** |

Per-experiment-log class-level detail: class 1 F1 dropped to 0.552, class 2 F1 rose to 0.281 (an improvement on this one class, but not enough to offset broad degradation elsewhere — notably class 5, which the log records as substantially degraded).

**Conclusion — NEGATIVE RESULT:** the tested augmentation configuration reduced Macro F1 below baseline. A plausible (not proven) explanation offered in the notebook: these images appear consistently framed/positioned, and the model may rely on positional cues the augmentation disrupted, without enough training budget to compensate.

---

## Section 12 — Experiment 4: Model B

**Architecture (verified from loaded `models/model_b_best.keras`):** 4 convolutional blocks, 2 Conv2D layers per block (32→32→64→64→128→128→128→128 filters), BatchNormalization + ReLU after every conv, MaxPooling after each block, GlobalAveragePooling2D, Dense(128)+ReLU, Dropout(0.5), Dense(11, softmax). **No L2 regularization** (confirmed — `kernel_regularizer=None` on every conv layer, checked directly).

**Verified parameter count (from loaded checkpoint): 602,347.**

**Motivation / differences from Model A:** deeper (4 blocks vs. 3), double conv per block (8 conv layers vs. 3), BatchNorm added, GlobalAveragePooling replacing Flatten+large-Dense (Model A's Flatten→Dense(64) step alone accounted for 262,208 of its 286,219 params — GAP was intended to shift capacity into the convolutional feature extractor).

**Training behavior / overfitting:** training accuracy climbed to 90%+ while validation accuracy plateaued much lower, and validation loss stopped improving early (best epoch 4 of 12 epochs run) while training loss kept falling for the remaining epochs — the training log directly shows this pattern.

**Actual result:**

| Metric | Baseline | Model B | Change |
|---|---|---|---|
| Accuracy | 0.8907 | 0.8858 | −0.0049 |
| **Macro F1** | **0.8053** | **0.7918** | **−0.0135** |
| Class 2 F1 | 0.021 | 0.021 | 0.000 (no improvement) |

**Conclusion:** simply increasing capacity did not solve the class 1/2 problem and reduced Macro F1 relative to baseline — the added capacity manifested as overfitting rather than better generalization.

---

## Section 13 — Experiment 5: Model C

**Architecture (verified from loaded `models/model_c_best.keras`):** 4 convolutional blocks, **1** Conv2D layer per block (32→64→128→128 filters), BatchNormalization + ReLU after each conv, **L2 kernel regularization (λ=1e-4) confirmed present on all 4 conv layers** (checked directly on the loaded layer objects), MaxPooling after each block, GlobalAveragePooling2D, Dense(64)+ReLU, Dropout(0.5), Dense(11, softmax).

**Verified parameter count (from loaded checkpoint): 250,635** — smaller than Model A despite one more block, because GAP removes the large Flatten→Dense cost.

**Motivation:** test whether Model B's overfitting was caused by *raw capacity* (2 convs/block, no weight decay) rather than the BatchNorm/GAP pattern itself, by halving conv-per-block and adding L2 as a direct, evidence-based response to Model B's demonstrated overfitting.

**Actual result:**

| Metric | Baseline | Model C | Change |
|---|---|---|---|
| Accuracy | 0.8907 | 0.8825 | −0.0082 |
| **Macro F1** | **0.8053** | **0.7796** | **−0.0257** |
| Class 2 F1 | 0.021 | 0.079 | +0.058 (small improvement, still far below usable) |

Training was visibly unstable in early epochs (validation accuracy swung sharply between consecutive epochs before stabilizing), and this experiment's Macro F1 also came in *below* Model B's in this specific run — a reversal from another documented rerun where C beat B, indicating Model C's relative standing versus Model B is not stable across reruns, though **both consistently trail Model A in every run observed.**

**What this experiment taught us:** regularization changes alone did not close the gap to the baseline, and did not reliably fix class 2 — reinforcing that the dominant problem is not simply "too much capacity, needs more regularization."

---

## Section 14 — Experiment 6: Focal Loss

**Why considered:** class weighting (frequency-based reweighting) failed; focal loss reweights by *prediction difficulty* instead, which is mechanistically different — testing whether a difficulty-aware loss (rather than a frequency-aware one) could help.

**Verified γ = 2.0**, implemented as a custom function:
```python
def sparse_categorical_focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        ...
        focal_weight = tf.pow(1.0 - probs, gamma)
        ce = -tf.math.log(probs)
        return tf.reduce_mean(focal_weight * ce)
    return loss_fn
```
Confirmed passed directly as `loss=sparse_categorical_focal_loss(gamma=2.0)` in `model.compile()`. **Deliberately no class-balancing α term** — to isolate the difficulty-focusing mechanism from the already-tested frequency-reweighting mechanism (Experiment 2). All other settings (architecture = Model A, optimizer, learning rate, batch size) were unchanged.

**Actual result:**

| Metric | Baseline | Focal Loss | Change |
|---|---|---|---|
| Accuracy | 0.8907 | 0.8760 | −0.0147 |
| **Macro F1** | **0.8053** | **0.7925** | **−0.0128** |
| Class 2 F1 | 0.021 | 0.021 | 0.000 (no change) |

**Conclusion:** focal loss did not improve Macro F1 and did not fix the class 1/2 confusion — it was not carried forward as the final approach because it underperformed the plain baseline on the primary metric.

---

## Section 15 — Master Experiment Comparison

**All values re-verified directly against `results/metrics/experiment_log.csv` for this report — every number below matches that file exactly.**

| Experiment | Architecture/Change | Params | Accuracy | Macro Prec. | Macro Recall | **Macro F1** | Class 1 F1 | Class 2 F1 | Conclusion |
|---|---|---|---|---|---|---|---|---|---|
| Baseline Model A | 3 blocks, 16-32-64, Flatten+Dense64 | 286,219 | 0.8907 | 0.8866 | 0.8353 | **0.8053** | 0.684 | 0.021 | **Best — selected** |
| + Class Weighting | Balanced `class_weight` on Model A | 286,219 | 0.8732 | 0.7808 | 0.8148 | 0.7833 | 0.676 | 0.000 | Negative |
| + Augmentation | Rotation/translation/zoom on Model A | 286,219 | 0.8426 | 0.8033 | 0.7702 | 0.7763 | 0.552 | 0.281 | Negative (worst) |
| Model B | 4 blocks ×2 conv, BN, GAP, no L2 | 602,347 | 0.8858 | 0.8464 | 0.8183 | 0.7918 | 0.662 | 0.021 | Negative (overfit) |
| Model C | 4 blocks ×1 conv, BN, L2, GAP | 250,635 | 0.8825 | 0.8613 | 0.8179 | 0.7796 | 0.687 | 0.079 | Negative |
| + Focal Loss (γ=2.0) | Focal loss on Model A | 286,219 | 0.8760 | 0.8721 | 0.8207 | 0.7925 | 0.689 | 0.021 | Negative |

**Every single one of the five alternatives to the baseline reduced Macro F1.** This consistent pattern is itself the central experimental finding of Phase 1.

---

## Section 16 — Error Analysis

**Dominant confusion pair — class 1 ↔ class 2.** From the notebook's confusion-pair ranking (all 11×11 matrix entries, sorted by count):

| Count | True → Predicted |
|---|---|
| **94** | **2 → 1** |
| 53 | 5 → 4 |
| 21 | 8 → 7 |
| 20 | 7 → 8 |
| 16 | 9 → 4 |
| 12 | 0 → 9 |

**OBSERVED FACT:** 94 of 95 true class-2 validation samples were predicted as class 1 in the relevant final baseline/error-analysis run (only 1 correctly identified as class 2). This is the single largest confusion pair anywhere in the entire 11×11 matrix — nearly double the next-largest (53).

Class 1's own accuracy was high in isolation (103/104 correct), meaning class 1 essentially absorbed almost the entirety of class 2's samples in addition to its own.

**Confidence analysis (OBSERVED):**

| | Confidence |
|---|---|
| Correct predictions (all classes) | 0.937 |
| Incorrect predictions (all classes) | 0.663 |
| Class 2 → predicted class 1 errors | 0.568 |
| Class 1 → predicted class 2 errors | n/a (0 such errors occurred in this run) |

The model's confidence on the class-2→1 errors (0.568) is well below its confidence when correct (0.937) — **the model was generally uncertain on these difficult errors, not extremely confident.** This is stated as an observed fact from the confidence distribution, not an inference.

**Data-quality analysis (OBSERVED, from training-set per-class pixel statistics):**

| Class | N | Mean Intensity | Std Intensity | Mean Variance | Mean Near-Zero % |
|---|---|---|---|---|---|
| 1 | 630 | 185.14 | 57.27 | 1926.2 | 2.57% |
| 2 | 614 | 183.05 | 58.28 | 1893.4 | 3.44% |
| 4 | 1,132 | 124.58 | 27.04 | 2878.7 | 3.49% |
| 5 | 1,119 | 134.31 | 29.54 | 2941.8 | 2.26% |

Classes 1 and 2 have nearly identical raw pixel statistics (mean intensity 185.14 vs. 183.05). No blank/near-blank images or gross anomalies were found in either class via the outlier check (1st-percentile-variance flagging found only the statistically expected small number of low-variance images per class — 7 in class 1, 7 in class 2 — consistent with normal distribution tails, not data corruption).

**OBSERVED:** class 1/2 is a major, dominant confusion pair; class 2 is very frequently predicted as class 1; confidence analysis was performed; incorrect examples were visually inspected (`error_analysis_2_as_1.png`, `error_analysis_top_confident_errors.png`); per-class performance was examined in full.

**HYPOTHESIS (explicitly not proven):** the combination of (a) five different, mechanistically distinct interventions all failing to fix this specific confusion, (b) near-identical raw pixel statistics between the two classes, and (c) the model's genuine uncertainty (not overconfidence) on these errors, is **consistent with** classes 1 and 2 being difficult to visually distinguish at the available 64×64 resolution. This project does **not** claim the classes are "proven indistinguishable" — that would require testing at higher resolution or with additional information this project does not have access to.

---

## Section 17 — Reproducibility

**Controls implemented (verified present in the notebook's Section 1 code, executed before any TensorFlow import):**
- `os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"` — disables the TensorFlow CPU kernel path identified as the primary source of run-to-run numerical variation.
- `tf.config.experimental.enable_op_determinism()`
- `keras.utils.set_random_seed(SEED)` with `SEED = 42` — seeds Python `random`, NumPy, and TensorFlow together.
- `os.environ["PYTHONHASHSEED"] = "42"`

**Run-to-run variation observed (documented, not hidden):** even with these controls, repeated full executions of the notebook produced slightly different metrics (e.g. baseline Macro F1 was observed at both 0.8053 and 0.8156 across two full runs during this project — a difference of about 0.01). Per-class metrics on small classes (e.g. class 2, 95 validation samples) showed noticeably more run-to-run variation than the aggregate Macro F1.

**Root cause (identified, not merely guessed):** TensorFlow's CPU backend can select different low-level kernel execution orders depending on runtime/thread conditions; since floating-point addition is not strictly associative, this compounds across many layers and training steps into small but real differences in final metrics.

**Honest limitation, as stated in the notebook itself:** *"this is not a claim of bit-exact reproducibility... exact bit-level reproducibility is not guaranteed on the current CPU environment."* Full determinism would require single-threaded execution, which was not applied by default due to the training-time cost on this CPU-only machine.

**Experiment logging/config tracking:** every experiment's configuration and validation metrics were appended to `results/metrics/experiment_log.csv` via a shared `log_experiment()` helper (verified: no hard-coded metric literals found anywhere in the notebook's code cells via direct regex scan). `results/metrics/experiment_config.json` records the shared configuration (seed, optimizer, learning rate, batch size, loss, preprocessing description, validation protocol).

---

## Section 18 — Test Set Evaluation

**This evaluation already exists in the notebook (Phase 9) and was NOT rerun to produce this report.**

**Source: `results/metrics/final_test_results.json`, re-verified for this report.**

| Metric | Value |
|---|---|
| Test Accuracy | **0.7339** |
| Test Macro Precision | 0.7007 |
| Test Macro Recall | 0.6857 |
| **Test Macro F1** | **0.6840** |

**Per-class test results:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 | 0.859 | 0.840 | 0.849 | 811 |
| 1 | 0.473 | 0.688 | 0.561 | 439 |
| 2 | 0.412 | 0.180 | 0.250 | 445 |
| 3 | 0.889 | 0.678 | 0.770 | 510 |
| 4 | 0.453 | 0.585 | 0.511 | 704 |
| 5 | 0.524 | 0.378 | 0.439 | 693 |
| 6 | 0.888 | 0.931 | 0.909 | 2,078 |
| 7 | 0.893 | 0.887 | 0.890 | 397 |
| 8 | 0.882 | 0.888 | 0.885 | 439 |
| 9 | 0.739 | 0.743 | 0.741 | 1,343 |
| 10 | 0.694 | 0.744 | 0.718 | 968 |

Confusion matrix: `final_test_confusion_matrix.png`.

**This evaluation occurred after model selection** — verified by direct code-structure inspection: the model-selection cell (comparing Model A/B/C/Focal) reads only validation-derived result dictionaries (`res_a`, `res_b`, `res_c`, `res_focal`, all built from `model.predict(X_val)`), and the test-evaluation cells appear afterward in the notebook, using `model_a.predict(X_test)` for the first and only time.

**Validation → test gap:** Macro F1 dropped from 0.8053 (validation) to 0.6840 (test), a drop of 0.1213 — larger than the ~0.01–0.02 run-to-run reproducibility margin documented in Section 17.

**Possible explanations (explicitly HYPOTHESES, not proven causes):**
1. **Model-selection bias** — six candidates were compared on the same 2,452-image validation set before selecting Model A; the winning model's validation score can be somewhat optimistically biased by the comparison process itself (a well-known statistical effect of repeated model comparison against one validation set).
2. **A possible distributional difference between the validation and test splits** — the drop is broad across most classes rather than isolated to one or two, which is *consistent with* (but does not *prove*) a structural difference between the splits that this project has no metadata available to confirm or rule out.

Neither explanation is confirmed. The test set was evaluated exactly once, and the model was not modified after seeing this result.

---

## Section 19 — Final Model Selection

**Model A (baseline) was selected**, based on:
- **Highest validation Macro F1** among all six candidates (0.8053) — the primary, pre-specified criterion.
- **Fewest parameters** (286,219 vs. up to 602,347 for Model B) — a simpler model.
- **Least overfitting** — Model A did not show the train/validation divergence that Model B and Model C exhibited.
- **Best overall validation behavior**, considered alongside accuracy, per-class F1, and the confusion matrix — not on any single number in isolation.

**Why accuracy alone was not used:** Model B had comparable/higher accuracy (0.8858) than some might expect to matter, but its Macro F1 (0.7918) was clearly below Model A's (0.8053) — using accuracy alone would have obscured this and risked selecting a model that fails minority classes more severely, which is exactly the failure mode this project set out to avoid (Section 8).

---

## Section 20 — What Failed

| Approach | Why We Tried It | What Happened | What We Learned |
|---|---|---|---|
| Class Weighting | Class 2 had near-zero recall; weighting is the standard first response to imbalance | Macro F1 dropped (0.8053→0.7833); class 2 F1 went to 0.000 | The problem is not frequency-driven — classes 1/2 have almost equal sample counts, so weighting barely changed their relative treatment |
| Augmentation | Test whether more training variation improves robustness/generalization | Macro F1 dropped further (→0.7763), worst of all six experiments | Generic spatial augmentation can hurt when the model may rely on consistent framing/positional cues |
| Model B (deeper) | Test whether more capacity resolves the confusion | Macro F1 dropped (→0.7918); visible overfitting (train accuracy well above validation) | More capacity without matching regularization simply overfits; doesn't fix a visual-similarity problem |
| Model C (regularized) | Test whether Model B's failure was fixable via less capacity + L2 | Macro F1 still below baseline (→0.7796); unstable early training; inconsistent vs. Model B across reruns | Regularization improved training stability characteristics somewhat but did not close the gap to the simple baseline |
| Focal Loss | Test difficulty-aware reweighting instead of frequency-aware reweighting | Macro F1 dropped (→0.7925); class 2 F1 unchanged at 0.021 | Even a mechanistically different reweighting strategy did not resolve the confusion — reinforces that this is not simply a "make the model try harder" problem |

Every failed experiment was informative: together they narrow the explanation space for the class 1/2 problem away from "insufficient model effort/capacity/frequency-handling" and toward genuine visual similarity (see Section 16).

---

## Section 21 — Obstacles Encountered

- **Class imbalance** (5.64× ratio) — directly observed and quantified in Section 2.
- **Class 1/2 confusion** — the dominant, unresolved technical bottleneck (Section 16).
- **Overfitting in Model B** — directly observed via train/validation accuracy and loss divergence.
- **Augmentation hurting performance** — observed, not merely feared in advance.
- **Class weighting failing to solve the difficult class** — observed, and made the specific target problem worse.
- **Focal loss failing to improve Macro F1** — observed.
- **Run-to-run CPU reproducibility variation** — observed and root-caused (Section 17); required implementing determinism controls mid-project.
- **Validation/test performance gap** — observed and honestly reported (Section 18), not fully explained.
- **Uncertain dataset provenance** — no metadata file exists in the workspace; addressed by not asserting an unverified dataset name as fact throughout this project.
- **Computational cost of training** — all six experiments plus verification runs were executed on a CPU-only environment; individual training runs ranged from a few minutes (Model A variants) to over 25 minutes (Model B), making full-notebook reruns a genuinely expensive, multi-hour undertaking across the project.

None of these are exaggerated beyond what is directly supported by the logged artifacts referenced above.

---

## Section 22 — What We Learned

- **Bigger is not always better** — Model B (602K params) underperformed Model A (286K params) on the primary metric.
- **Class weighting does not necessarily solve visually difficult classes** — it addresses frequency imbalance, not visual confusability, and can actively hurt Macro F1 when misapplied to a confusion that isn't frequency-driven.
- **Augmentation must respect image semantics** — generic spatial transforms are not automatically beneficial; they measurably hurt performance in this specific tested configuration.
- **Macro F1 is essential for imbalanced classification** — accuracy alone would have hidden the class 2 failure entirely (89% accuracy, 0.021 F1 on that class).
- **Error analysis is essential** — it identified the *specific* dominant failure mode (class 1/2) rather than leaving "the model isn't perfect" as a vague, unhelpful conclusion.
- **Validation performance and test performance can differ meaningfully** — a 0.12 Macro F1 gap was observed and reported honestly, not hidden or tuned away.
- **Simple custom CNNs can outperform more complex alternatives on a given dataset** — this was the single most consistent finding across all six experiments.

---

## Section 23 — Final Phase 1 Conclusion

**The original CNN project (Phase 1) has been completed.** We successfully:

- Analyzed the dataset (structure, class distribution, pixel statistics, duplicate/leakage checks).
- Implemented preprocessing (normalization, channel handling, label handling), applied identically and correctly across all three splits.
- Built a custom baseline CNN (Model A) and two additional custom architectures (Model B, Model C) — no pretrained backbones used anywhere.
- Trained six distinct model/strategy combinations, each with a single, clearly isolated changed variable.
- Evaluated every experiment on a consistent metric suite (accuracy, macro precision/recall/F1, per-class F1, confusion matrix).
- Performed detailed error analysis (confusion-pair ranking, confidence analysis, representative-example inspection) and data-quality checks (pixel statistics, outlier flagging).
- Selected a final model (Model A) based on validation Macro F1, corroborated by simplicity and reduced overfitting — not accuracy alone.
- Performed a single, final test-set evaluation, after model selection, and reported the result honestly including an unresolved validation/test gap.

**This result is not presented as perfect.** The class 1/2 confusion remains unresolved after five distinct interventions; the validation/test gap is real and not fully explained; dataset provenance remains unverified. These limitations are documented throughout this report, not minimized.

---

## Section 24 — Future Work (Phase 2 — NOT YET PERFORMED)

The following are potential directions for a separate, future Phase 2 effort. **None of these have been implemented, tested, or started as part of Phase 1.**

- Systematic hyperparameter optimization (beyond the single learning rate tested).
- Learning-rate scheduling / warm-up strategies.
- Investigation of alternative optimizers.
- Improved or alternative regularization strategies for the class 1/2 problem specifically.
- Targeted architecture improvements (e.g. a two-stage classifier: coarse group prediction, then a specialist head for the confusable class group).
- Deeper investigation into *why* classes 1 and 2 are confused (would require either higher-resolution source data, if available, or additional labeled information this project does not currently have).
- Investigation into whether 64×64 resolution itself is a limiting factor (would require access to higher-resolution source images, which this project does not currently have).
- Investigation into the validation/test gap, if patient/scan-level metadata becomes available.
- Additional data, if the professor or dataset source can provide it.

---

*End of Phase 1 Complete Technical Report. All figures referenced above are available in `results/figures/` (originals, untouched) and copies in `phase1_final/presentation_assets/` (for presentation use only).*
