# Custom CNN — 11-Class 64×64 CT-Style Image Classification

A two-phase academic project: building and rigorously evaluating custom CNN architectures (no pretrained backbones) on an 11-class, imbalanced, 64×64 grayscale medical-style image dataset, provided by Prof. Jayaganthan R (IIT Madras).

**Dataset provenance note:** the class structure resembles a known public CT-organ benchmark, but no metadata file in the original dataset confirms this. This project reports only verified, directly-checked properties of the data throughout, and does not assert a dataset name as fact.

## Results

| | Validation | Test (evaluated once) |
|---|---|---|
| Accuracy | 0.891 | 0.734 |
| Macro F1 | 0.805 | 0.684 |

Final model: **Model A**, a lightweight 3-block custom CNN (286,219 parameters) — selected over five other tested configurations (class-weighted loss, augmentation, a deeper network, a regularized network, focal loss) because it had the highest validation Macro F1, the least overfitting, and the fewest parameters.

## Repository structure

```
organ_cnn_project.ipynb        Phase 1 — full pipeline: EDA, preprocessing, baseline,
                                6 controlled experiments, error analysis, model
                                selection, final test evaluation (138 cells)

results/                       Phase 1 figures and experiment logs
phase1_final/                  Phase 1 technical report, quick reference, PPT deck

phase2/                        Phase 2 — continuation
  README.md                    Phase 2 charter and rules
  PHASE1_PROJECT_RECORD.md     Summary of Phase 1 for continuation reference
  notebooks/                   Learning-rate experiments, seed-variance diagnostic,
                                and the hypothesis-driven architecture experiment
  results/, figures/           Phase 2 experiment logs and figures
```

## Key finding

Across 10 controlled experiments spanning both phases, the original baseline has never been beaten with statistical confidence. The project's central discovery is a **seed-driven bistability** in how the model confuses two specific classes (1 and 2) — it reliably commits to one or the other depending on random initialization, rather than being consistently wrong in one direction. This was established through a dedicated multi-seed noise-floor measurement (`phase2/notebooks/phase2_seed_variation_diagnostic.ipynb`), which also caught and debunked a false-positive result from a learning-rate scheduling experiment.

## Reproducing this project

The dataset (`.npy` files) and trained model checkpoints (`.keras` files) are **not included in this repository** (excluded via `.gitignore` — large binaries / provided course material, not for redistribution). To reproduce:

1. Place `train_images.npy`, `train_labels.npy`, `val_images.npy`, `val_labels.npy`, `test_images.npy`, `test_labels.npy` in the repository root.
2. `pip install tensorflow scikit-learn matplotlib numpy scipy jupyter`
3. Run `organ_cnn_project.ipynb` top to bottom (CPU-only; full run takes roughly an hour due to 6 training experiments plus diagnostics).

## Status

Phase 1 complete. Phase 2 diagnostics and one continuation experiment complete (result: inconclusive, see `phase2/notebooks/phase2_continuation_execution.ipynb`). Final test-set re-evaluation of any Phase 2 candidate is intentionally not performed — the test set is reserved for one final, explicitly-approved evaluation only.
