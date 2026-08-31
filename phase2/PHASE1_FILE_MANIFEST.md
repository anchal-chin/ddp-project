# Phase 1 File Manifest

Complete inventory of Phase 1 project artifacts, their purpose, and their protection status for Phase 2. This is a new Phase 2 file — creating it did not touch any file listed below.

**Rule for all rows marked "MUST remain untouched":** Phase 2 must never write to, rename, move, or delete these paths. Any Phase 2 file that conceptually replaces or extends one of these must be created under `phase2/` with a distinct name (see `phase2/README.md`).

## Dataset (original, provided by professor)

| Path | Purpose | Phase 1 original? | Protection |
|---|---|---|---|
| `train_images.npy` (55M) | Training images | Yes | MUST remain untouched |
| `train_labels.npy` (16K) | Training labels | Yes | MUST remain untouched |
| `val_images.npy` (9.6M) | Validation images | Yes | MUST remain untouched |
| `val_labels.npy` (4.0K) | Validation labels | Yes | MUST remain untouched |
| `test_images.npy` (35M) | Test images | Yes | MUST remain untouched |
| `test_labels.npy` (12K) | Test labels | Yes | MUST remain untouched |

## Notebook

| Path | Purpose | Phase 1 original? | Protection |
|---|---|---|---|
| `organ_cnn_project.ipynb` (3.9M) | Complete Phase 1 notebook — EDA, preprocessing, baseline, all 6 experiments, error analysis, model selection, final test evaluation. 138 cells, executed top-to-bottom, zero errors (independently audited). | Yes | MUST remain untouched |

## Model checkpoints (`models/`)

| Path | Purpose | Phase 1 original? | Protection |
|---|---|---|---|
| `models/model_a_baseline_best.keras` (3.4M) | **Final selected Phase 1 model** (286,219 params) | Yes | MUST remain untouched |
| `models/model_a_weighted_best.keras` (3.4M) | Experiment 2 (class weighting) checkpoint | Yes | MUST remain untouched |
| `models/model_a_augmented_best.keras` (3.4M) | Experiment 3 (augmentation) checkpoint | Yes | MUST remain untouched |
| `models/model_b_best.keras` (7.1M) | Experiment 4 (deeper CNN, 602,347 params) checkpoint | Yes | MUST remain untouched |
| `models/model_c_best.keras` (3.0M) | Experiment 5 (regularized CNN, 250,635 params) checkpoint | Yes | MUST remain untouched |
| `models/model_a_focal_best.keras` (3.4M) | Experiment 6 (focal loss) checkpoint | Yes | MUST remain untouched |

## Results / metrics (`results/`)

| Path | Purpose | Phase 1 original? | Protection |
|---|---|---|---|
| `results/figures/` (24 PNG files) | All EDA, training-curve, confusion-matrix, and error-analysis figures generated during Phase 1 | Yes | MUST remain untouched |
| `results/metrics/experiment_log.csv` | Master log of all 6 Phase 1 experiments — the authoritative numeric source of truth for Phase 1 | Yes | MUST remain untouched |
| `results/metrics/experiment_config.json` | Shared Phase 1 experiment configuration (seed, optimizer, batch size, preprocessing description) | Yes | MUST remain untouched |
| `results/metrics/final_test_results.json` | Final Phase 1 test-set evaluation result | Yes | MUST remain untouched |

## Phase 1 documentation package (`phase1_final/`)

| Path | Purpose | Phase 1 original? | Protection |
|---|---|---|---|
| `phase1_final/PHASE1_COMPLETE_TECHNICAL_REPORT.md` | Full 24-section Phase 1 technical report | Yes (created as Phase 1 documentation) | MUST remain untouched |
| `phase1_final/PHASE1_QUICK_REFERENCE.md` | One-page Phase 1 number reference | Yes | MUST remain untouched |
| `phase1_final/PRESENTATION_SPEAKER_NOTES.md` | Per-slide speaker notes for the Phase 1 PPT | Yes | MUST remain untouched |
| `phase1_final/Phase1_Custom_CNN_Project.pptx` | Phase 1 presentation (15 slides) | Yes | MUST remain untouched |
| `phase1_final/presentation_assets/` | Figures/diagrams used in the Phase 1 PPT (copies of originals + newly generated diagrams, all from Phase 1 data) | Yes | MUST remain untouched |

## Environment

| Path | Purpose | Phase 1 original? | Protection |
|---|---|---|---|
| `.venv/` | Python virtual environment (TensorFlow, scikit-learn, python-pptx, etc.) | Yes (built during Phase 1) | Shared — safe to reuse for Phase 2, do not delete/rebuild destructively |

## Phase 2 (new, created by this task)

| Path | Purpose |
|---|---|
| `phase2/PHASE1_PROJECT_RECORD.md` | This session's summary record of Phase 1 (new file, does not modify Phase 1) |
| `phase2/PHASE1_FILE_MANIFEST.md` | This file |
| `phase2/README.md` | Phase 2 charter and rules |
| `phase2/experiments/` | Phase 2 experiment scripts/notebooks (empty, awaiting approval) |
| `phase2/models/` | Phase 2 model checkpoints (empty, awaiting approval) |
| `phase2/results/` | Phase 2 metrics/logs (empty, awaiting approval) |
| `phase2/figures/` | Phase 2 figures (empty, awaiting approval) |
| `phase2/logs/` | Phase 2 training logs (empty, awaiting approval) |
| `phase2/notebooks/` | Phase 2 notebooks, if used (empty, awaiting approval) |

## Verification performed for this manifest

Directly re-listed and size-checked every path above at the time of writing this document (no training, no modification). Dataset file timestamps confirmed as the original `Jan 1 1980` placeholder timestamps (unchanged since Phase 1 began), notebook and all other paths confirmed present.
