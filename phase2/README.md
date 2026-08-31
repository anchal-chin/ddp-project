# Phase 2 — Custom CNN Optimization (Independent Extension)

## Relationship to Phase 1

**Phase 1** was the original assigned project: a professor-provided 64×64 grayscale, 11-class image dataset, custom CNN development, six controlled experiments, error analysis, model selection, and a single final test evaluation. Phase 1 is **complete and frozen** — see `phase2/PHASE1_PROJECT_RECORD.md` for the full summary and `phase2/PHASE1_FILE_MANIFEST.md` for the exact list of protected artifacts. Nothing in Phase 1 is modified, moved, or deleted by Phase 2 work.

**Phase 2** is a new, independent extension undertaken after Phase 1's completion, to investigate whether the existing custom CNN can be improved further.

## Phase 2 Research Question

> Can systematic optimization of the existing custom CNN improve validation Macro F1 and generalization, without unnecessarily increasing model complexity?

This is **not** "make accuracy as high as possible." An experiment that raises accuracy while lowering Macro F1 is **not** a success (see Phase 1's own Model B/focal-loss results, which raised or matched accuracy while lowering Macro F1 — exactly the failure mode Phase 2 must avoid repeating uncritically).

## Metrics

- **Primary:** validation Macro F1
- **Secondary:** accuracy, macro precision/recall, per-class F1 (especially class 1 and class 2), confusion matrix, validation loss, train/validation gap (overfitting), parameter count, and — where feasible — some sense of whether an observed difference exceeds Phase 1's documented ~0.01–0.02 run-to-run reproducibility margin before it's called a real improvement.

## Test Set Rule

The Phase 1 test result (**Accuracy ≈ 73.39%, Macro F1 ≈ 0.6840**) is a **historical benchmark only**. Phase 2 rules, without exception:
- The test set is **not** used for any Phase 2 training, tuning, model selection, or comparison.
- The test set is **not** re-evaluated repeatedly.
- No Phase 2 model is selected, adjusted, or abandoned based on the Phase 1 test number.
- If a final Phase 2 test evaluation is ever desired, **work stops and explicit approval is requested first** — exactly as it was in Phase 1.

All Phase 2 model development uses **training + validation data only**.

## What Has Already Been Tried (Phase 1 — do not blindly repeat)

| Approach | Result | Why it likely failed |
|---|---|---|
| Class weighting (`balanced`, from train labels) | Macro F1 0.8053 → 0.7833 (negative) | Corrects frequency, not visual similarity; classes 1/2 have almost equal counts |
| Augmentation (rotation/translation/zoom, no flip) | Macro F1 → 0.7763 (negative, worst) | Possibly disrupted positional cues the model relies on |
| Model B (4 blocks ×2 conv, BN, GAP, no L2, 602K params) | Macro F1 → 0.7918 (negative) | Overfit — capacity without matching regularization |
| Model C (4 blocks ×1 conv, BN, L2, GAP, 250K params) | Macro F1 → 0.7796 (negative) | Improved training stability somewhat, still below baseline |
| Focal loss (γ=2.0, no α) | Macro F1 → 0.7925 (negative) | Difficulty-aware reweighting also did not resolve class 1/2 |

Any Phase 2 proposal that resembles one of these must explain concretely how it differs and why the new evidence justifies trying it — not simply repeat it with minor tweaks.

## Decision Boundaries — What Requires Explicit Approval Before Proceeding

Per the project owner's explicit instruction, the following require asking first, not just proceeding:
- Changing the dataset, train/validation split, removing/relabeling samples
- Using test data for anything
- Horizontal flipping (semantic safety not yet established for this data)
- A substantially different loss function or sampling strategy
- A substantially different (new) architecture
- Running more than 3–5 experiments in a batch
- Expensive confirmation reruns
- Any major change to project methodology

Straightforward, low-risk execution of an already-approved experiment does not require re-asking at every step.

## Experiment Design Rule

Every Phase 2 experiment must state, before running: hypothesis, single primary change, variables held constant, expected reason for improvement, success criterion, failure criterion — then, after running: actual measured result and conclusion. See `phase2/PHASE2_EXPERIMENT_LOG.md` (created once the first experiment is approved and run).

## File Safety

Every Phase 2 experiment produces **new** artifacts, named to avoid any collision with Phase 1:
- `phase2/models/modelA_<change>.keras`
- `phase2/results/`, `phase2/figures/`, `phase2/logs/` — Phase 2 only
- Phase 1's `models/`, `results/`, `organ_cnn_project.ipynb`, and `phase1_final/` are never written to.

## Status

**Phase 2 has not yet started any training.** This README, `PHASE1_PROJECT_RECORD.md`, and `PHASE1_FILE_MANIFEST.md` constitute the documentation-and-freeze step requested before any Phase 2 experiment. The first proposed experiments are presented separately (see the accompanying chat message) and await explicit approval before any training begins.
