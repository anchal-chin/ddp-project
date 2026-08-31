# Continuation Exemplar Report — Target Benchmark for Intern Evaluation

> **What this document is.** A fully worked, week-by-week exemplar of what a *successful, honest* two-month continuation of the organ-slice CNN project looks like — including one fixed best-result number to use as a comparison benchmark. **Every number, date, and outcome below is illustrative.** No training for this continuation has been run. This is the shape a good outcome takes, constructed so you can hold the intern's *actual* weekly output up against it and see where it matches, exceeds, or falls short — not a record of work already done. Treat it the way you'd treat a model-solution key for an assignment: useful for grading, not a thing to hand in as your own result.

---

## Part A — Why this continuation is logical, not arbitrary

Before the exemplar, the reasoning that makes it defensible:

| Prior finding (real, logged) | What it rules out | What it leaves open |
|---|---|---|
| Class weighting: Macro F1 0.805→0.783 | Frequency-based fixes | — |
| Augmentation: 0.805→0.776 | Generic spatial augmentation | — |
| Deeper CNN (Model B): 0.805→0.792, overfit | Capacity alone | — |
| Regularized CNN (Model C): 0.805→0.780 | Regularization alone | — |
| Focal loss: 0.805→0.793 | Difficulty-reweighted loss | — |
| ReduceLROnPlateau: 0.816, but pre-dated the LR drop | Learning-rate scheduling as the lever | — |
| 4-seed diagnostic: noise floor = 0.797–0.817 | Trusting any single-run delta | A confirmed, multi-seed, *architecture*-level change, motivated by a specific hypothesis about the confusable pair — the one category never tried |

That right-hand cell is the entire justification for everything below. It isn't chosen because it sounds impressive — it's what's left after seven other doors closed.

---

## Part B — The fixed benchmark number

**Target: Macro F1 = 0.84, confirmed as a mean across 3 seeds.**

Why 0.84 specifically, not a rounder or larger number:
- Baseline mean: 0.805. Measured noise ceiling: 0.817. **0.84 clears the noise band by +0.023** — a margin large enough that, if genuinely confirmed across seeds, it cannot reasonably be explained by the same chance mechanism that produced the ReduceLROnPlateau false positive.
- It does **not** require solving the confusable-pair problem outright — a partial, well-characterized improvement is enough to reach it, which is what "not extraordinary" should mean here.
- It is close enough to the top of the previously-scoped 0.83–0.86 range to count as a strong outcome without requiring the top of that range, which is closer to "large breakthrough" territory (~12% likelihood, per the earlier feasibility assessment).

**If the intern's real result differs from 0.84 — this is expected, not a failure signal by itself.** Use the reading table in Part F, not just distance-from-0.84, to judge it.

---

## Part C — Week-by-week exemplar (illustrative narrative)

### Week 1 — Reproduction
Retrained baseline Model A once, unmodified, seed 42. Result: Macro F1 0.806, inside the documented 0.797–0.817 band — confirms the pipeline and environment are correctly understood before anything new is attempted. No changes proposed yet.

### Week 2 — Error analysis and hypothesis formation
Pulled every validation image involved in a confusable-pair error (both directions) and inspected them directly rather than relying only on aggregate confusion counts. Cross-referenced against the pixel-statistics finding already on record (the two classes have nearly identical mean intensity: 185.1 vs. 183.1) and the confidence finding (model confidence on these errors ≈0.57, far below its 0.94 when correct — genuine uncertainty, not overconfidence).

**Hypothesis formed:** the discriminating detail between the two classes is fine-grained and concentrated near object boundaries; the baseline's three successive 2×2 max-pools (64→32→16→8) may be discarding exactly that detail before the network ever gets to use it, more aggressively than for the other nine classes, which appear to be separable on coarser, whole-image statistics.

*(This is a genuinely reasonable CNN hypothesis, not just a plausible-sounding label — it's exactly the kind of one-line, falsifiable claim Week 2 is supposed to produce, and it's the kind of thing that would actually need to come from the intern's own inspection of the real images, not be assumed in advance.)*

### Weeks 3–4 — Designed experiment: "Model D" (detail-preserving variant)
**Single change:** one additional 3×3 convolution inserted before the *first* max-pool only (so the network extracts one more layer of fine detail at full 64×64 resolution before any downsampling), everything else identical to baseline Model A. ~311,000 parameters (a modest increase from 286,219 — not a capacity jump in Model B's league).

Trained under 3 seeds (42, 7, 123) from the outset, identical protocol to the Phase 2 seed diagnostic.

**Illustrative result:** mean Macro F1 = 0.829, std = 0.012 across the 3 seeds. Above baseline, but not yet clearing the target with confidence — the range (0.817–0.841) still brushes the top of the noise band. Logged as a partial positive, not yet a confirmed win.

### Week 5 — One iteration
Given Week 3–4's partial result, one focused refinement (not a new hypothesis): added mild L2 regularization (λ=1e-4) to the new conv layer only, informed directly by Model C's earlier finding that L2 improved training stability. Re-trained under the same 3 seeds.

**Illustrative result:** mean Macro F1 = 0.841, std = 0.009. Now clearly above the noise ceiling (0.817) by a margin (+0.024) larger than the run-to-run std itself — the threshold this project treats as a believable signal rather than noise. **This is the fixed benchmark value referenced in Part B.**

Per-class detail (illustrative): class 1 F1 moved from 0.684 → 0.71; class 2 F1 moved from 0.021 → 0.31. Still far from solved for class 2, but a real, multi-seed-confirmed movement — not a collapse-to-zero or a flip to the other class, which is what several Phase 1/2 attempts produced instead.

### Week 6 — Confirmation
Full metric suite (accuracy, macro precision/recall/F1, per-class F1, confusion matrix) computed once more on the Week 5 configuration to confirm reproducibility of the process itself, and compared directly against both the Phase 1 baseline and the measured noise band in a single table for the report. No test-set access at any point.

### Week 7 — Documentation
Experiment log updated with every run from Weeks 1–6, including the Week 3–4 partial result (not hidden, since it's real supporting evidence for the Week 5 refinement's motivation). Written comparison report drafted in the same tone as the existing Phase 1/2 reports — stating plainly that class 2 remains hard, not overselling the 0.31 as "solved."

### Week 8 — Presentation and the closing action
Slide deck built, dry run completed. **With supervisor approval**, the Week-5 model is evaluated exactly once on the untouched test set — the same test set held back since the original 73.4%/0.684 baseline result.

*(This is the one place this exemplar deliberately does not manufacture a fake precise number. Phase 1's own baseline showed roughly a 0.12 Macro-F1 drop from validation to test — 0.805 down to 0.684. If a similar proportional gap held here, a validation Macro F1 of 0.841 would plausibly land somewhere in the low-to-mid 0.70s on test. That is a rough, explicitly speculative extrapolation from one prior data point, not a forecast to hand the intern in advance — the real test number, whatever it is, is what actually closes the project, and it should surprise no one if it differs from this extrapolation by a meaningful margin.)*

---

## Part D — What this exemplar deliberately does NOT do

- It does not claim class 2 is solved. F1 0.31 is a real improvement over 0.021, not a resolved problem.
- It does not skip the failed intermediate step (Weeks 3–4's partial result) — a good real report won't have a clean, uninterrupted rise either.
- It does not invent a precise test-set number — that's flagged explicitly as speculation, because asserting one would be exactly the kind of unconfirmed claim this project's own history (the ReduceLROnPlateau incident) warns against.
- It does not make the architecture change exotic. One added conv layer, one regularization tweak — small, explainable, and traceable to a specific hypothesis, which is the actual bar, not novelty for its own sake.

---

## Part E — Comparison table format (use this structure for the real report)

| Metric | Baseline (0.805) | Noise band | Intern's actual result | Verdict |
|---|---|---|---|---|
| Macro F1 | 0.805 | 0.797–0.817 | *(fill in)* | *(fill in)* |
| Accuracy | 0.891 | — | *(fill in)* | — |
| Class 1 F1 | 0.684 | — | *(fill in)* | — |
| Class 2 F1 | 0.021 | — | *(fill in)* | — |
| Confirmed across seeds? | n/a | n/a | *(yes/no)* | pass/fail gate |

---

## Part F — How to grade the real outcome against this exemplar

| What the intern actually brings back | Compared to the 0.84 exemplar | Verdict |
|---|---|---|
| ≥0.84, multi-seed confirmed, clear margin over noise | Met or exceeded | **Strong** — matches or beats the worked exemplar |
| 0.817–0.84, multi-seed confirmed | Below target, above noise | **Solid** — real signal, smaller than the exemplar's, still legitimate |
| 0.797–0.817, or unconfirmed | Inside noise band or unverified | **Inconclusive** — not a failure, but not yet a result either; the process matters more here than the number |
| A rigorous negative result with a ruled-out hypothesis | No number to compare | **Acceptable** — this project's own history (7 of 8 prior attempts) shows this is a normal, valid outcome |
| Any "improvement" from a single uncounted run | N/A | **Reject the claim, not the student** — send back for multi-seed confirmation; this is the one hard methodological line |

---

*This document supersedes no prior file — `phase2/STUDENT_HANDOFF_PLAN.md` and `phase2/FEASIBILITY_ASSESSMENT.md` remain the operative planning documents. This report exists specifically to give you one concrete, fully-worked exemplar to hold the intern's real weekly output against.*
