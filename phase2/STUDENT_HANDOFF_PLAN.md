# Phase 2 Continuation — 8-Week Plan for Incoming Student

**Purpose of this document:** a handoff plan for a junior student continuing this project for 1-2 months, ending in a mid-review-presentable milestone. This document sets *targets and expectations for planning purposes* — it is not a record of work already done. The one benchmark number in Section 3 is a **calibration target**, derived honestly from the project's own prior evidence, not a claimed result. Compare the student's *actual* numbers against it at the end to judge progress — do not present this document's target as if it were achieved.

---

## 1. Where the Project Actually Stands (what the student inherits)

- **Phase 1 (complete, frozen):** custom 3-block CNN ("Model A," 286,219 params) selected as the best of 6 tested configurations. Validation Macro F1 = 0.805, validation accuracy = 89.1%. Final test evaluation (done once): 73.4% accuracy, 0.684 Macro F1. Dominant unresolved issue: classes 1 and 2 are confused with each other far more than any other class pair.
- **Phase 2 so far (Experiments 7-9, complete):** tested `ReduceLROnPlateau` and a lower constant learning rate — both negative or statistically indistinguishable from noise. Ran a 4-seed diagnostic and found the baseline's own natural run-to-run Macro F1 variation is **±0.02** (std ≈ 0.009, range ≈ 0.020). This is the single most important number the new student needs to internalize: **any claimed improvement smaller than ~0.02 Macro F1 is not trustworthy without multi-seed confirmation.**
- **What has NOT been tried:** a carefully designed (not just bigger) custom architecture change targeting the class 1/2 confusion specifically; any form of targeted/two-stage modeling; deeper visual/statistical inspection of the class 1/2 pair beyond what Phase 2 already did.

This means the student is not starting from zero — they're starting from a project that has already ruled out several plausible directions. That's valuable context, not a discouraging sign, and it should shape what they attempt next (see Section 4).

---

## 2. The Story to Set Up For Them

Frame it to the student exactly like this, so they understand *why* the plan looks the way it does:

> "Six experiments and a statistical noise-floor study already ruled out the easy wins — class weighting, augmentation, more capacity, a learning-rate schedule. That's not a dead end, it's a map: it tells you where *not* to spend your two months. Your job is to take the one genuinely untested, evidence-motivated direction — a properly designed architecture change aimed specifically at the class 1/2 problem — and either confirm it helps (with statistical rigor, not a single lucky run) or rule it out just as cleanly as everything before it. A clean negative result, well documented, is a completely acceptable outcome here. A rushed, unconfirmed 'positive' result is not."

This keeps the student's target modest and achievable while preserving the project's actual scientific culture (documented in Phase 1/2: never trust an unconfirmed delta, never hide a negative result).

---

## 3. The One Benchmark Target (planning calibration only)

**Target: validation Macro F1 in the range 0.83-0.86, confirmed as a mean across 3 seeds (not a single run), by end of Week 6.**

Why this specific range, not a bigger or smaller one:
- It sits **meaningfully above** the ±0.02 noise floor established in Phase 2 (0.805 + 0.02 ≈ 0.825 is the minimum that would even be distinguishable from noise; 0.83-0.86 gives real headroom above that).
- It is **not extraordinary** — it does not require solving the class 1/2 problem outright, just improving overall balance enough to move the average meaningfully.
- It is grounded in what similar architecture refinements achieved elsewhere in this project: Model C (regularized, moderate-capacity) came within ~0.01-0.03 of baseline despite being a first attempt with no iteration; a second, more deliberate attempt with a full 4-6 weeks of iteration room is a reasonable ask to close and modestly exceed that gap.

**How to use this number when reviewing the student's actual work:**
- **≥0.86, confirmed across seeds:** excellent — exceeded expectations, worth extending their involvement or escalating the finding.
- **0.83-0.86, confirmed across seeds:** target met — solid, presentable mid-review result.
- **0.805-0.83, confirmed across seeds, OR a clean negative result with strong error-analysis insight:** acceptable — this project's own history shows negative results are common and valuable here; judge the *rigor*, not just the number.
- **Any "improvement" reported from a single run, not confirmed across seeds:** not acceptable regardless of the number — this is the one methodological red line, since Phase 2 already proved single-run deltas in this exact range are frequently just noise.

---

## 4. Week-by-Week Plan (8 weeks)

### Week 1 — Onboarding and Reproduction
- Read `phase1_final/PHASE1_COMPLETE_TECHNICAL_REPORT.md`, `phase2/PHASE1_PROJECT_RECORD.md`, `phase2/README.md`, and the Phase 2 experiment/diagnostic notebooks.
- Set up the environment, retrain baseline Model A **once** to confirm they can reproduce a Macro F1 in the 0.79-0.82 range (per the documented noise band) — this is a checkpoint that they understand the pipeline, not a research task.
- Deliverable: a short (1-page) written summary, in their own words, of what's already been tried and why it failed. This forces genuine understanding rather than skimming.

### Week 2 — Deep Error Analysis on Classes 1/2
- Build on Phase 2's confusion/confidence analysis: pull every validation image involved in a class-1/2 error, inspect them directly (not just aggregate stats).
- Check literature/documentation for what classes 1 and 2 are believed to represent (if dataset provenance can be clarified) — even partial confirmation would be valuable context no one has obtained yet.
- Deliverable: an error-analysis note proposing a **specific, falsifiable hypothesis** for why 1/2 are confused (e.g., "the discriminating feature is a thin structure that maxpooling destroys" or similar) — this hypothesis should motivate Week 3-5's architecture choice, not be chosen arbitrarily.

### Week 3-4 — One Architecture Experiment (designed, not guessed)
- Design **one** architecture variant directly motivated by the Week 2 hypothesis (e.g., smaller pooling stride to preserve fine detail, a wider first block, a mild attention/skip mechanism — whatever the hypothesis actually points to).
- Follow the exact experiment-design discipline already established in this project: hypothesis, single change, held-constant variables, success/failure criteria, written *before* training.
- Train under **3 seeds** from the start (not one run followed by "should I check other seeds" — bake it in), same protocol as the Phase 2 seed diagnostic.
- Deliverable: a logged, multi-seed result compared against the 0.805 baseline mean.

### Week 5 — Iterate Once, Based on Evidence
- If Week 3-4 showed promise: one focused follow-up (e.g., a hyperparameter within that architecture), still multi-seed.
- If Week 3-4 was negative: pivot to a second, different hypothesis from Week 2's analysis — do not just re-tune the same failed idea.
- This is the only point in the plan where a second full experiment is scheduled — keep it to one pivot, not an open-ended search.

### Week 6 — Consolidate and Confirm
- Take whatever the best candidate is (even if it's still the Phase 1 baseline) and do a final confirmation run: 3 seeds, full metric suite (accuracy, macro P/R/F1, per-class F1, confusion matrix), compared explicitly against Phase 1 baseline and against the Phase 2 noise band.
- No test-set access at any point in this plan — that decision is reserved for you (the supervisor) after reviewing Week 6's outcome.

### Week 7 — Documentation
- Update the Phase 2 experiment log with every run (including negative ones).
- Write a short comparison report: what was tried, what worked/didn't, honest interpretation — following the exact tone of the existing Phase 1/2 reports (state uncertainty plainly, don't oversell small deltas).

### Week 8 — Mid-Review Prep
- Build a short slide deck (reuse the Phase 1 PPT's visual style/structure as a template — 6-8 slides is enough for a mid-review, not the full 15).
- Dry-run the presentation with you before the actual review.
- Buffer week for slippage — if Weeks 3-5 ran long, this is where it gets absorbed.

---

## 5. What "Presentable" Means Here (so the bar is clear to the student)

A presentable mid-review outcome does **not** require beating the baseline. It requires:
1. One well-motivated architecture hypothesis, clearly tied to actual error analysis (not guessed).
2. Multi-seed confirmation of whatever result they get — positive or negative.
3. Honest comparison against the Phase 1 baseline and the Phase 2 noise floor.
4. A clear, defensible answer to "what would you try next" if asked.

That's a legitimate, gradeable two-month deliverable regardless of whether the number goes up, stays flat, or the class 1/2 problem remains unsolved — which, given the project's own history, is the realistic range of outcomes to expect.
