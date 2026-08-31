# Speaker Notes — Phase 1 Presentation

15 slides. ~1 min/slide average → roughly a 15-minute talk; adjust pacing per your professor's time slot.

---

### Slide 1 — Title
**Shows:** Project title, subtitle, your name.
**Say:** Brief intro — "This is Phase 1 of my dual-degree ML project: building and evaluating custom CNNs on an 11-class image dataset my professor provided."
**Time:** 20-30 sec.
**Likely question:** None yet — this is just orientation.

---

### Slide 2 — Problem & Objective
**Shows:** 64×64 grayscale → CNN → 11 classes pipeline.
**Say:** "We were given a pre-split dataset of 64×64 grayscale images across 11 classes, and asked to build custom CNN architectures — not pretrained models — and evaluate them properly."
**Time:** 30-40 sec.
**Technical point:** Emphasize "custom" — no ResNet/ImageNet backbones anywhere in this project.
**Likely question:** "Why not use a pretrained model?" → Because the task required custom architectures, and pretrained backbones are built for larger natural images — overkill and mismatched for 64×64 grayscale data.

---

### Slide 3 — Dataset
**Shows:** Train/val/test counts, resolution, sample image grid (real images from the dataset).
**Say:** "13,932 training images, 2,452 validation, 8,827 test — all 64×64 grayscale, 11 classes, integer-labeled."
**Time:** 30-40 sec.
**Technical point:** Be upfront that exact dataset provenance/name isn't independently confirmed from any file we have — say this plainly if asked, don't guess a name with false confidence.
**Likely question:** "What dataset is this exactly?" → "The structure matches a known public medical-imaging benchmark's class layout, but we couldn't confirm that from the files themselves, so we're not asserting it as fact."

---

### Slide 4 — Dataset Analysis
**Shows:** Class distribution bar chart.
**Say:** "The classes are imbalanced — about 5.6× between the largest and smallest class. This shaped a lot of our later decisions, especially choosing Macro F1 as our main metric."
**Time:** 40-50 sec.
**Technical point:** Point out class 6 (~25% of data) vs. class 2 (~4.4%).
**Likely question:** "Did you check for data leakage or duplicates?" → "Yes — we hashed every image and confirmed zero duplicates within or across splits."

---

### Slide 5 — Preprocessing
**Shows:** uint8[0-255] → normalize → [0-1] → channel dim → CNN pipeline diagram.
**Say:** "Standard preprocessing: normalize to [0,1], add an explicit channel dimension for Conv2D, and keep labels as plain integers since we used sparse categorical cross-entropy."
**Time:** 25-35 sec.
**Technical point:** Same function applied identically to train/val/test — no leakage risk from differential preprocessing.
**Likely question:** "Why not one-hot encode the labels?" → "Sparse categorical cross-entropy accepts integer labels directly — one-hot would be redundant extra preprocessing."

---

### Slide 6 — Baseline Custom CNN
**Shows:** Model A architecture diagram, 286,219 parameters.
**Say:** "Our baseline: 3 convolutional blocks, increasing filters 16→32→64, then a dense layer and dropout before the final softmax. Deliberately simple — meant to establish a reliable starting point, not to be the final answer."
**Time:** 45-60 sec.
**Technical point:** Note where most parameters live (the Flatten→Dense step — 262K of 286K) — this becomes relevant later when explaining Model B/C's design.
**Likely question:** "Why only 3 blocks?" → "64×64 grayscale images are much simpler than natural photos — a shallow network isn't automatically underpowered here."

---

### Slide 7 — Baseline Performance
**Shows:** Accuracy 89.07%, Macro F1 80.53%, confusion matrix.
**Say:** "Strong overall accuracy, but Macro F1 is noticeably lower — that gap is the first sign something's uneven across classes."
**Time:** 40-50 sec.
**Technical point:** This is exactly why accuracy alone would have been misleading here.
**Likely question:** "Why is Macro F1 so much lower than accuracy?" → Leads directly into Slide 8.

---

### Slide 8 — The Main Bottleneck
**Shows:** Confusion matrix highlighting class 1/2, or the error-analysis image; the "94/95" statistic.
**Say:** "94 of 95 true class-2 validation images were predicted as class 1 — by far the single biggest confusion in the entire matrix. Interestingly, the model wasn't confidently wrong here — its confidence on these specific errors was noticeably lower than on its correct predictions, suggesting genuine difficulty rather than a bug."
**Time:** 60-75 sec — this is a key slide, take your time.
**Technical point:** Don't overclaim the cause — say "consistent with visual similarity," not "proven."
**Likely question:** "Did you check if the images are actually mislabeled?" → "We checked pixel statistics and looked for outliers/blank images — found nothing abnormal. We can't rule out label noise with certainty, but there's no direct evidence of it either."

---

### Slide 9 — Experimental Investigation
**Shows:** Roadmap diagram (Baseline → Weighting → Augmentation → Model B → Model C → Focal Loss).
**Say:** "Once we found this bottleneck, we didn't just guess — we ran six controlled experiments, each changing exactly one thing, to test different hypotheses about why."
**Time:** 30-40 sec.
**Technical point:** Emphasize "controlled" — same data split, same evaluation protocol every time.

---

### Slide 10 — Experiment Results
**Shows:** Macro F1 bar chart across all 6 experiments.
**Say:** "Here's the punchline: every single alternative we tried scored lower than the plain baseline on Macro F1. That's not what we expected going in."
**Time:** 45-60 sec — let this land, it's the most important slide.
**Technical point:** Point out this isn't a chart trick — axis starts near the data range but isn't misleadingly truncated to exaggerate differences.
**Likely question:** "Doesn't that mean the experiments failed?" → "They 'failed' at improving Macro F1, but each one taught us something specific — that's the next slide."

---

### Slide 11 — What Didn't Work & Why
**Shows:** 4-5 short blocks (Class Weighting, Augmentation, Model B, Model C, Focal Loss) with one-line "why."
**Say:** Walk through each briefly — "Class weighting corrects for frequency, but classes 1 and 2 have almost the same number of samples, so it didn't address the real issue. Augmentation likely disrupted positional cues the model relied on. Model B simply overfit. Focal loss focuses on hard examples but still didn't move the needle."
**Time:** 60-90 sec.
**Technical point:** Frame this as "here's what we ruled out," not "here's what we got wrong."

---

### Slide 12 — Final Model
**Shows:** Model A again, selection rationale.
**Say:** "We selected the baseline as our final model — highest Macro F1, fewest parameters, least overfitting. We didn't pick it because it was simple by default; it won on the evidence."
**Time:** 30-40 sec.
**Technical point:** Selection was based on validation Macro F1, not accuracy and not test performance.
**Likely question:** "Why not just accuracy?" → Refer back to Slide 7/8 — accuracy would have hidden the class 2 failure.

---

### Slide 13 — Final Test Evaluation
**Shows:** Test accuracy 73.39%, test Macro F1 68.40%, validation-vs-test comparison chart.
**Say:** "We evaluated on the test set exactly once, after selecting the model — and it's noticeably lower than validation. We're upfront about that gap. Two honest possibilities: comparing six models on one validation set can introduce some optimistic bias, or there may be a structural difference between the validation and test splits we don't have the metadata to check."
**Time:** 60-75 sec.
**Technical point:** Emphasize the test set was touched exactly once — no tuning after seeing this number.
**Likely question:** "Why didn't you fix this?" → "Because tuning based on the test result would defeat the purpose of having a held-out test set — we're reporting it honestly instead."

---

### Slide 14 — Key Findings
**Shows:** 5-6 bullet summary.
**Say:** Quick recap — "We built a working custom CNN pipeline, found a specific bottleneck through error analysis, tested five hypotheses to address it, and none worked — which is itself a real, defensible finding, not a failure of the project."
**Time:** 40-50 sec.

---

### Slide 15 — Conclusion & Future Work
**Shows:** Conclusion statement + future work bullets.
**Say:** "Phase 1 gave us a complete, honestly-evaluated pipeline and a clear, evidence-backed problem to investigate further. That's exactly what motivates Phase 2 — none of the future work items have been started yet."
**Time:** 30-40 sec.
**Technical point:** Be explicit that Phase 2 hasn't begun — don't let the professor think future work is already done.

---

## General Tips
- If asked a question you don't have the artifact-backed answer to, say "that's not something we verified — good direction for Phase 2" rather than guessing.
- Lead with the finding, not the metric, on Slide 10 — the "everything we tried made it worse" result is more memorable and more scientifically interesting than any single number.
- Don't apologize for the negative results — frame them as evidence, which is exactly what they are.
