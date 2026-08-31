# Two-Month Feasibility Assessment — Will the Incoming Student Achieve a Good Result?

## Direct answer

**Yes, almost certainly a presentable result — but likely not a big score improvement.** The project's own history (8 configurations tested across Phase 1 and Phase 2, all either negative or statistically indistinguishable from noise) shows this baseline is unusually hard to beat outright. The realistic, honest target is a rigorously confirmed small improvement, or a well-characterized negative result — and given this project's established culture, both count as a genuine, gradeable outcome.

## Calibrated likelihood by outcome (judgment estimate, anchored to real numbers)

| Outcome | Likelihood |
|---|---|
| Presentable outcome (rigorous process, any number) | ~80% |
| Confirmed modest gain (0.83–0.86 Macro F1, multi-seed) | ~35% |
| Clean negative result (well-documented, hypothesis ruled out) | ~45% |
| Large breakthrough (>0.86, class 1/2 meaningfully resolved) | ~12% |

The middle two rows aren't separate attempts — they're the two most likely shapes the single 8-week effort takes, together accounting for most of the 80% figure.

## Why the bar sits where it does (every row below is a real, logged result)

| Attempt | Phase | Macro F1 | Outcome |
|---|---|---|---|
| Baseline Model A (286K params) | 1 | 0.805 | selected |
| Class-weighted loss | 1 | 0.783 | negative |
| Augmentation | 1 | 0.776 | negative |
| Model B — deeper, 602K params | 1 | 0.792 | negative |
| Model C — regularized, 251K params | 1 | 0.780 | negative |
| Focal loss (γ=2.0) | 1 | 0.793 | negative |
| ReduceLROnPlateau | 2 | 0.816 | within noise |
| Constant lr = 5e-4 | 2 | 0.801 | negative |

**Measured noise band:** four same-configuration runs (1 existing + 3 new diagnostic seeds) produced Macro F1 between **0.797 and 0.817**. Anything the student reports inside that band, from a single run, is indistinguishable from chance.

**Realistic 8-week target:** 0.83–0.86, confirmed as a mean across 3 seeds.

## What determines which way it goes

- **Raises the odds:** a specific, falsifiable hypothesis for the class 1/2 confusion before touching the architecture — not a guessed tweak.
- **Raises the odds:** multi-seed training baked in from the first experiment, not bolted on after a promising single run.
- **Time risk:** CPU-only training — a 3-seed confirmation costs 3× the wall-clock of one run. Budget for this explicitly in weeks 3–6.
- **Lowers the odds:** treating a single-run delta inside the 0.797–0.817 band as real — the project has direct proof this happens (the ReduceLROnPlateau false positive).

## Eight-week shape (full detail in `phase2/STUDENT_HANDOFF_PLAN.md`)

1. **Wk 1** — reproduce baseline once, confirm understanding of the pipeline.
2. **Wk 2** — deep error analysis on class 1/2 → one specific, falsifiable hypothesis.
3. **Wk 3–4** — one designed architecture experiment, 3 seeds from the start.
4. **Wk 5** — one iteration or one pivot, still multi-seed.
5. **Wk 6** — final confirmation run, full comparison vs. baseline and noise band.
6. **Wk 7–8** — documentation, slides, dry run, buffer.

## How to read his result when it lands

| What comes back | Read it as |
|---|---|
| ≥0.86, multi-seed confirmed | Excellent — worth extending or escalating. |
| 0.83–0.86, multi-seed confirmed | Target met — solid, presentable mid-review result. |
| 0.805–0.83, confirmed — or a rigorous negative result | Acceptable — judge the rigor, not just the number. |
| Any "improvement," single run only | Not acceptable regardless of the number — the one hard line. |

*Numbers sourced from `results/metrics/experiment_log.csv` and `phase2/results/`.*
