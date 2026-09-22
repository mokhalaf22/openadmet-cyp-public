# Data findings

Exploratory analysis of the pinned dataset
(`openadmet/cyp-challenge-train-test` @ `3ac9c5db`). The confirmed conventions
that modelling code must follow live in `CLAUDE.md` ("Confirmed data
conventions"); this file is the evidence behind them — tables, plots, and noise
estimates.

Isoforms: CYP1A2, CYP2C9, CYP2D6, CYP3A4. "Arm" = `direct_inhibition` (no
preincubation) vs `TDI_condition` (+NADPH preincubation). Analysis source is
`cyp-challenge-TRAIN_TDI.csv` unless noted; `cyp-challenge-TRAIN_inhibition.csv`
is a redundant subset (see below).

## 1. Censoring is already encoded in the credible intervals

The reported `_conf_low` / `_conf_high` bounds carry the left-censoring signal,
so the `(-inf, 4.0]` special case is unnecessary — consume the bounds verbatim.

| isoform | n | conf_low floor | % at floor (~1.03) | median width | **% width > 2** |
|---|---|---|---|---|---|
| CYP1A2 | 1412 | 1.030 | 4.0% | 0.33 | 8.4% |
| CYP2C9 | 1285 | 1.049 | 1.9% | 0.53 | 6.6% |
| CYP2D6 | 1493 | 1.030 | 4.8% | 0.27 | 8.6% |
| CYP3A4 | 2335 | 1.029 | 10.9% | 0.38 | **23.6%** |

- **Interval width is bimodal**: a dominant narrow mode at ~0.2–0.5 plus a
  distinct second cluster at ~2.0–2.8. The second cluster is the censored
  population. CYP3A4 has by far the most censoring (23.6% of rows wider than 2
  log units), consistent with it having the most weak/inactive compounds.
- **`conf_low` piles up at a finite floor ~1.03** (not −inf, not 4.0). A
  censored compound is `[~1.03, conf_high]` with a wide interval.

![Q1 censoring](figures/q1_censoring.png)

## 2. `TRAIN_inhibition` is a redundant strict subset of `TRAIN_TDI`

Use `TRAIN_TDI` as the single training source and ignore `TRAIN_inhibition`.

- Name sets: `in TDI not in inhibition = 1240`; `in inhibition not in TDI = 0`.
  TRAIN_inhibition ⊂ TRAIN_TDI.
- For every shared compound and every isoform the direct pIC50 is
  **byte-identical**: Pearson = 1.0000, MAE = 0.0000, max abs diff = 0.0000.
  Same assay campaign, not a re-run.
- The 1,240 TDI-only compounds add **0** direct-arm labels — 1,238 are CYP3A4
  `TDI_condition`-only measurements. Pooling does not buy direct-arm compounds.

![Q2 overlap](figures/q2_overlap.png)

## 3. Preincubation shift supports `delta >= 0` for the scored isoforms

Shift = `TDI_condition - direct`, over rows with both arms measured.

| isoform | n | median shift | % negative | % < −0.1 | **% < −0.301** | ~noise on diff |
|---|---|---|---|---|---|---|
| CYP1A2 | 1412 | +0.024 | 38.5% | 9.5% | 2.5% | 0.12 |
| CYP2C9 | 1285 | +0.007 | 46.8% | 17.7% | 4.6% | 0.19 |
| **CYP2D6** | 1493 | +0.154 | 13.5% | 3.2% | **0.3%** | 0.10 |
| **CYP3A4** | 2334 | +0.307 | 13.3% | 7.0% | **2.6%** | 0.11 |

- The two **scored** TDI isoforms (CYP2D6, CYP3A4) are predominantly positive;
  the `delta >= 0` softplus is a good fit.
- CYP1A2/CYP2C9 (not scored for TDI) center on ≈0. Their negatives are frequent
  but small — comparable to the propagated measurement noise (~0.12 / ~0.19,
  from the reported `_std` columns via √(σ_direct² + σ_tdi²)) — and only 2.5% /
  4.6% exceed −0.301. The constraint is harmless there.

![Q3 shift](figures/q3_shift.png)

## Noise estimates

**Shift-noise estimator (used throughout this document).** For each row,
propagate the two reported per-measurement stds into the noise on the shift,
`σ_shift = √(σ_direct² + σ_TDI²)`, then take the **median over the specified
row set**. The "shift noise" column below is that median over all both-arms
rows. (An earlier note quoted ~0.10 / ~0.11 for CYP2D6/CYP3A4 from
`hypot(median σ_direct, median σ_TDI)`; that hypot-of-medians shortcut is
dropped in favour of this per-row-then-median derivation, which is what §5 also
uses.)

| isoform | median direct `_std` | median TDI `_std` | shift noise (all both-arms rows) |
|---|---|---|---|
| CYP1A2 | 0.084 | 0.088 | 0.121 |
| CYP2C9 | 0.137 | 0.133 | 0.194 |
| CYP2D6 | 0.069 | 0.065 | 0.098 |
| CYP3A4 | 0.095 | 0.061 | 0.123 |

## 4. Empirical LOQ, and why the CYP3A4 "direct-less" rows are not censored

### Empirical LOQ ≈ 2.0, with no hard edge

Lower tail of the observed direct pIC50 point estimates (rows that have one):

| isoform | n | min | p1 | p5 | % within 0.5 of min |
|---|---|---|---|---|---|
| CYP1A2 | 1412 | 1.906 | 2.000 | 2.679 | 4.0% |
| CYP2C9 | 1285 | 2.098 | 2.249 | 3.248 | 2.2% |
| CYP2D6 | 1493 | 1.947 | 2.024 | 2.617 | 4.4% |
| CYP3A4 | 2335 | 1.909 | 1.964 | 2.094 | 12.1% |

There is **no hard left edge at 4.0** — nor a spike piling on any single floor
value. The point estimates simply thin out around pIC50 ≈ 1.9–2.1 (pooled
p1 ≈ 2.0). The censoring signal lives in `conf_low` (which floors near 1.03,
see §1), not in the point estimate. So the assumed `4.0` limit of quantitation
is wrong; the empirical LOQ of the fitted point is ~2.0.

### The direct arm was never assayed for 1,249 CYP3A4 rows

CYP3A4 has 1,249 rows with a measured `TDI_condition` pIC50 but no direct pIC50
(≈35% of its 3,584 TDI labels). These are **not left-censored weak compounds**:

- **Every direct-side column is 100% NaN** for these rows — `pIC50`,
  `conf_low`, `conf_high`, `std`. Not a low fit or a wide fit: no fit at all.
- **Emax-file evidence:** all 1,249 appear in `TRAIN_Emax.csv` with a
  TDI-condition Emax measured and **zero** direct-inhibition Emax. The direct
  arm was never run.
- Their TDI-condition potency **skews higher** than both-arms rows (median
  **5.40 vs 4.63**): 88.5% exceed pIC50 4.0, 66.8% exceed 5.0, and 283 rows
  (22.7%) exceed 6.0 (1 µM — genuinely potent). Examples, all `is_TDI=False`:
  `OCNT-0454261` (7.27), `OCNT-0454289` (6.89), `OCNT-0454217` (6.78).
- **All 1,249 are `is_TDI = False`** — an assigned default, since no direct arm
  means no measurable shift. It carries no experimental information about `mu`.

![Q4 direct-less](figures/q2b_directless.png)

Consequence: never supervise `mu` on these rows (a `(-inf, LOQ]` target would
fabricate a large shift on potent compounds), and never let them into TDI
classification training or internal evaluation.

### Organizer labeling taxonomy (challenge Space FAQ)

The Space FAQ (`config.py`) confirms `is_TDI` is a deterministic function of the
two arms relative to the pIC50 = 4 reliable lower limit:

- **Positive** — direct pIC50 > 4 and shift > 0.301 (2-fold).
- **Negative** — direct pIC50 > 4 and shift ≤ 0.301.
- **Inferred positive** — direct pIC50 < 4 but TDI-arm pIC50 > 4.301.
- **Assigned negative** — direct pIC50 < 4 *and* TDI-arm pIC50 < 4 ("labeled
  negative by convention").

Scored positive = positives + inferred positives; scored negative = negatives +
assigned negatives. The FAQ adds that on the blinded test, predictions are
requested for every compound, but "only compounds whose label can be assigned
with confidence contribute to the score." Note the organizers' formal *assigned
negative* (both arms measured and < 4) is a **different** population from the
training-only *direct-arm-never-assayed* rows above.

## 5. CYP2D6 TDI: a partial label-noise / boundary ceiling

Recorded **before** attempting to beat it, so any later gain has an honest
baseline. The LightGBM baseline reaches OOF MCC ≈ 0.35 on CYP3A4 but only ≈ 0.10
on CYP2D6. This section asks whether that gap is intrinsic (label noise near the
2-fold boundary) rather than purely a featurization problem.

**Folds are not the cause.** Positives are well spread across the baseline
scaffold folds for both isoforms:

| fold | CYP2D6 n / pos / rate | CYP3A4 n / pos / rate |
|---|---|---|
| 0 | 299 / 72 / 24.1% | 467 / 147 / 31.5% |
| 1 | 299 / 62 / 20.7% | 467 / 161 / 34.5% |
| 2 | 299 / 68 / 22.7% | 467 / 157 / 33.6% |
| 3 | 298 / 65 / 21.8% | 467 / 159 / 34.0% |
| 4 | 298 / 57 / 19.1% | 466 / 140 / 30.0% |

**2D6 positives sit closer to the 2-fold boundary and are measured more
noisily.** The distance-to-flip is `δ − 0.301` for shift-positives (direct ≥ 4)
and `TDI_arm − 4.301` for inferred positives (direct < 4). The shift-noise here
is the same `√(σ_direct²+σ_TDI²)` estimator as §Noise estimates, restricted to
each isoform's **true positives** (the population this question is about); the
all-rows values in §Noise estimates differ because CYP3A4's negatives are
noisier than its positives.

| metric | CYP2D6 | CYP3A4 |
|---|---|---|
| true positives (shift / inferred) | 324 (273 / 51) | 764 (680 / 84) |
| base rate | 21.7% | 32.7% |
| shift-positives with δ in (0.301, 0.401] | 32.2% | 26.2% |
| median margin-to-flip | 0.169 | 0.210 |
| p25 margin-to-flip | 0.069 | 0.095 |
| shift noise among positives | 0.126 | 0.075 |
| positives within 1× noise of flipping | 38.6% | 19.2% |
| positives within 2× noise of flipping | 61.4% | 38.0% |

![Q5 boundary](figures/q5_boundary.png)

**Honest framing.** CYP2D6 faces a label-noise / boundary ceiling that CYP3A4
largely does not: its positives are both nearer the 2-fold cutoff and measured
with roughly 1.7× the noise, so ~60% of them sit within 2σ of flipping to
negative versus ~38% for CYP3A4 — a large share are effectively coin-flips at
the label boundary even with perfect features. This is compounded by CYP2D6
having less than half as many positives (324 vs 764) at a lower base rate. It is
**not an absolute ceiling**: CYP3A4 reaches MCC 0.35 with its own boundary
cases, and CYP2D6's small positive set leaves room for better features or more
data to help. (Caveat on the estimator: measured over *all* both-arms rows
instead of positives, the two isoforms' within-2σ fractions are comparable
(~54% vs ~57%); the contrast above is specifically a property of the positive
sets, which is the relevant population for a label-noise argument.)

## 6. Baseline submission — OOF vs blind leaderboard

To test whether scaffold-split OOF tracks the blind leaderboard while there is
still time to correct course, the reference baseline — blinded predictions
**generated from commit `5d5dbdd`** via `make baseline && make submit` — is
**submitted to the interim leaderboard**, so the OOF-vs-blind comparison on
2026-09-25 is anchored to one specific model. The upload itself is an
interactive Space form tied to a HuggingFace account and public disclosure
checkboxes (open-source code + report link, proprietary-data flag), so it is
performed by a maintainer, not automated; the validated files and the numbers to
compare are recorded here.

- **Submission portal:** open. The challenge is a single continuous stage;
  submissions run **2026-08-17 → 2026-11-03 (23:59 UTC)**. The **intermediate
  leaderboard deadline is 2026-09-24 (23:59 UTC)** and the interim leaderboard
  (a one-time full-test-set performance reveal) is released **2026-09-25**.
- **Submission code commit:** `5d5dbdd` (files produced by `make baseline &&
  make submit`: `submissions/regression.parquet`,
  `submissions/classification.parquet`).
- **Local scaffold-split OOF (the numbers being compared):**

| isoform | OOF ST-RAE | OOF MAE | OOF MCC (TDI) |
|---|---|---|---|
| CYP1A2 | 0.526 ± 0.017 | 0.652 ± 0.013 | — |
| CYP2C9 | 0.364 ± 0.019 | 0.489 ± 0.017 | — |
| CYP2D6 | 0.617 ± 0.012 | 0.600 ± 0.044 | 0.097 ± 0.045 |
| CYP3A4 | 0.298 ± 0.019 | 0.570 ± 0.023 | 0.347 ± 0.059 |

- **Interim blind leaderboard result:** _to be filled from the 2026-09-25
  reveal._ If the blind ST-RAE is materially worse than OOF — especially on the
  isoforms where the baseline predictions are narrowest (see the mean-regression
  check) — scaffold OOF is optimistic on this hit-expansion test set and the
  training/eval protocol needs revisiting before 2026-11-03.

## Reproduce

Numbers and plots regenerated from `data/` (pinned revision) by the EDA scripts
and `cyp.baseline`. Everything above is derived solely from the public challenge
dataset.
