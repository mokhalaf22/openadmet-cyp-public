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

## 7. Narrow baseline predictions are shrinkage, not domain shift

The baseline's predicted pIC50 spans are much narrower than the training
targets. Comparing training targets to *blinded* predictions confounds two
causes (shrinkage vs the test set being far from training); comparing **OOF
predictions to blinded predictions from the same fitted models** separates them.

| isoform | train target (std / IQR) | OOF pred (std / IQR) | blinded pred (std / IQR) |
|---|---|---|---|
| CYP1A2 | 1.03 / 1.00 | 0.36 / 0.45 | 0.40 / 0.58 |
| CYP2C9 | 0.78 / 0.92 | 0.36 / 0.47 | 0.45 / 0.70 |
| CYP2D6 | 0.92 / 0.83 | 0.23 / 0.30 | 0.22 / 0.25 |
| CYP3A4 | 1.09 / 1.50 | 0.71 / 0.88 | 0.72 / 0.88 |

OOF and blinded predictions are equally narrow on every isoform (within ~0.05
std), and both are far narrower than the targets. The model reverts toward the
mean **in-domain**, on held-out training compounds — the blinded set adds
nothing to the narrowness. So this is **shrinkage, not domain shift**; the lever
is regularization / the objective, not domain adaptation.

**Similarity of the blinded set to training (ECFP4 Tanimoto, nearest neighbour).**
Median NN similarity to the full training set is **0.587** (p25 0.543, p75
0.639); every blinded compound has a neighbour above 0.4, but only **10%
(76/750) clear 0.7**. Per-isoform (to each direct-present training subset)
medians are 0.47–0.54 with 3–6% above 0.7.

| target set | median NN Tanimoto | % > 0.7 |
|---|---|---|
| full training (6,145) | 0.587 | 10% |
| CYP1A2 subset | 0.517 | 4% |
| CYP2C9 subset | 0.518 | 3% |
| CYP2D6 subset | 0.471 | 3% |
| CYP3A4 subset | 0.538 | 6% |

So the blinded set is **not** near-duplicate hit expansion off the training
rows — the analogs are genuinely new, sitting in a neighbourhood the model has
seen but not on top of it. This makes scaffold-split OOF a **more credible proxy
for blind performance than originally assumed** (the announcement's "75 potent
parents + nearest chemisimilars" framing implied much tighter overlap). The
report's framing should follow this measurement rather than the announcement's
description.

**Leakage check.** 0 of 750 blinded `Molecule_Name`s appear in training.

**Interpretive note (recorded, to be tested — not assumed).** Prediction width
tracks predictive accuracy across isoforms: CYP3A4 is widest (blinded pred std
0.72) and has the best OOF ST-RAE (0.298); CYP2D6 is narrowest (0.22) and has
the worst (0.617), with CYP2C9 and CYP1A2 ordered in between on both axes
simultaneously. That is exactly how an L1-fitted conditional median behaves when
the features explain limited variance — it shrinks toward the median where it is
uncertain. So shrinkage is **not assumed to be a defect**; it is a hypothesis to
test against the metric (does widening the predictions improve ST-RAE?), not a
bug to fix on sight. See §8.

## 8. Regularization grid: widening predictions does not help the metric

Testing the §7 hypothesis directly — refit the baseline regression with weaker
regularization (`colsample_bytree` 0.5→0.8, trees 500→1500) and read off, per
isoform, the OOF prediction std and OOF ST-RAE (mean ± fold std). Baseline is
`(0.5, 500)`.

| isoform | (colsample, trees) | OOF pred std | OOF ST-RAE (mean ± fold sd) |
|---|---|---|---|
| CYP1A2 | (0.5, 500) *baseline* | 0.359 | 0.526 ± 0.017 |
| CYP1A2 | (0.5, 1500) | 0.393 | 0.524 ± 0.021 |
| CYP1A2 | (0.8, 1500) | 0.398 | 0.529 ± 0.022 |
| CYP2C9 | (0.5, 500) *baseline* | 0.361 | 0.364 ± 0.019 |
| CYP2C9 | (0.5, 1500) | 0.385 | 0.361 ± 0.020 |
| CYP2C9 | (0.8, 1500) | 0.395 | 0.362 ± 0.020 |
| CYP2D6 | (0.5, 500) *baseline* | 0.228 | 0.617 ± 0.012 |
| CYP2D6 | (0.5, 1500) | 0.241 | 0.618 ± 0.014 |
| CYP2D6 | (0.8, 1500) | 0.248 | 0.618 ± 0.010 |
| CYP3A4 | (0.5, 500) *baseline* | 0.712 | 0.298 ± 0.019 |
| CYP3A4 | (0.5, 1500) | 0.743 | 0.295 ± 0.018 |
| CYP3A4 | (0.8, 1500) | 0.752 | 0.296 ± 0.017 |

(The `(0.8, 500)` cell is omitted for brevity; it sits between the rows shown and
changes nothing.)

Weaker regularization does widen the predictions — OOF pred std rises ~0.02–0.04
per isoform — but **ST-RAE is flat**: every setting is within fold-to-fold
variance of the baseline on every isoform. The largest nominal move is CYP3A4
(0.298 → 0.295 at 1500 trees), 0.003 against a ±0.018 fold std, i.e. noise.

**Conclusion.** The shrinkage is not a regularization defect. Consistent with §7,
these are L1 conditional-median predictions shrinking where the features explain
limited variance; forcing them wider neither helps nor hurts the metric. So the
regularization lever is exhausted — the submission is **unchanged** (no ST-RAE
improvement beyond fold variance, per the decision rule). Gains will have to come
from better signal (features/representation) or the interval-hinge two-head
objective, not from de-shrinking a median regressor.

## 9. Interim leaderboard state

Recorded 2026-09-22 from the challenge leaderboard. Metrics are macro-averaged
across the scored isoforms: **MA-ST-RAE** (regression, lower is better) and
**MA-MCC** (classification, higher is better).

| track | leaderboard top | rank 11 | ours |
|---|---|---|---|
| Regression (MA-ST-RAE) | 0.3814 | 0.4404 | 0.451 (local macro OOF) |
| Classification (MA-MCC) | 0.4647 | 0.4091 | 0.222 (baseline macro; CYP3A4 0.347, CYP2D6 0.097) |

Reading: on regression our local macro OOF (0.451) would sit just off the rank-11
cut (0.4404), and the field is tight (top 0.3814). On classification we are far
back — macro 0.222 vs a 0.4091 rank-11 cut — dragged down by CYP2D6 (0.097). Most
leaderboard entries carry **no model-report link**, so the field's methods are
largely undisclosed.

## 10. TDI threshold calibration: a discrimination gap, not a calibration one

The baseline's TDI thresholds are tuned to the OOF-argmax MCC (CYP3A4 0.35,
CYP2D6 0.10). Two facts about that:

**MCC's optimal threshold is prevalence-invariant here.** Reweighting the OOF to
simulate blind positive rates of 15/20/25% barely moves the argmax — CYP3A4 stays
at 0.35 under every balance, CYP2D6 stays at 0.10 (nudging to 0.30 only at 15%,
with identical MCC). MCC already accounts for class balance, so re-tuning the cut
to the blind prevalence recovers no MCC.

**Our submitted positive rates were far above prevalence.** At the tuned cuts the
submission called **46.9%** (CYP2D6) and **40.1%** (CYP3A4) of the 750 blinded
compounds positive, against an estimated blind scored prevalence near **17%**.
That is not a fixable miscalibration: it is what the MCC-optimal thresholds
produce given weak classifiers (the low CYP2D6 cut is "best of a bad lot").

**So the classification gap is a discrimination problem, not a calibration one.**
The lever is model quality — better features / the two-head model — not the
threshold, especially for CYP2D6, which also carries the §5 label-noise ceiling.

*Caveat:* this all assumes the blind set's per-class score distributions resemble
OOF. With no blind labels we cannot check it; §7 argues OOF is a credible proxy.

**Applied change — CYP2D6 threshold 0.10 → 0.30 (variance reduction, not a score
grab).** Simulated MCC is unchanged (0.097 at both cuts), but 0.10 sits at an
extreme of the decision function where the predicted positive count is highly
sensitive to any shift in the blind score distribution; 0.30 gives the same
expected MCC with less variance, and at the ~17% estimated prevalence the sweep
actually favours it. The change drops the CYP2D6 submitted positive rate from
46.9% to **13.2%** (99/750) — in line with prevalence — with no expected MCC cost.
Encoded as `baseline.TDI_THRESHOLD_OVERRIDE`; the classification submission was
regenerated (CYP3A4 unchanged at 0.35 / 40.1%).

## 11. Two-head encoder diagnostic: the fold-variance blow-up was optimization

Before running ablations, the neural two-head model's baseline-closest config
(ecfp + per-isoform + point) was far worse than LightGBM with fold std 3–4× the
baseline's. This section isolates why.

**Feature cleaning.** Dropping degenerate columns before standardizing (rather
than clipping after): `Ipc` (1), near-zero-variance (14), non-finite (0) → 2,250
of 2,265 features kept.

**Four learners, identical cleaned features and folds** (OOF ST-RAE, mean ± fold sd):

| isoform | LightGBM | Ridge | sklearn-MLP | torch-MLP (pre-fix) |
|---|---|---|---|---|
| CYP1A2 | 0.525 ± 0.013 | 0.593 ± 0.025 | 2.038 ± 0.296 | 0.844 ± 0.057 |
| CYP2C9 | 0.367 ± 0.021 | 0.384 ± 0.027 | 2.111 ± 0.146 | 0.660 ± 0.063 |
| CYP2D6 | 0.612 ± 0.009 | 0.690 ± 0.025 | 2.101 ± 0.153 | 0.896 ± 0.073 |
| CYP3A4 | 0.297 ± 0.020 | 0.309 ± 0.020 | 0.945 ± 0.099 | 0.460 ± 0.035 |

**Ridge nearly matches LightGBM** — the direct-pIC50 signal in these features is
largely linear, and a regularized linear model is near-optimal. Both neural nets
underperformed: `sklearn-MLP` catastrophically (worse than predict-a-constant),
the torch MLP by ~0.15–0.35 with inflated fold variance.

**Root causes (all fixed).** The CYP2D6 loss curves showed train weighted-L1
collapsing to ~0 by epoch 50 while validation ST-RAE plateaued high — classic
overfitting, plus two setup bugs:

1. **Unbounded features** (`Ipc` ~1e14, rare bits >70σ) destabilized the MLP →
   *drop* them before standardizing.
2. **Output-bias decay:** strong `weight_decay` under Adam shrank the output
   bias toward 0, pulling predictions toward 0 instead of the mean pIC50 (ST-RAE
   >1) → *standardize the target* so 0 is the mean.
3. **Too few optimizer steps / overfitting** from full-batch training and a wide,
   weakly-regularized net → *minibatch Adam* with stronger regularization.

![MLP loss curves](figures/mlp_diag_curves.png)

**After the fixes**, the two-head control (ecfp + per-iso + point) lands at
0.634 / 0.454 / 0.708 / 0.338 with fold std **0.012–0.037** (was 0.06–0.54) —
the fold-variance blow-up was optimization, now resolved. Seed-ensembling and
further regularization did not close the residual ~0.02–0.07 gap to ridge: on a
near-linear signal an MLP can match but not beat a regularized linear model, so
that remainder is model-class, not a bug.

**Consequence for the ablation study.** The control is the two-head
`ecfp+per-iso+point` config; each switch's effect is measured as a delta from it,
with Ridge and LightGBM kept as external reference columns. The ECFP+MLP control
is a stable sanity floor slightly below ridge; the D-MPNN encoder (switch c) is
where learned representations could actually beat ECFP+GBM.

## Reproduce

Numbers and plots regenerated from `data/` (pinned revision) by the EDA scripts,
`cyp.baseline`, `cyp.twohead`, and `experiments/`. Everything above is derived
solely from the public challenge dataset.
