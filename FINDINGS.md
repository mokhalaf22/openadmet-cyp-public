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

Median reported per-measurement std and the propagated noise on the arm
difference (used above to judge whether negative shifts are real):

| isoform | direct `_std` | TDI `_std` | ~diff noise |
|---|---|---|---|
| CYP1A2 | 0.084 | 0.088 | 0.121 |
| CYP2C9 | 0.137 | 0.133 | 0.191 |
| CYP2D6 | 0.069 | 0.065 | 0.095 |
| CYP3A4 | 0.095 | 0.061 | 0.113 |

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

## Reproduce

Numbers and plots regenerated from `data/` (pinned revision) by the EDA scripts.
Everything above is derived solely from the public challenge dataset.
