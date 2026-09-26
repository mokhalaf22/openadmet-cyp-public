# OpenADMET CYP Inhibition Blind Challenge — Model Report

This is the public model report for our entry. All results are **out-of-fold
(OOF) on scaffold splits** unless stated; none are on the blinded test set (its
labels are withheld). Full derivations, tables, and plots are in
[`FINDINGS.md`](FINDINGS.md); this document is the summary and rationale.

## TL;DR

- **Regression (direct pIC50, 4 isoforms).** Shared D-MPNN with an
  **interval-hinge** loss + a CYP3A4 D-MPNN/LightGBM blend reached macro OOF ST-RAE
  0.427 — but the **interim blind reveal was 0.9356 (rank 200)**: **the ranking
  holds (Spearman 0.6336) while the magnitudes collapse** (R² ≈ 0). Cause: L1
  shrinkage × a train/test domain shift (the blind set is *wider* than training,
  MAD ~1.14 vs 0.72), and a **confirmed CYP2D6 target shift** (the test excluded
  CYP2D6 hit-expansion, so CYP2D6 test compounds are less potent). OOF was an
  honest proxy for *ordering* but not for *scale*.
- **TDI classification (CYP3A4, CYP2D6).** Blind macro MCC **0.273 (rank 73)** —
  better than our OOF (~0.23; the blind set is easier). This is near the Tier-1
  band (26 entries statistically tied down to 0.3406), and the submission predated
  our CYP2D6 threshold fix.
- **Corrections applied post-reveal:** classification thresholds (CYP2D6 0.10→0.30,
  CYP3A4 0.35→0.45) to stop over-calling; and a regression dispersion + **validated
  CYP2D6 location** correction (§24). The identified real lever is a
  **tabular-foundation-model** approach (CheMeleon + predicted primary-screen +
  TabICL; ~0.68 blind in the OpenADMET post) — future work.
- **The differentiators are negative results**: the "derived TDI label" is
  empirically *worse* than a plain classifier; external AID 1851 *degrades* the
  model; the ECFP4 signal is essentially linear; and — most importantly — an
  **honest OOF-vs-blind post-mortem** (ranking preserved, magnitudes lost to
  domain shift).
- **No proprietary data.** One external public source (AID 1851) was examined and
  **not used**. Code is open (this repository).

## 1. Data and provenance

- **Source:** Hugging Face dataset `openadmet/cyp-challenge-train-test`, pinned at
  revision **`3ac9c5dbb83eec5780ec7fa511908698cfe1396d`** (last modified
  2026-08-27). Downloaded via `src/cyp/download.py` (`make data`) at that exact
  SHA — never HEAD — so the numbers are reproducible from a clean checkout.
- **Files used:** `cyp-challenge-TRAIN_TDI.csv` (6,145 rows; both direct and
  TDI-condition arms plus `{ISO}_is_TDI`) as the single training source, and
  `cyp-challenge-TEST-BLINDED.csv` (750 rows; **SMILES + Molecule_Name only**).
- **External source (examined, unused):** PubChem **AID 1851** (Veith CYP qHTS
  panel), 16,560 compounds × 5 isoforms. Public data — the proprietary-data flag
  is unchecked. It was tested as an auxiliary source and dropped (§8); disclosed
  here for completeness.

## 2. Confirmed data conventions

Verified empirically before modelling (FINDINGS §1–§5, §10):

- **Censoring is already in the credible intervals.** We consume the reported
  `{ISO}_pIC50_{arm}_conf_low`/`_conf_high` **verbatim** as interval bounds; there
  is no `(-inf, 4.0]` special case. The empirical lower limit of the fitted point
  is ~2.0, not 4.0, and censored compounds are encoded as a low `conf_low` (floor
  ~1.03) with a wide interval.
- **`TRAIN_TDI` is the single source.** `TRAIN_inhibition`'s direct values are
  byte-identical to `TRAIN_TDI`'s and its compounds are a strict subset.
- **`delta ≥ 0` (softplus).** Preincubation shifts are predominantly positive for
  the scored isoforms; enforced architecturally.
- **A missing direct arm means "not assayed," not "censored."** ~1,249 CYP3A4
  rows have a TDI arm but no direct DRC (Emax confirms the arm was never run);
  their `is_TDI=False` is a bookkeeping default and they are excluded from TDI
  train/eval via `src/cyp/guards.py`. This is distinct from the organizers'
  formal "assigned negative" (both arms measured and < 4).

## 3. Featurization

- **ECFP4 counts (2048 bits) + RDKit 2D descriptors** (`src/cyp/features.py`).
  For neural models, degenerate columns (`Ipc`, near-zero-variance, non-finite)
  are dropped *before* standardizing rather than clipped after — an unbounded
  `Ipc` (~1e14) and rare bits otherwise destabilize training (FINDINGS §11).
- **D-MPNN graph features** via chemprop's `BondMessagePassing` + mean
  aggregation, for the graph encoder.

## 4. The interval (censored) formulation, and why it works

Every target is an interval, not a point. The loss is an **interval hinge**: L1
distance to the nearest credible-interval bound, zero inside the interval
(`src/cyp/losses.py`). This matches the competition's soft-threshold metric
exactly, so training and scoring agree.

Empirically (FINDINGS §14), switching from point targets to interval targets
improved macro OOF ST-RAE from 0.445 to 0.434, and **recovered the CYP3A4 gap**:
CYP3A4 went from 0.323 to 0.305 (≈ the LightGBM parity of 0.297). CYP3A4 is
precisely the isoform with **23.6% of rows wider than 2 log units**, and its gap
had survived every encoder tuning attempt (more epochs, more width, GBM vs MLP;
§13). Scoring against the reported bounds — rather than a point the assay never
measured — is what closed it. The width-weighted L1 pull on top of the hinge added
nothing (null; §14): once the hinge is in place, the wide-interval signal is
already captured.

## 5. Final architecture

**Regression (`make baseline` is the LightGBM reference; the reported model is the
D-MPNN two-head):**
- Shared multi-task **D-MPNN** encoder (depth 3, d_h 200) with per-isoform
  direct-pIC50 heads, **interval targets**, 3-seed ensemble. Multi-task sharing
  helped (−0.032 macro) and a learned representation beat ECFP4 (−0.033; §12).
- **CYP3A4** uses a 50/50 blend of the D-MPNN and LightGBM (the two make different
  errors; the blend beats both — 0.276 vs 0.305 / 0.297; §17).
- Macro OOF ST-RAE **0.427**.

**TDI classification:** a **classifier head / model**, *not* the derived label
(§15). CYP3A4 stays GBM-favoured. (Final classification numbers in §7.)

## 6. Two-quantity (two-head) formulation and the derived-label refutation

The architecture predicts `mu` (direct pIC50) and `delta ≥ 0` (the preincubation
shift), so the TDI-arm pIC50 is `mu + delta` by construction — the design in
CLAUDE.md. On the **regression** side this is sound and free: adding TDI-arm
supervision does not hurt direct ST-RAE (0.433; §15).

But the headline idea — deriving the boolean TDI label *for free* from the two
arms via the organizers' rule — **does not hold up empirically** (§15). The
derived label collapses on CYP2D6 (MCC ≈ 0.00–0.05) because a *smooth* regression
`delta` cannot reproduce the sharp 0.301 threshold crossings that CYP2D6's
boundary-dominated positives require (§5). A plain classifier places its own
decision boundary and wins: CYP2D6 0.125 vs 0.031 derived, CYP3A4 ~tie. **We
therefore recommend a classifier for TDI, not the derived label** — reported here
as an honest negative result on the report's own proposed innovation.

## 7. Ablation summary

Regression, OOF ST-RAE macro (lower is better; global scaffold folds, 3-seed
ensemble unless noted):

| step | macro ST-RAE | note |
|---|---|---|
| LightGBM reference | 0.451 | ECFP4 + descriptors |
| two-head control (ECFP, per-isoform, point) | 0.522 | neural control |
| + multi-task shared encoder | 0.490 | −0.032 (helps) |
| + D-MPNN encoder (vs ECFP) | 0.457 | −0.033 (helps) |
| tuned D-MPNN (dh200, 300ep) | 0.445 | point targets |
| + interval targets | 0.434 | −0.011 (helps; recovers CYP3A4) |
| + width-weighted pull | 0.436 | null |
| + CYP3A4 D-MPNN/LightGBM blend (final) | **0.427** | −0.007 |

TDI classification, OOF MCC (3-seed ensemble):

| approach | CYP2D6 | CYP3A4 |
|---|---|---|
| baseline LightGBM classifier | 0.097 | 0.347 |
| derived label (from two-head) | 0.031 | 0.337 |
| **shared-model classifier head (final)** | **0.125** | **0.336** |
| standalone classifier (own encoder) | 0.111 | 0.288 |

The **shared-model classifier head is the final TDI model**: a standalone
TDI-only encoder is *worse* (0.111 / 0.288), so encoder-sharing helps
classification rather than costing it — the regression data regularizes the
encoder (§20). MCC seed spread here is ~0.03–0.06, larger than the regression
floor, so these numbers carry that uncertainty.

Levers that did **not** move classification (all null or negative): threshold
recalibration (§10), `shift_prior` (§15), Emax / single-concentration features
(§16, also test-unavailable), training-prevalence reweighting (§18), including the
excluded direct-less rows (§18, hurts), predicted-`delta` as a feature (§19,
null), AID 1851 auxiliary heads (§8, hurts), and a standalone TDI-only encoder (§7, worse).

## 8. Negative results (the differentiator)

1. **The derived TDI label is worse than a classifier** (§6, §15). Architectural
   elegance did not translate to MCC; the task is discrimination-limited (§5, §10).
2. **PubChem AID 1851 auxiliary heads degrade the model.** Adding 5 Veith aux
   heads to the shared encoder moved direct ST-RAE 0.433 → 0.448 and CYP2D6/CYP3A4
   MCC 0.125/0.336 → 0.065/0.310 — worse on every metric. The explanation is
   structural distance: the blinded compounds' nearest-neighbour ECFP4 Tanimoto to
   AID 1851 is only **median 0.368** (vs 0.587 to our own training), so the
   distant qHTS chemistry pulls the shared encoder off our DRC chemistry. A large
   external dataset is not automatically useful; chemical-space overlap gates it.
3. **The signal on ECFP4 is essentially linear.** Ridge regression ≈ LightGBM
   (macro 0.489 vs 0.451) and a well-optimized MLP does no better (§11). Extra
   model capacity does not help — the ceiling is **representational, not
   capacity-limited**, which is exactly why the learned D-MPNN representation, not
   more tuning, was the lever that moved the floor.
4. **The predicted-primary-screen feature is the lever; the foundation-model
   wrapper is unproven on our hardware** (§26). We tried the OpenADMET tabular-FM
   pipeline — CheMeleon D-MPNN embeddings (PCA-256) + the predicted-primary-screen
   feature → TabICL — but **could not run the method as configured**: TabICL's
   default `n_estimators=8` does not complete on this 24 GB M4 Pro (stalls at ~12 %
   CPU regardless of offload mode). Only a reduced `n_estimators=4` variant ran,
   and it was **comparable** to our LightGBM with the same feature (macro OOF 0.444
   vs 0.433, gap inside the per-fold spread) — which is *not* evidence the method
   loses, only that we could not test it at full strength. What the exercise does
   confirm, model-class-independently, is the OpenADMET decomposition (CheMeleon-only
   0.83 → full 0.68 blind): **the structure-predicted primary-screen log2FC feature
   (§25) carries the gain**, and we already use it. That lever is a *feature*, not a
   foundation model.

## 9. Validation reliability — OOF underestimated blind error ~2×

A finding worth stating plainly for anyone building on this benchmark: our
**scaffold-split OOF macro ST-RAE was 0.451, while the blind interim was 0.9356**
— OOF underestimated blind error by roughly **2×**. Scaffold OOF preserved the
*ranking* (blind Spearman 0.6336) but badly underestimated the *magnitude* error.

Two causes (diagnosed, not speculated):
- **Prediction compression under domain shift.** L1/interval regression predicts
  the conditional median, which is compressed; on a blind set drawn from a
  different potency regime this collapses to near the predict-a-constant baseline
  (R² ≈ 0). Our predicted MAD was ~0.36 vs our training targets' 0.72.
- **A confirmed CYP2D6 target-distribution shift.** The test set was built by hit
  expansion on CYP3A4/CYP1A2/CYP2C9 only, so CYP2D6 test compounds are less potent
  than training; predicting at the training-potent level (~4.7) is a systematic
  over-prediction on the test.

Lesson: on this benchmark, scaffold-split OOF is a usable proxy for *ordering* but
not for *calibrated magnitude*. A held-out split that deliberately mimics the
test's potency shift (per isoform) would have caught this; plain scaffold CV did
not, because it cannot see target-space shift (only feature-space, which we
checked and found absent).

## 10. Limitations

- **All numbers are scaffold-split OOF, not blind.** Scaffold OOF is a credible
  proxy here — the blinded set is *not* near-duplicate hit expansion (median NN
  Tanimoto 0.587 to training; §7 of FINDINGS) — but this must be confirmed against
  a submission.
- **The classification gap to the field is unexplained.** Every lever in the
  provided data is exhausted; the remaining explanation lies in the top entries'
  features/representation/data, which we could not access.
- **AID 1851 negative has two caveats** (FINDINGS §19): compounds overlapping our
  data were excluded (possibly removing scale-anchoring molecules), and the
  per-epoch aux subsampling variance was not measured against the seed floor.
  Neither plausibly flips the sign.
- **Neural tuning used multi-threaded (non-bitwise-reproducible) training**; the
  ~0.004 seed-ensemble spread bounds the resulting noise, and every reported delta
  is judged against it.

## 11. Reproducibility & disclosures

- Pinned data revision; deterministic scaffold folds; `make data | inspect |
  baseline | submit`. Experiments in `experiments/` are phased and checkpointed.
- **Disclosures:** no proprietary data was used. PubChem AID 1851 (public) was
  examined and not used in the final model. Code is open-source (this repository).
