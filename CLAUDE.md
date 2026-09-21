# CLAUDE.md — openadmet-cyp-public

Public entry to the OpenADMET CYP Inhibition Blind Challenge.
This repo will be linked publicly as the required model report. Assume every
file here is world-readable.

## Hard rules

1. **Nothing from the ESP / point-cloud project enters this repo.** Not code,
   not results, not comparisons, not a sentence in a README. That work is
   unpublished lab IP. If a task seems to need it, stop and ask.
2. **Never merge external CYP data into the scored pIC50 columns.** External
   sources (PubChem AID 1851 / Veith, ChEMBL, BindingDB) are different assays
   on different scales. Each `source × isoform × readout` becomes its own
   column and its own prediction head. No cross-assay rescaling, ever.
3. **Never touch the blinded test file except to read SMILES and IDs.** No
   fitting on it, no transductive tricks, no using its descriptor distribution
   to pick hyperparameters.
4. **Never use random train/val splits.** Scaffold or Butina cluster splits
   only. See "Why" below.
5. **One submission per team/lab.** Do not create alternate leaderboard
   entries.

## Task

Two tracks, 750 blinded compounds:

- **Direct inhibition (regression)** — pIC50 for CYP1A2, CYP2C9, CYP2D6,
  CYP3A4. Metric: macro-averaged Soft-Threshold Relative Absolute Error
  (MA-ST-RAE). Error is distance to the nearest bound of the ground-truth
  credible interval; inside the interval scores zero. Lower is better; ~1.0 is
  the predict-a-constant baseline.
- **TDI (classification)** — boolean, IC50 shift after +NADPH preincubation
  exceeds 2-fold. Scored by MCC. **Only CYP3A4 and CYP2D6 are scored.**

Deadline: final submissions 2026-11-03, one minute to midnight UTC.

## Modelling approach (the thing that makes this entry distinct)

Do not build three independent models. Predict **two quantities per compound
per isoform**:

- `mu` — direct-arm pIC50
- `delta` — the preincubation shift, constrained `>= 0` via softplus

The TDI-arm pIC50 is `mu + delta` by construction. The boolean label then
follows from the organizers' own definition rather than a separate classifier:

- if `mu >= 4`:  positive iff `delta > 0.301`
- if `mu <  4`:  positive iff `mu + delta > 4.301`   (the "inferred positive" case)

Rationale: the two arms are the same compound, turnover can only make a
compound a *better* inhibitor, and the label is a deterministic function of the
two pIC50s. Enforcing that architecturally means the ~1,500 DRCs per isoform
train both tracks at once, and the inferred-positive class stops being a
special case.

**Losses are interval-hinge, not squared error.** See `src/cyp/losses.py`.
Every target is an interval, not a point:

- observed value with a credible interval -> `[lo, hi]`
- left-censored below the lowest tested dose -> `(-inf, 4.0]`

Loss is zero inside the interval and L1 outside. This matches the scoring
metric exactly and stops the model learning to predict fiction for compounds
whose true value is only known to be "below 4".

Backbone: Chemprop D-MPNN (or CheMeleon embeddings) + RDKit descriptors,
multi-task heads across the four isoforms sharing one encoder. The training
matrix is sparse — most compounds carry only one or two isoforms — so mask
missing targets in the loss.

## Why scaffold splits

The 750-compound test set is hit expansion: 75 potent parents plus their ~10
nearest Enamine chemisimilars each. The leaderboard/blind split is made *by
series*, so all analogs of a parent land on the same side. Random CV puts
near-duplicates in both train and val and produces a number that will not
survive submission. Always report scaffold-split OOF.

## Dataset

Source: Hugging Face dataset `openadmet/cyp-challenge-train-test`.

**Pinned revision:** `3ac9c5dbb83eec5780ec7fa511908698cfe1396d`
(last modified 2026-08-27). Downloaded via `src/cyp/download.py`, which passes
this SHA explicitly — never HEAD. `make data` fetches it into
`data/cyp-challenge-train-test/`. If upstream changes, bump the SHA here and in
`download.py` in the same commit and re-run `make inspect`.

Files (confirmed by `make inspect`): `cyp-challenge-TEST-BLINDED.csv` (750,
IDs+SMILES only), `cyp-challenge-TRAIN_inhibition.csv` (4,905, direct arm),
`cyp-challenge-TRAIN_TDI.csv` (6,145, both arms + `{ISO}_is_TDI`),
`cyp-challenge-TRAIN_Emax.csv` (6,145), and
`cyp-challenge-single-concentration-TRAIN.csv` (17,504, raw screen).

## Confirmed data conventions

Verified from the pinned dataset by EDA (tables, plots, and noise estimates in
`FINDINGS.md`). Where these conflict with the narrative above, these win.

- **Censoring is already in the intervals — no `(-inf, 4.0]` case.** Use
  `{ISO}_pIC50_{arm}_conf_low` / `_conf_high` verbatim as the interval bounds
  `[lo, hi]` in the hinge loss. Left-censored compounds are encoded as a low
  `conf_low` (finite floor ~1.03) with a wide interval (width piles up at
  ~2.0–2.8); the reported bounds already say "only known to be below X". Drop
  the flat `4.0` cap described under "Modelling approach" and consume the
  reported bounds everywhere.
- **`TRAIN_TDI` is the single training source; `TRAIN_inhibition` is ignored.**
  Its direct-arm values are byte-identical to `TRAIN_TDI` (Pearson 1.0, MAE 0)
  and its compounds are a strict subset. The direct and TDI arms are the same
  assay campaign, not separate runs, so one shared direct-arm head — no
  per-file heads. (`TRAIN_TDI` adds ~1,238 CYP3A4 TDI-condition-only rows on
  top; it never adds direct-arm rows.)
- **`delta >= 0` (softplus) is justified for the scored TDI isoforms.** CYP2D6
  and CYP3A4 preincubation shifts are predominantly positive; negatives are
  ~13% and almost all within measurement noise. CYP1A2/CYP2C9 (not scored for
  TDI) center on zero, where the constraint is harmless.
- **A missing direct arm means "not assayed", NOT "censored".** ~1,249 CYP3A4
  rows (~35% of its TDI labels) have a measured TDI arm but no direct DRC: every
  direct-side column is NaN and the Emax file confirms the direct arm was never
  run. They are frequently potent under TDI conditions (median TDI pIC50 5.4),
  so this is not left-censoring. Their `is_TDI = False` is an assigned default
  carrying no experimental information. Never supervise `mu` on these rows, and
  exclude them from TDI classification training and internal evaluation (guard:
  `cyp.guards.assigned_negative_mask`). Distinct from the organizers' formal
  "assigned negative" class (direct < 4 AND TDI-arm < 4), which is scored on the
  blinded test — see FINDINGS.md.
- **Empirical LOQ of the fitted direct pIC50 is ~2.0, not 4.0**, with no hard
  edge; the censoring signal is in `conf_low` (floor ~1.03), not the point.

Two things share the phrase "assigned negative" and must never be conflated.
The organizers' **assigned negative** (direct pIC50 < 4 AND TDI-arm pIC50 < 4)
is a compound measured in *both* arms and found inactive in both — real
experimental content, and part of the scored negative class on the blinded test.
Our **direct-arm-never-assayed** rows have no direct measurement at all (Emax
confirms the arm was not run); their `is_TDI = False` is a bookkeeping default
with no experimental content, and they are excluded from TDI train/eval via
`guards.py`. Same phrase, opposite information value — keep them separate.

## Environment

MacBook Pro M4 Pro, 24 GB. CPU is fine for everything here — a D-MPNN over
~5–6k molecules trains in minutes. If Chemprop throws dtype errors on MPS,
force CPU rather than debugging it; it costs no real time.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

```
make inspect     # print every downloaded file + columns; run this FIRST
make baseline    # LightGBM reference submission
make train       # the two-head censored model
make submit      # validate schema + write submission.csv
```

## Submission schema

Recorded verbatim from the challenge Space `config.py` / Submit tab (do not
infer these). Two **independent** files, one per track — each a `.parquet`
(preferred) or `.csv` with **exactly 750 rows**, one per test compound, no
`NaN`/`inf`.

Identifier columns (both files, `IDENTIFIER_COLUMNS`): `SMILES`, `Molecule_Name`.

Regression track (`REQUIRED_REGRESSION_COLUMNS` = identifiers + endpoints):
```
SMILES, Molecule_Name,
CYP1A2_pIC50_direct_inhibition, CYP2C9_pIC50_direct_inhibition,
CYP2D6_pIC50_direct_inhibition, CYP3A4_pIC50_direct_inhibition
```
Each endpoint is a `float` pIC50.

Classification track (`REQUIRED_CLASSIFICATION_COLUMNS` = identifiers + endpoints):
```
SMILES, Molecule_Name, CYP2D6_is_TDI, CYP3A4_is_TDI
```
Each endpoint is a `bool` (`True`/`False` or `1`/`0`). Note the endpoint order
is CYP2D6 then CYP3A4.

Server-side validation (`submission.py::_read_tabular_submission`) only checks:
file is `.parquet`/`.csv`, `len(df) == 750`, and all required columns are
present (`set(required) - set(df.columns)`); extra columns are ignored
server-side. Our `make submit` validator is deliberately stricter.

## Before every submission

Run `make submit`, which must assert:
- exactly 750 rows
- IDs match the blinded file exactly, same order-independent set
- no NaNs in any prediction column
- TDI columns are boolean, not probabilities
- column names match the Submit tab verbatim

## Reporting obligations

- Proprietary data use must be disclosed. We use none — keep it that way, and
  say so explicitly in the report.
- Open code is tracked by checkbox. This repo is the link.
- There is a separate award for most innovative approach, judged partly
  independently of leaderboard rank. The censored two-head formulation is the
  story; write it up as a statistical argument, not as "we used a big model".

## Style

- No results in commit messages or the README until they are OOF-validated on
  scaffold splits.
- Every claimed improvement needs a before/after on the same folds. Gains on
  raw MAE that vanish under ST-RAE are not gains.
- Prefer small readable functions over clever pipelines. This code will be read
  by strangers.
