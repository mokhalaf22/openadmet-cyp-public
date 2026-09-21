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
