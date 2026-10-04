# OpenADMET CYP Inhibition Blind Challenge — public entry

Our entry to the OpenADMET CYP Inhibition Blind Challenge: direct-inhibition pIC50 regression
for CYP1A2/2C9/2D6/3A4 (macro soft-threshold relative absolute error) and TDI classification for
CYP3A4/CYP2D6 (MCC).

## Read this first

- **[`REPORT.md`](REPORT.md)** — the model report: the argument, the final models, and what to
  reuse vs. not bother trying. Start here.
- **[`FINDINGS.md`](FINDINGS.md)** — the full evidence log (numbered experiments §1–§38): every
  ablation, negative result, live-board result, and derivation behind the report.

**One-line summary.** The ceiling on this benchmark is *representational*: a learned D-MPNN
representation with an interval-hinge loss, plus one structure-predicted primary-screen feature,
moved the floor as far as public data allows; the remaining gap is a property of the data (the
blinded compounds are Enamine chemistry with almost no public measured data), and the actual
leaderboard gains came from validated post-hoc calibration.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make data       # fetch the pinned dataset revision into data/ (never HEAD)
make inspect     # print every downloaded file + columns
make baseline    # LightGBM reference + classifier (writes predictions under data/)
make submit      # validate submission schema
pytest           # data-invariant, loss, split, and submission tests
```

The experiment scripts behind the report live in [`experiments/`](experiments/) (see
[`experiments/README.md`](experiments/README.md)); each is phased and checkpointed. Numbers in
the report and findings regenerate from the pinned data revision.

## Layout

- `src/cyp/` — library: `download`, `features`, `splits` (scaffold folds), `losses`
  (interval hinge + ST-RAE), `guards`, `baseline`, `twohead`, `submit`.
- `experiments/` — ablations and the modelling studies (results as `*.json`; large
  per-seed arrays and data intermediates are git-ignored).
  - `runner.py` + `phases.py` — the gated phase runner used for the later experiments: steps
    are checkpointed, every result is appended to `ledger.json`, and a readable summary is
    written to [`LEDGER.md`](experiments/LEDGER.md). It stops at each phase gate by design.
- `tests/` — invariants, losses, splits, submission schema.
- `figures/` — EDA plots (from the public training data).

## Data and disclosures

Data is **not committed** (`data/` is git-ignored); `make data` fetches the pinned public
revision. No proprietary data is used. Public external sources (PubChem AID 1851, ChEMBL/PubChem
near-neighbours, the Octant CYP release) were examined and are disclosed in REPORT §9; none is
merged into a scored prediction column. A pre-ingest structure-level leakage check of the
same-lab Octant release found and quarantined blinded test compounds present there (REPORT §9,
FINDINGS §34) — reported to the organizers.
