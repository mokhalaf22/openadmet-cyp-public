# Experiments

One-off experiments that drove a documented decision. Each reuses the package
modules (`cyp.features`, `cyp.splits`, `cyp.losses`, `cyp.guards`) and the same
scaffold folds as the baseline, so results are directly comparable. Run from the
repo root with the venv active, e.g. `PYTHONPATH=src python experiments/reg_grid.py`.

Findings live in `FINDINGS.md`; this directory keeps the scripts reproducible.

| script | question it answered | decision it drove |
|---|---|---|
| `reg_grid.py` | Does relaxing LightGBM regularization (`colsample_bytree` 0.5→0.8, trees 500→1500) widen the shrunk baseline predictions enough to improve OOF ST-RAE? | **No** — predictions widen (~0.02–0.04 std) but ST-RAE stays flat within fold-to-fold variance on every isoform. Per the decision rule, the interim submission was **left unchanged** (baseline `(0.5, 500)`, commit `5d5dbdd`). See FINDINGS §8. |
