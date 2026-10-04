"""Phase 3(a): per-fold proxy pIC50 targets for screen-only molecules.

Trains log2FC -> DRC pIC50 on compounds that have BOTH readouts, then predicts a proxy pIC50
for the ~4.4k molecules that appear only in the single-concentration screen. Those become
extra, down-weighted training ROWS in phase3_train.py -- distinct from the predicted-primary
FEATURE (§25) already in the model.

Two correctness points:
  * the mapping is fit PER FOLD on that fold's training rows only; a proxy model fit on
    validation-fold compounds would leak their DRC labels into the training signal;
  * run as its own process because LightGBM segfaults when it shares one with chemprop's
    OpenMP (see §26 engineering note), hence also n_jobs=1.

Writes experiments/proxy_targets.npz: smiles, proxy (n, 4, n_folds), fit quality per fold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

sys.path.insert(0, "src")
from cyp.features import featurize  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/cyp-challenge-train-test"
ISOS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
NF, SEED = 5, 0


def lgbm():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                             min_child_samples=20, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.6, reg_lambda=1.0, random_state=SEED,
                             deterministic=True, force_row_wise=True, n_jobs=1, verbosity=-1)


def main() -> None:
    df = pd.read_csv(D / "cyp-challenge-TRAIN_TDI.csv")
    sc = pd.read_csv(D / "cyp-challenge-single-concentration-TRAIN.csv")
    z = np.load(ROOT / "experiments/plog_gfold.npz")
    GFOLD, oko, Xo = z["GFOLD"], z["oko"], z["Xo"]

    scp = sc.pivot_table(index="SMILES", columns="enzyme", values="log2fc_estimate",
                         aggfunc="mean").reindex(columns=ISOS)
    sc_smiles = scp.index.tolist()
    SCY = scp.to_numpy()

    def ik(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return inchi.MolToInchiKey(m) if m else None

    # RETARGETED (see FINDINGS §44): there are NO screen-only molecules -- all 4,376 unique
    # screen compounds are already in the DRC table. The opportunity is instead the sparse
    # DRC matrix: (compound, isoform) CELLS with a measured log2FC but no DRC pIC50. Those
    # get proxy targets and enter as extra down-weighted supervision on existing rows.
    sc_ik = [ik(s) for s in sc_smiles]
    sc_map = {k: SCY[i] for i, k in enumerate(sc_ik) if k is not None}
    L = np.array([sc_map.get(ik(s), [np.nan] * 4) for s in df["SMILES"]], float)
    Y = np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS], 1)
    fillable = (~np.isnan(L)) & np.isnan(Y) & oko[:, None]
    print(f"supervised DRC cells: {int(((~np.isnan(Y)) & oko[:, None]).sum())} | "
          f"proxy-able cells (log2FC present, DRC absent): {int(fillable.sum())}", flush=True)

    proxy = np.full((len(df), 4, NF), np.nan)
    quality = {}
    for f in range(NF):
        for j, iso in enumerate(ISOS):
            # fit the log2FC -> DRC mapping on this fold's TRAINING rows only
            both = (~np.isnan(Y[:, j])) & oko & (~np.isnan(L[:, j])) & (GFOLD != f) & (GFOLD >= 0)
            tr = np.where(both)[0]
            if len(tr) < 50:
                continue
            m = lgbm()
            m.fit(np.concatenate([Xo[tr], np.nan_to_num(L[tr])], 1), Y[tr, j])
            need = np.where(fillable[:, j])[0]
            if len(need):
                proxy[need, j, f] = m.predict(
                    np.concatenate([Xo[need], np.nan_to_num(L[need])], 1))
            # honesty check: recover DRC pIC50 on the held-out fold
            va = np.where((~np.isnan(Y[:, j])) & oko & (~np.isnan(L[:, j])) & (GFOLD == f))[0]
            if len(va) > 5:
                p = m.predict(np.concatenate([Xo[va], np.nan_to_num(L[va])], 1))
                quality.setdefault(iso, []).append(float(np.corrcoef(p, Y[va, j])[0, 1]))
        print(f"  fold {f} done", flush=True)

    np.savez(ROOT / "experiments/proxy_targets.npz", proxy=proxy,
             fillable=fillable, folds=NF)
    print("\nproxy model quality (Pearson r, held-out fold, log2FC+structure -> DRC pIC50):")
    for iso in ISOS:
        q = quality.get(iso, [])
        print(f"  {iso}: {np.mean(q):.3f}" if q else f"  {iso}: n/a")
    print(f"saved experiments/proxy_targets.npz  "
          f"({int(fillable.sum())} proxy cells x {NF} folds)")


if __name__ == "__main__":
    main()
