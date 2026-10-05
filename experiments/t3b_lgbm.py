"""Decorrelated member (a): LightGBM CYP2D6 specialist on ECFP + descriptors.

Chosen for DECORRELATION, not individual strength (§47: our members sat at rank
correlation 0.89-0.95, which is why averaging could not gain). This member differs from the
shared D-MPNN in model class, feature set (no predicted-primary feature -- §31b identified
that feature as what converged our models) and objective (width-weighted L1 rather than the
interval hinge).

Own process: LightGBM segfaults when it shares one with chemprop's OpenMP (§26).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "src")
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/cyp-challenge-train-test"
ISO, NF, SEED = "CYP2D6", 5, 0


def main() -> None:
    df = pd.read_csv(D / "cyp-challenge-TRAIN_TDI.csv")
    z = np.load(ROOT / "experiments/plog_gfold.npz")
    GFOLD, oko, Xo = z["GFOLD"], z["oko"], z["Xo"]      # ECFP + descriptors only
    y = df[f"{ISO}_pIC50_direct_inhibition"].to_numpy(float)
    lo = df[f"{ISO}_pIC50_direct_inhibition_conf_low"].to_numpy(float)
    hi = df[f"{ISO}_pIC50_direct_inhibition_conf_high"].to_numpy(float)
    w = hi - lo
    w = np.where(np.isnan(w), np.nanmedian(w), w)
    sw = 1.0 / (1.0 + w)
    pres = (~np.isnan(y)) & oko

    oofs = []
    for seed in (0, 1, 2):
        oof = np.full(len(df), np.nan)
        for f in range(NF):
            tr = np.where(pres & (GFOLD != f) & (GFOLD >= 0))[0]
            va = np.where(pres & (GFOLD == f))[0]
            if not len(tr) or not len(va):
                continue
            m = lgb.LGBMRegressor(objective="regression_l1", n_estimators=500, learning_rate=0.03,
                                  num_leaves=31, min_child_samples=20, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                                  random_state=seed, deterministic=True, force_row_wise=True,
                                  n_jobs=1, verbosity=-1)
            m.fit(Xo[tr], y[tr], sample_weight=sw[tr])
            oof[va] = m.predict(Xo[va])
        oofs.append(oof)
        print(f"  lgbm seed {seed} done", flush=True)
    np.save(ROOT / "experiments/t3b_lgbm_oof.npy", np.nanmean(oofs, 0))
    print("saved experiments/t3b_lgbm_oof.npy")


if __name__ == "__main__":
    main()
