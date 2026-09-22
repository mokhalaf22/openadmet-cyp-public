"""Regularization grid experiment (not a redesign). Regression only.

For each isoform and each (colsample_bytree, n_estimators) setting, run the
baseline's exact scaffold-split OOF and report OOF prediction std and OOF ST-RAE
(mean +/- fold std). Featurize once, reuse across settings.
"""
import numpy as np, pandas as pd, torch
import lightgbm as lgb
from cyp.features import featurize
from cyp.splits import scaffold_folds
from cyp.losses import st_rae as st_rae_torch

D="data/cyp-challenge-train-test/"
ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]
SEED=0; N_FOLDS=5
GRID=[(0.5,500),(0.5,1500),(0.8,500),(0.8,1500)]  # (colsample_bytree, n_estimators); baseline = (0.5,500)

df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
print("featurizing once...")
X,ok,_=featurize(df["SMILES"].tolist())

def st_rae(pred,lo,hi,truth):
    return st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (pred,lo,hi,truth)))

def reg(colsample,n_est):
    return lgb.LGBMRegressor(objective="regression_l1",n_estimators=n_est,learning_rate=0.03,
        num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,
        colsample_bytree=colsample,reg_lambda=1.0,random_state=SEED,deterministic=True,
        force_row_wise=True,n_jobs=1,verbosity=-1)

def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0

for iso in ISOS:
    y_all=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float)
    present=~np.isnan(y_all)&ok
    idx=np.where(present)[0]
    smi=df["SMILES"].to_numpy()[idx]
    fold,_=scaffold_folds(smi.tolist(),N_FOLDS,SEED)
    y=y_all[idx]; Xs=X[idx]
    lo=df[f"{iso}_pIC50_direct_inhibition_conf_low"].to_numpy(float)[idx]
    hi=df[f"{iso}_pIC50_direct_inhibition_conf_high"].to_numpy(float)[idx]
    w=hi-lo; w=np.where(np.isnan(w),np.nanmedian(w),w); weight=1.0/(1.0+w)
    print(f"\n=== {iso} (n={len(idx)}) ===  baseline=(0.5,500)")
    print(f"  {'colsample':>9} {'trees':>5} {'OOF pred std':>13} {'OOF ST-RAE (mean+/-fold sd)':>28}")
    for cs,ne in GRID:
        oof=np.full(len(idx),np.nan); srae=[]
        for f in range(N_FOLDS):
            va=fold==f; tr=~va
            m=reg(cs,ne); m.fit(Xs[tr],y[tr],sample_weight=weight[tr])
            oof[va]=m.predict(Xs[va])
            srae.append(st_rae(oof[va],lo[va],hi[va],y[va]))
        tag="  <-- baseline" if (cs,ne)==(0.5,500) else ""
        print(f"  {cs:>9} {ne:>5} {std1(oof):>13.3f}   {np.mean(srae):>8.3f} +/- {std1(srae):>5.3f}{tag}")
print("\ndone")
