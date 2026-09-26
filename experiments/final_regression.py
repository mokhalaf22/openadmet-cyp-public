"""Final regression submission (§26 winner): LightGBM + predicted-primary-screen.

The like-for-like comparison (§26) on our scaffold folds put LightGBM +
predicted-log2FC at macro OOF 0.433, ahead of the CheMeleon-256 + predicted-log2FC
-> TabICL tabular-FM pipeline at 0.444. So the winner is our GBM + predicted
primary screen. This script produces its BLINDED test predictions and applies the
validated CYP2D6 -0.5 location shift (§24), then writes a schema-valid parquet.

Uses the cached features tabfm.py wrote (lightgbm-only process — no torch/chemprop,
so no OpenMP conflict):
  experiments/tabfm_feat.npz   Xo (train base feats, ECFP+desc, degenerate dropped),
                               Xt (test base feats), oko/okt validity masks
  experiments/tabfm_plog.npz   plog_o (OOF predicted-log2FC, train), plog_t (full-model, test)

pIC50 model = exactly §25's: LightGBM regression_l1, width-weighted, trained on all
present rows per isoform with [base | 4 predicted-log2FC]. Train uses the OOF
predicted-log2FC (plog_o) to match the validated feature distribution; test uses the
full-model predicted-log2FC (plog_t).
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "src")
from cyp.submit import write_submission, load_blinded_ids, REGRESSION_ENDPOINTS
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; SEED=0
CYP2D6_SHIFT=-0.5  # validated location shift (§24): blinded CYP2D6 is a less-potent distribution
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv"); test=pd.read_csv(D+"cyp-challenge-TEST-BLINDED.csv")

z=np.load("experiments/tabfm_feat.npz"); Xo=z["Xo"]; Xt=z["Xt"]; oko=z["oko"]; okt=z["okt"]
zp=np.load("experiments/tabfm_plog.npz"); plog_o=zp["o"]; plog_t=zp["t"]
Ftr=np.concatenate([Xo,plog_o],1); Fte=np.concatenate([Xt,plog_t],1)

def lgbm(): return lgb.LGBMRegressor(objective="regression_l1",n_estimators=500,learning_rate=0.03,
    num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,
    reg_lambda=1.0,random_state=SEED,deterministic=True,force_row_wise=True,n_jobs=8,verbosity=-1)

pred=np.full((len(test),4),np.nan)
for j,iso in enumerate(ISOS):
    y=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float)
    lo=df[f"{iso}_pIC50_direct_inhibition_conf_low"].to_numpy(float); hi=df[f"{iso}_pIC50_direct_inhibition_conf_high"].to_numpy(float)
    w=hi-lo; w=np.where(np.isnan(w),np.nanmedian(w),w); wt=1.0/(1.0+w)
    pres=~np.isnan(y)&oko&~np.isnan(Ftr).any(1)
    tr=np.where(pres)[0]
    m=lgbm(); m.fit(Ftr[tr],y[tr],sample_weight=wt[tr])
    p=m.predict(Fte)
    # invalid test rows (featurize failed): fall back to train mean
    bad=~okt | np.isnan(Fte).any(1)
    p[bad]=float(y[tr].mean())
    if iso=="CYP2D6": p=p+CYP2D6_SHIFT
    pred[:,j]=p
    print(f"{iso}: test mean={p.mean():.3f} std={p.std():.3f} (train mean={y[tr].mean():.3f}){'  [CYP2D6 -0.5]' if iso=='CYP2D6' else ''}",flush=True)

sub=test[["SMILES","Molecule_Name"]].copy()
for j,ep in enumerate(REGRESSION_ENDPOINTS): sub[ep]=pred[:,j]
assert not sub[REGRESSION_ENDPOINTS].isna().any().any(), "NaN in predictions"
out=write_submission(sub,"regression","submissions/regression_final.parquet",load_blinded_ids())
print(f"\nwrote {out}  rows={len(sub)}  cols={list(sub.columns)}")
