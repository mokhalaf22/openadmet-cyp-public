"""Direction 1 (§31a) part A: predicted-Emax surrogate features, GFOLD-aligned.

§16 ruled out RAW Emax as test-unavailable — correct. But a STRUCTURE-derived
prediction of Emax is test-available, same pattern as predicted-log2FC (§25). Train
LightGBM structure->Emax for all 8 targets ({ISO}_EmaxVsPosCtrl on both arms:
direct_inhibition and TDI_condition), take GFOLD-OOF predictions for train and
full-model predictions for the 750 blinded. Cache to emax_gfold.npz for the D-MPNN
+primary+Emax retrain. lightgbm-only (no torch/chemprop) -> no OpenMP conflict.
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,"src")
import lightgbm as lgb
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5; SEED=0
ARMS=["direct_inhibition","TDI_condition"]
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv"); test=pd.read_csv(D+"cyp-challenge-TEST-BLINDED.csv")
emx=pd.read_csv(D+"cyp-challenge-TRAIN_Emax.csv")
z=np.load("experiments/plog_gfold.npz"); Xo=z["Xo"]; Xt=z["Xt"]; oko=z["oko"]; GFOLD=z["GFOLD"]
# align Emax targets to df row order via Molecule_Name
emx=df[["Molecule_Name"]].merge(emx,on="Molecule_Name",how="left")
cols=[f"{i}_EmaxVsPosCtrl_{a}" for i in ISOS for a in ARMS]  # 8 targets
E=np.stack([emx[c].to_numpy(float) for c in cols],1)
def lg(): return lgb.LGBMRegressor(n_estimators=500,learning_rate=0.03,num_leaves=31,min_child_samples=20,
    subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=1.0,random_state=SEED,deterministic=True,
    force_row_wise=True,n_jobs=8,verbosity=-1)
pemax_o=np.full((len(df),8),np.nan); pemax_t=np.full((len(test),8),np.nan)
print("predicted-Emax OOF quality (structure -> Emax), GFOLD:")
for k,c in enumerate(cols):
    ok=oko & ~np.isnan(E[:,k])
    oofp=np.full(len(df),np.nan)
    for f in range(NF):
        tr=np.where(ok&(GFOLD!=f)&(GFOLD>=0))[0]; va=np.where(ok&(GFOLD==f))[0]
        if len(tr) and len(va): m=lg(); m.fit(Xo[tr],E[tr,k]); oofp[va]=m.predict(Xo[va]); pemax_o[va,k]=oofp[va]
    trall=np.where(ok)[0]; m=lg(); m.fit(Xo[trall],E[trall,k]); pemax_t[:,k]=m.predict(Xt)
    mm=~np.isnan(oofp)&ok
    print(f"  {c:42} n={int(mm.sum())}  Pearson={np.corrcoef(oofp[mm],E[mm,k])[0,1]:.3f}  Spearman={spearmanr(oofp[mm],E[mm,k]).correlation:.3f}")
np.savez("experiments/emax_gfold.npz",pemax_o=pemax_o,pemax_t=pemax_t,cols=np.array(cols))
print("saved experiments/emax_gfold.npz  (8 predicted-Emax cols)")
