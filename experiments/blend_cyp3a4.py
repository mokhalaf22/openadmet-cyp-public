import numpy as np, pandas as pd, lightgbm as lgb
from cyp.features import featurize
from cyp.splits import scaffold_folds
from cyp.losses import st_rae as st_rae_torch
import torch
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5; SEED=0
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
X,ok,names=featurize(df["SMILES"].tolist())
var=X[ok].var(0); drop=(var<1e-8)|~np.isfinite(X[ok]).all(0)
drop[[i for i,n in enumerate(names) if n=="Ipc"]]=True
Xc=X[:,~drop].astype(np.float32)
valid=np.where(ok)[0]; gf,_=scaffold_folds(df["SMILES"].to_numpy()[valid].tolist(),NF,SEED)
GFOLD=np.full(len(df),-1); GFOLD[valid]=gf
def col(i,s): return f"{i}_pIC50_direct_inhibition{s}"
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0
# D-MPNN interval OOF (3-seed ensemble)
dmpnn=np.nanmean([np.load(f"experiments/p2_interval_s{s}.npy") for s in [0,1,2]],0)  # (N,4)
# LightGBM OOF on same global folds, all isoforms (for reference) + 3A4 blend
def lgbm_oof(iso):
    y=df[col(iso,"")].to_numpy(float); lo=df[col(iso,"_conf_low")].to_numpy(float); hi=df[col(iso,"_conf_high")].to_numpy(float)
    w=hi-lo; w=np.where(np.isnan(w),np.nanmedian(w),w); wt=1.0/(1.0+w)
    oof=np.full(len(df),np.nan)
    present=~np.isnan(y)&ok
    for f in range(NF):
        va=np.where(present&(GFOLD==f))[0]; tr=np.where(present&(GFOLD!=f)&(GFOLD>=0))[0]
        m=lgb.LGBMRegressor(objective="regression_l1",n_estimators=500,learning_rate=0.03,num_leaves=31,
            min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=1.0,
            random_state=SEED,deterministic=True,force_row_wise=True,n_jobs=8,verbosity=-1)
        m.fit(Xc[tr],y[tr],sample_weight=wt[tr]); oof[va]=m.predict(Xc[va])
    return oof
def iso_strae(pred,iso):
    y=df[col(iso,"")].to_numpy(float); lo=df[col(iso,"_conf_low")].to_numpy(float); hi=df[col(iso,"_conf_high")].to_numpy(float)
    present=~np.isnan(y)&ok; sr=[]
    for f in range(NF):
        va=np.where(present&(GFOLD==f))[0]
        if len(va): sr.append(strae(pred[va],lo[va],hi[va],y[va]))
    return np.mean(sr),std1(sr)
print("LightGBM 3A4 OOF (global folds)...")
lg3a4=lgbm_oof("CYP3A4")
j=ISOS.index("CYP3A4")
print("\nCYP3A4 blend weight sweep (w = D-MPNN weight):")
best=(1e9,None)
for w in [0.0,0.25,0.5,0.6,0.75,1.0]:
    blend=w*dmpnn[:,j]+(1-w)*lg3a4
    m,s=iso_strae(blend,"CYP3A4"); 
    print(f"  w={w:.2f}: 3A4 ST-RAE={m:.3f}±{s:.3f}")
    if m<best[0]: best=(m,w)
print(f"  best: w={best[1]} 3A4={best[0]:.3f}")
# final macro: D-MPNN interval for 1A2/2C9/2D6, blend for 3A4
finals={}
for i,iso in enumerate(ISOS):
    if iso=="CYP3A4":
        bw=best[1]; pred=bw*dmpnn[:,j]+(1-bw)*lg3a4; finals[iso]=iso_strae(pred,iso)
    else:
        finals[iso]=iso_strae(dmpnn[:,i],iso)
print("\nFINAL per-isoform (D-MPNN interval; CYP3A4=blend):")
for iso in ISOS: print(f"  {iso}: {finals[iso][0]:.3f}±{finals[iso][1]:.3f}")
macro=np.mean([finals[i][0] for i in ISOS])
print(f"\nFINAL macro ST-RAE = {macro:.3f}")
print("refs: D-MPNN-only interval macro 0.434 | LightGBM 0.451 | leaderboard top 0.381 rank11 0.440")
