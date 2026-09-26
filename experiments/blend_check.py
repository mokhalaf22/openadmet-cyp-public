"""CYP3A4 blend check + LightGBM+primary blinded predictions (§26 #3, final step).

Winner base is D-MPNN+primary (0.415 macro OOF, GFOLD). §17 found a 50/50
D-MPNN/LightGBM blend helped CYP3A4; re-test that with the +primary variants on
GFOLD, and save LightGBM+primary OOF + blinded test preds (needed for any blend
and as a fallback). lightgbm-only — no torch/chemprop, so no OpenMP conflict.
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, torch
sys.path.insert(0,"src")
from cyp.losses import st_rae as st_rae_torch
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5; SEED=0
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv"); test=pd.read_csv(D+"cyp-challenge-TEST-BLINDED.csv")
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0
z=np.load("experiments/plog_gfold.npz"); GFOLD=z["GFOLD"]; oko=z["oko"]; okt=z["okt"]
Xo=z["Xo"]; Xt=z["Xt"]; plog_o=z["plog_o"]; plog_t=z["plog_t"]
Ftr=np.concatenate([Xo,plog_o],1); Fte=np.concatenate([Xt,plog_t],1)
dmpnn=np.load("experiments/dmpnn_primary_oof.npy")  # D-MPNN+primary OOF (GFOLD)

def lg(): return lgb.LGBMRegressor(objective="regression_l1",n_estimators=500,learning_rate=0.03,num_leaves=31,
    min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=1.0,random_state=SEED,
    deterministic=True,force_row_wise=True,n_jobs=8,verbosity=-1)

# LightGBM+primary OOF (GFOLD) + blinded test preds
lg_oof=np.full((len(df),4),np.nan); lg_test=np.full((len(test),4),np.nan)
for j,iso in enumerate(ISOS):
    y=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float)
    lo=df[f"{iso}_pIC50_direct_inhibition_conf_low"].to_numpy(float); hi=df[f"{iso}_pIC50_direct_inhibition_conf_high"].to_numpy(float)
    w=hi-lo; w=np.where(np.isnan(w),np.nanmedian(w),w); wt=1.0/(1.0+w)
    pres=~np.isnan(y)&oko
    for f in range(NF):
        va=np.where(pres&(GFOLD==f))[0]; tr=np.where(pres&(GFOLD!=f)&(GFOLD>=0))[0]
        if len(va) and len(tr): m=lg(); m.fit(Ftr[tr],y[tr],sample_weight=wt[tr]); lg_oof[va,j]=m.predict(Ftr[va])
    trall=np.where(pres)[0]; m=lg(); m.fit(Ftr[trall],y[trall],sample_weight=wt[trall]); lg_test[:,j]=m.predict(Fte)
np.save("experiments/lgbm_primary_oof.npy",lg_oof); np.save("experiments/lgbm_primary_test.npy",lg_test)

Y=np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS],1)
LO=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS],1)
HI=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS],1); M=~np.isnan(Y)
def iso_strae(pred,j):
    sr=[]
    for f in range(NF):
        r=np.where((GFOLD==f)&M[:,j])[0]
        if len(r): sr.append(strae(pred[r,j],LO[r,j],HI[r,j],Y[r,j]))
    return np.mean(sr),std1(sr)

print("CYP3A4 blend sweep (w = D-MPNN+primary weight, OOF ST-RAE):")
j=3; best=(1e9,None)
for w in [0.0,0.25,0.5,0.75,1.0]:
    bl=w*dmpnn[:,j]+(1-w)*lg_oof[:,j]; s,_=iso_strae(np.stack([bl]*4,1),j)
    print(f"  w={w:.2f}: CYP3A4={s:.3f}");
    if s<best[0]: best=(s,w)
print(f"  best CYP3A4: w={best[1]} -> {best[0]:.3f}")

# macro with D-MPNN+primary everywhere vs D-MPNN+primary + best CYP3A4 blend
perd={i:iso_strae(dmpnn,j)[0] for j,i in enumerate(ISOS)}
macro_dmpnn=np.mean(list(perd.values()))
blendc=best[1]*dmpnn[:,3]+(1-best[1])*lg_oof[:,3]
mix=dmpnn.copy(); mix[:,3]=blendc
macro_mix=np.mean([iso_strae(mix,j)[0] for j in range(4)])
print(f"\nmacro D-MPNN+primary (all iso) = {macro_dmpnn:.3f}")
print(f"macro D-MPNN+primary + CYP3A4 blend(w={best[1]}) = {macro_mix:.3f}")
print("done")
