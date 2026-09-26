"""Predicted-log2FC aligned to the D-MPNN's own scaffold folds (GFOLD), plus the
LightGBM base / +predicted-primary OOF on those SAME folds (§26 follow-up, #3).

Why: §25's LightGBM+primary (0.433) and §17's D-MPNN blend (0.427) were validated
on DIFFERENT scaffold-fold partitions (tabfm union folds vs phase2 GFOLD), so they
are not strictly comparable, and the predicted-primary feature was never put on the
D-MPNN. This script rebuilds everything on GFOLD (phase2's exact folds) so the four
candidates — {LightGBM, D-MPNN} x {base, +primary} — compare like-for-like.

lightgbm-only (imports rdkit/numpy via cyp.features, NOT torch/chemprop), so the
OpenMP conflict (§26) does not arise and n_jobs=8 is safe. Caches:
  experiments/plog_gfold.npz : GFOLD, oko (validity), plog_o (GFOLD-OOF pred-log2FC
  for train), plog_t (full-model pred-log2FC for the 750 blinded), Xo/Xt (base feats)
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, torch
sys.path.insert(0,"src")
from cyp.features import featurize
from cyp.splits import scaffold_folds, murcko_scaffold
from cyp.losses import st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5; SEED=0
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv"); test=pd.read_csv(D+"cyp-challenge-TEST-BLINDED.csv")
sc=pd.read_csv(D+"cyp-challenge-single-concentration-TRAIN.csv")
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0

print("featurizing...",flush=True)
Xo,oko,names=featurize(df["SMILES"].tolist())
var=Xo[oko].var(0); keep=~((var<1e-8)|~np.isfinite(Xo[oko]).all(0)); keep[[i for i,n in enumerate(names) if n=="Ipc"]]=False
Xo=Xo[:,keep].astype(np.float32)
Xt,okt,_=featurize(test["SMILES"].tolist()); Xt=Xt[:,keep].astype(np.float32)
scp=sc.pivot_table(index="SMILES",columns="enzyme",values="log2fc_estimate",aggfunc="mean").reindex(columns=ISOS)
sc_smiles=scp.index.tolist(); SCY=scp.to_numpy()
Xs,oks,_=featurize(sc_smiles); Xs=Xs[:,keep].astype(np.float32)

# GFOLD: EXACTLY phase2's folds — scaffold_folds over the train valid rows, seed 0.
valid=np.where(oko)[0]
gf,_=scaffold_folds(df["SMILES"].to_numpy()[valid].tolist(),NF,0)
GFOLD=np.full(len(df),-1); GFOLD[valid]=gf
# map single-conc compounds onto GFOLD via their Murcko scaffold (leakage-safe:
# a scaffold in train fold f -> single-conc compounds of that scaffold are fold f)
scaf2fold={}
for i in valid: scaf2fold.setdefault(murcko_scaffold(df["SMILES"].iloc[int(i)]),GFOLD[i])
sfold=np.array([scaf2fold.get(murcko_scaffold(s),-1) for s in sc_smiles])

def lg(obj="regression"): return lgb.LGBMRegressor(objective=obj,n_estimators=500,learning_rate=0.03,num_leaves=31,
    min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=1.0,random_state=SEED,
    deterministic=True,force_row_wise=True,n_jobs=8,verbosity=-1)

print("predicted-log2FC OOF aligned to GFOLD...",flush=True)
plog_o=np.full((len(df),4),np.nan); plog_t=np.full((len(test),4),np.nan)
for j in range(4):
    ok_sc=oks&~np.isnan(SCY[:,j])
    for f in range(NF):
        tr=np.where(ok_sc&(sfold!=f)&(sfold>=0))[0]; va=np.where((GFOLD==f)&oko)[0]
        if len(tr) and len(va): m=lg(); m.fit(Xs[tr],SCY[tr,j]); plog_o[va,j]=m.predict(Xo[va])
    trall=np.where(ok_sc)[0]; m=lg(); m.fit(Xs[trall],SCY[trall,j]); plog_t[:,j]=m.predict(Xt)

np.savez("experiments/plog_gfold.npz",GFOLD=GFOLD,oko=oko,okt=okt,plog_o=plog_o,plog_t=plog_t,Xo=Xo,Xt=Xt)
print("saved experiments/plog_gfold.npz",flush=True)

# LightGBM base vs +primary on GFOLD (matches §25 recipe: regression_l1, width-weighted)
print("\nLightGBM OOF ST-RAE on GFOLD (base vs +predicted-primary):",flush=True)
mb=[]; mp=[]
for j,iso in enumerate(ISOS):
    y=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float)
    lo=df[f"{iso}_pIC50_direct_inhibition_conf_low"].to_numpy(float); hi=df[f"{iso}_pIC50_direct_inhibition_conf_high"].to_numpy(float)
    w=hi-lo; w=np.where(np.isnan(w),np.nanmedian(w),w); wt=1.0/(1.0+w)
    pres=~np.isnan(y)&oko; srb=[]; srp=[]
    for f in range(NF):
        va=np.where(pres&(GFOLD==f))[0]; tr=np.where(pres&(GFOLD!=f)&(GFOLD>=0))[0]
        if len(va)==0 or len(tr)==0: continue
        m=lg("regression_l1"); m.fit(Xo[tr],y[tr],sample_weight=wt[tr]); srb.append(strae(m.predict(Xo[va]),lo[va],hi[va],y[va]))
        Xtr=np.concatenate([Xo[tr],plog_o[tr]],1); Xva=np.concatenate([Xo[va],plog_o[va]],1)
        m=lg("regression_l1"); m.fit(Xtr,y[tr],sample_weight=wt[tr]); srp.append(strae(m.predict(Xva),lo[va],hi[va],y[va]))
    b,p=np.mean(srb),np.mean(srp); mb.append(b); mp.append(p)
    print(f"  {iso}: base={b:.3f}  +primary={p:.3f}  ({p-b:+.3f})",flush=True)
print(f"  macro: base={np.mean(mb):.3f}  +primary={np.mean(mp):.3f}  ({np.mean(mp)-np.mean(mb):+.3f})",flush=True)
print("done")
