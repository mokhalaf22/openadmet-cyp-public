"""Predicted primary-screen feature (the tabular-FM lever, §25).

The single-concentration screen (log2FC) is test-UNAVAILABLE as a raw readout
(§16), but a model that PREDICTS it from structure is test-available. Here we:
  1. train a LightGBM log2FC predictor per enzyme on the single-conc screen,
  2. produce OUT-OF-FOLD predicted log2FC for our compounds (scaffold folds shared
     across datasets, so no scaffold leaks between the log2FC model and the pIC50
     eval),
  3. add those 4 predicted-log2FC columns as features to the pIC50 regression and
     compare OOF ST-RAE with vs without.

If the predicted-primary-screen feature helps here, it validates the piece before
layering CheMeleon + TabICL.
"""
import numpy as np, pandas as pd, lightgbm as lgb, torch
from cyp.features import featurize
from cyp.splits import murcko_scaffold, scaffold_folds
from cyp.losses import st_rae as st_rae_torch
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5; SEED=0
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv"); test=pd.read_csv(D+"cyp-challenge-TEST-BLINDED.csv")
sc=pd.read_csv(D+"cyp-challenge-single-concentration-TRAIN.csv")
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0

# single-conc pivoted to per-compound x 4 enzymes
scp=sc.pivot_table(index="SMILES",columns="enzyme",values="log2fc_estimate",aggfunc="mean").reindex(columns=ISOS)
sc_smiles=scp.index.tolist(); SCY=scp.to_numpy()

print("featurizing (ours + test + single-conc)...",flush=True)
Xo,oko,names=featurize(df["SMILES"].tolist())
var=Xo[oko].var(0); drop=(var<1e-8)|~np.isfinite(Xo[oko]).all(0); drop[[i for i,n in enumerate(names) if n=="Ipc"]]=True
keep=~drop
Xo=Xo[:,keep].astype(np.float32)
Xs,oks,_=featurize(sc_smiles); Xs=Xs[:,keep].astype(np.float32)
Xt,okt,_=featurize(test["SMILES"].tolist()); Xt=Xt[:,keep].astype(np.float32)

# shared scaffold->fold map across the union
union=list(dict.fromkeys(df["SMILES"].tolist()+test["SMILES"].tolist()+sc_smiles))
ufold,_=scaffold_folds(union,NF,SEED)
scaf2fold={}
for s,f in zip(union,ufold): scaf2fold.setdefault(murcko_scaffold(s),f)
def foldof(smis): return np.array([scaf2fold.get(murcko_scaffold(s),-1) for s in smis])
ofold=foldof(df["SMILES"].tolist()); sfold=foldof(sc_smiles)

def lgbm_reg(objective="regression"):
    return lgb.LGBMRegressor(objective=objective,n_estimators=500,learning_rate=0.03,num_leaves=31,
        min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=1.0,
        random_state=SEED,deterministic=True,force_row_wise=True,n_jobs=8,verbosity=-1)

# 1-2: OOF predicted log2FC for our compounds (per enzyme), scaffold-fold-aligned
print("training log2FC predictor (OOF, fold-aligned)...",flush=True)
pred_log2fc=np.full((len(df),4),np.nan)
for j,iso in enumerate(ISOS):
    ok_sc=oks & ~np.isnan(SCY[:,j])
    for f in range(NF):
        tr=np.where(ok_sc & (sfold!=f))[0]; va=np.where((ofold==f)&oko)[0]
        if len(tr)==0 or len(va)==0: continue
        m=lgbm_reg(); m.fit(Xs[tr],SCY[tr,j]); pred_log2fc[va,j]=m.predict(Xo[va])
# report log2FC model OOF quality on single-conc (sanity)
q=[]
for j,iso in enumerate(ISOS):
    ok_sc=oks&~np.isnan(SCY[:,j]); oofp=np.full(len(sc_smiles),np.nan)
    for f in range(NF):
        tr=np.where(ok_sc&(sfold!=f))[0]; va=np.where(ok_sc&(sfold==f))[0]
        if len(tr) and len(va): m=lgbm_reg(); m.fit(Xs[tr],SCY[tr,j]); oofp[va]=m.predict(Xs[va])
    mm=~np.isnan(oofp)&ok_sc; r=np.corrcoef(oofp[mm],SCY[mm,j])[0,1]; q.append(r)
    print(f"  log2FC {iso}: OOF Pearson r={r:.3f} (n={int(mm.sum())})")

# 3: pIC50 regression OOF, with vs without predicted log2FC feature
print("\npIC50 regression OOF ST-RAE (base = ECFP+desc; +ps = add 4 predicted log2FC):",flush=True)
print(f"  {'iso':7} {'base':>14} {'+pred-primary':>14}")
macro_b=[]; macro_p=[]
for j,iso in enumerate(ISOS):
    y=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float)
    lo=df[f"{iso}_pIC50_direct_inhibition_conf_low"].to_numpy(float); hi=df[f"{iso}_pIC50_direct_inhibition_conf_high"].to_numpy(float)
    w=hi-lo; w=np.where(np.isnan(w),np.nanmedian(w),w); wt=1.0/(1.0+w)
    pres=~np.isnan(y)&oko
    srb=[]; srp=[]
    for f in range(NF):
        va=np.where(pres&(ofold==f))[0]; tr=np.where(pres&(ofold!=f)&(ofold>=0))[0]
        if len(va)==0 or len(tr)==0: continue
        # base
        mb=lgbm_reg("regression_l1"); mb.fit(Xo[tr],y[tr],sample_weight=wt[tr]); pb=mb.predict(Xo[va])
        srb.append(strae(pb,lo[va],hi[va],y[va]))
        # + predicted primary screen (4 cols)
        Xtr=np.concatenate([Xo[tr],pred_log2fc[tr]],1); Xva=np.concatenate([Xo[va],pred_log2fc[va]],1)
        mp=lgbm_reg("regression_l1"); mp.fit(Xtr,y[tr],sample_weight=wt[tr]); pp=mp.predict(Xva)
        srp.append(strae(pp,lo[va],hi[va],y[va]))
    b,p=np.mean(srb),np.mean(srp); macro_b.append(b); macro_p.append(p)
    print(f"  {iso:7} {b:.3f}±{std1(srb):.3f}  {p:.3f}±{std1(srp):.3f}  ({p-b:+.3f})")
print(f"  macro   base={np.mean(macro_b):.3f}  +pred-primary={np.mean(macro_p):.3f}  ({np.mean(macro_p)-np.mean(macro_b):+.3f})")
print("\ndone")
