"""Tabular-FM pipeline (§26): [CheMeleon PCA-256 + predicted-log2FC(4)] -> TabICL.

Built as the OpenADMET post describes; validated ONCE on our scaffold folds with
default settings (no tuning toward these folds — our OOF is a weak guide, §22).
Compared like-for-like against our LightGBM + predicted-primary-screen (0.433, §25)
on the same folds. CheMeleon embeddings cached.
"""
import sys, os
import numpy as np, pandas as pd, torch, lightgbm as lgb
from rdkit import Chem, RDLogger
from sklearn.decomposition import PCA
from cyp.features import featurize
from cyp.splits import murcko_scaffold, scaffold_folds
from cyp.losses import st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5; SEED=0
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv"); test=pd.read_csv(D+"cyp-challenge-TEST-BLINDED.csv")
sc=pd.read_csv(D+"cyp-challenge-single-concentration-TRAIN.csv")
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0

# ---- CheMeleon embeddings (cached) ----
from chemprop.nn import BondMessagePassing, MeanAggregation
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer, MultiHotAtomFeaturizer
from chemprop.data import BatchMolGraph
def chemeleon_embed(smiles, cache):
    if os.path.exists(cache): return np.load(cache)
    ck=torch.load("data/chemeleon/chemeleon_mp.pt",weights_only=True)
    mp=BondMessagePassing(**ck["hyper_parameters"]); mp.load_state_dict(ck["state_dict"]); mp.eval()
    feat=SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v2()); agg=MeanAggregation()
    embs=np.zeros((len(smiles),2048),dtype=np.float32)
    idx=[i for i,s in enumerate(smiles) if isinstance(s,str) and Chem.MolFromSmiles(s)]
    B=256
    for k in range(0,len(idx),B):
        b=idx[k:k+B]; bmg=BatchMolGraph([feat(Chem.MolFromSmiles(smiles[i])) for i in b])
        with torch.no_grad(): e=agg(mp(bmg),bmg.batch).numpy()
        for r,i in enumerate(b): embs[i]=e[r]
        print(f"  chemeleon {cache.split('/')[-1]}: {k+len(b)}/{len(idx)}",flush=True)
    np.save(cache,embs); return embs
print("CheMeleon embeddings...",flush=True)
Eo=chemeleon_embed(df["SMILES"].tolist(),"experiments/chemeleon_train.npy")
Et=chemeleon_embed(test["SMILES"].tolist(),"experiments/chemeleon_test.npy")

# ---- predicted-log2FC (fold-aligned, reused approach from §25) ----
Xo,oko,names=featurize(df["SMILES"].tolist())
var=Xo[oko].var(0); keep=~((var<1e-8)|~np.isfinite(Xo[oko]).all(0)); keep[[i for i,n in enumerate(names) if n=="Ipc"]]=False
Xo=Xo[:,keep].astype(np.float32)
scp=sc.pivot_table(index="SMILES",columns="enzyme",values="log2fc_estimate",aggfunc="mean").reindex(columns=ISOS)
sc_smiles=scp.index.tolist(); SCY=scp.to_numpy()
Xs,oks,_=featurize(sc_smiles); Xs=Xs[:,keep].astype(np.float32)
Xt,okt,_=featurize(test["SMILES"].tolist()); Xt=Xt[:,keep].astype(np.float32)
union=list(dict.fromkeys(df["SMILES"].tolist()+test["SMILES"].tolist()+sc_smiles))
ufold,_=scaffold_folds(union,NF,SEED); scaf2fold={}
for s,f in zip(union,ufold): scaf2fold.setdefault(murcko_scaffold(s),f)
def foldof(smis): return np.array([scaf2fold.get(murcko_scaffold(s),-1) for s in smis])
ofold=foldof(df["SMILES"].tolist()); sfold=foldof(sc_smiles)
def lg(): return lgb.LGBMRegressor(n_estimators=500,learning_rate=0.03,num_leaves=31,min_child_samples=20,
    subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=1.0,random_state=SEED,deterministic=True,force_row_wise=True,n_jobs=8,verbosity=-1)
print("predicted-log2FC (OOF for train, full-model for test)...",flush=True)
plog_o=np.full((len(df),4),np.nan); plog_t=np.full((len(test),4),np.nan)
for j in range(4):
    ok_sc=oks&~np.isnan(SCY[:,j])
    for f in range(NF):
        tr=np.where(ok_sc&(sfold!=f))[0]; va=np.where((ofold==f)&oko)[0]
        if len(tr) and len(va): m=lg(); m.fit(Xs[tr],SCY[tr,j]); plog_o[va,j]=m.predict(Xo[va])
    trall=np.where(ok_sc)[0]; m=lg(); m.fit(Xs[trall],SCY[trall,j]); plog_t[:,j]=m.predict(Xt)

# ---- PCA CheMeleon -> 256 (fit on train, unsupervised) ----
pca=PCA(n_components=256,random_state=SEED).fit(Eo[oko])
Po=pca.transform(Eo); Pt=pca.transform(Et)
Ftr=np.concatenate([Po,plog_o],1); Fte=np.concatenate([Pt,plog_t],1)   # [256 + 4]

# ---- TabICL per isoform, OOF (single run, defaults) ----
from tabicl import TabICLRegressor
print("\nTabICL OOF ST-RAE (CheMeleon256 + predicted-log2FC):",flush=True)
oof_tab=np.full((len(df),4),np.nan); macro=[]
for j,iso in enumerate(ISOS):
    y=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float)
    lo=df[f"{iso}_pIC50_direct_inhibition_conf_low"].to_numpy(float); hi=df[f"{iso}_pIC50_direct_inhibition_conf_high"].to_numpy(float)
    pres=~np.isnan(y)&oko&~np.isnan(Ftr).any(1); sr=[]
    for f in range(NF):
        va=np.where(pres&(ofold==f))[0]; tr=np.where(pres&(ofold!=f)&(ofold>=0))[0]
        if len(va)==0 or len(tr)==0: continue
        r=TabICLRegressor(random_state=SEED); r.fit(Ftr[tr],y[tr]); pr=r.predict(Ftr[va])
        oof_tab[va,j]=pr; sr.append(strae(pr,lo[va],hi[va],y[va]))
    macro.append(np.mean(sr)); print(f"  {iso}: {np.mean(sr):.3f}±{std1(sr):.3f}",flush=True)
print(f"  macro TabICL = {np.mean(macro):.3f}   (vs ours LightGBM+predicted-primary 0.433, §25)")
np.save("experiments/tabfm_oof.npy",oof_tab)
# also predict test for the final submission if TabICL wins
tab_test=np.full((len(test),4),np.nan)
for j,iso in enumerate(ISOS):
    y=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float); pres=~np.isnan(y)&oko&~np.isnan(Ftr).any(1)
    tr=np.where(pres)[0]; r=TabICLRegressor(random_state=SEED); r.fit(Ftr[tr],y[tr]); tab_test[:,j]=r.predict(Fte)
np.save("experiments/tabfm_test.npy",tab_test)
print("saved tabfm_oof.npy, tabfm_test.npy\ndone")
