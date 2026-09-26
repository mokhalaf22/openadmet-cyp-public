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
# chemprop import is LAZY: importing it loads a second OpenMP runtime that
# deadlocks TabICL's multi-threaded torch attention (and segfaulted LightGBM).
# On a cached run chemprop is never imported, so that conflict never arises.
def chemeleon_embed(smiles, cache):
    if os.path.exists(cache): return np.load(cache)
    from chemprop.nn import BondMessagePassing, MeanAggregation
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer, MultiHotAtomFeaturizer
    from chemprop.data import BatchMolGraph
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

# ---- featurize + fold map (fold-aligned, reused from §25) — cached (stage survives kills) ----
scp=sc.pivot_table(index="SMILES",columns="enzyme",values="log2fc_estimate",aggfunc="mean").reindex(columns=ISOS)
sc_smiles=scp.index.tolist(); SCY=scp.to_numpy()
FEAT="experiments/tabfm_feat.npz"
if os.path.exists(FEAT):
    z=np.load(FEAT); Xo,Xs,Xt=z["Xo"],z["Xs"],z["Xt"]; oko,oks,okt=z["oko"],z["oks"],z["okt"]; ofold,sfold=z["ofold"],z["sfold"]
    print("featurize+folds: cached",flush=True)
else:
    print("featurize (ours + single-conc + test)...",flush=True)
    Xo,oko,names=featurize(df["SMILES"].tolist())
    var=Xo[oko].var(0); keep=~((var<1e-8)|~np.isfinite(Xo[oko]).all(0)); keep[[i for i,n in enumerate(names) if n=="Ipc"]]=False
    Xo=Xo[:,keep].astype(np.float32)
    Xs,oks,_=featurize(sc_smiles); Xs=Xs[:,keep].astype(np.float32)
    Xt,okt,_=featurize(test["SMILES"].tolist()); Xt=Xt[:,keep].astype(np.float32)
    union=list(dict.fromkeys(df["SMILES"].tolist()+test["SMILES"].tolist()+sc_smiles))
    ufold,_=scaffold_folds(union,NF,SEED); scaf2fold={}
    for s,f in zip(union,ufold): scaf2fold.setdefault(murcko_scaffold(s),f)
    def foldof(smis): return np.array([scaf2fold.get(murcko_scaffold(s),-1) for s in smis])
    ofold=foldof(df["SMILES"].tolist()); sfold=foldof(sc_smiles)
    np.savez(FEAT,Xo=Xo,Xs=Xs,Xt=Xt,oko=oko,oks=oks,okt=okt,ofold=ofold,sfold=sfold)
    print(f"saved {FEAT}",flush=True)
# n_jobs=1: multi-threaded LightGBM segfaults (duplicate OpenMP runtime) once chemprop/sklearn are imported in-process.
def lg(): return lgb.LGBMRegressor(n_estimators=500,learning_rate=0.03,num_leaves=31,min_child_samples=20,
    subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=1.0,random_state=SEED,deterministic=True,force_row_wise=True,n_jobs=1,verbosity=-1)
import time
def T(): return time.strftime("%H:%M:%S")

# ---- predicted-log2FC (fold-aligned) — cached (stage survives kills) ----
PLOG="experiments/tabfm_plog.npz"
if os.path.exists(PLOG):
    z=np.load(PLOG); plog_o,plog_t=z["o"],z["t"]; print(f"[{T()}] predicted-log2FC: cached",flush=True)
else:
    print(f"[{T()}] predicted-log2FC (OOF for train, full-model for test)...",flush=True)
    plog_o=np.full((len(df),4),np.nan); plog_t=np.full((len(test),4),np.nan)
    for j in range(4):
        ok_sc=oks&~np.isnan(SCY[:,j])
        for f in range(NF):
            tr=np.where(ok_sc&(sfold!=f))[0]; va=np.where((ofold==f)&oko)[0]
            if len(tr) and len(va): m=lg(); m.fit(Xs[tr],SCY[tr,j]); plog_o[va,j]=m.predict(Xo[va])
        trall=np.where(ok_sc)[0]; m=lg(); m.fit(Xs[trall],SCY[trall,j]); plog_t[:,j]=m.predict(Xt)
        print(f"[{T()}]   log2FC {ISOS[j]} done",flush=True)
    np.savez(PLOG,o=plog_o,t=plog_t); print(f"[{T()}] saved {PLOG}",flush=True)

# ---- PCA CheMeleon -> 256 (fit on train, unsupervised; fast, recomputed each run) ----
print(f"[{T()}] PCA 2048->256...",flush=True)
pca=PCA(n_components=256,random_state=SEED).fit(Eo[oko])
Po=pca.transform(Eo); Pt=pca.transform(Et)
Ftr=np.concatenate([Po,plog_o],1); Fte=np.concatenate([Pt,plog_t],1)   # [256 + 4]

# ---- TabICL per isoform, OOF + test — checkpointed per FIT (fold/full) ----
# Warm fits are ~40s; first fit pays a ~50s warmup. Per-fit checkpointing so a
# session teardown / erratic kill loses at most one ~40s fit.
from tabicl import TabICLRegressor
CK="experiments/tabfm_ck.npz"
if os.path.exists(CK):
    z=np.load(CK); oof_tab=z["oof"]; tab_test=z["test"]; oof_done=z["oof_done"]; test_done=z["test_done"]
else:
    oof_tab=np.full((len(df),4),np.nan); tab_test=np.full((len(test),4),np.nan)
    oof_done=np.zeros((4,NF),bool); test_done=np.zeros(4,bool)
def save_ck(): np.savez(CK,oof=oof_tab,test=tab_test,oof_done=oof_done,test_done=test_done)
print(f"[{T()}] TabICL OOF ST-RAE (CheMeleon256 + predicted-log2FC):",flush=True)
for j,iso in enumerate(ISOS):
    y=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float)
    lo=df[f"{iso}_pIC50_direct_inhibition_conf_low"].to_numpy(float); hi=df[f"{iso}_pIC50_direct_inhibition_conf_high"].to_numpy(float)
    pres=~np.isnan(y)&oko&~np.isnan(Ftr).any(1)
    for f in range(NF):
        if oof_done[j,f]: continue
        va=np.where(pres&(ofold==f))[0]; tr=np.where(pres&(ofold!=f)&(ofold>=0))[0]
        if len(va)==0 or len(tr)==0: oof_done[j,f]=True; continue
        r=TabICLRegressor(random_state=SEED); r.fit(Ftr[tr],y[tr]); oof_tab[va,j]=r.predict(Ftr[va])
        oof_done[j,f]=True; save_ck(); print(f"[{T()}]     {iso} fold{f} done",flush=True)
    sr=[]
    for f in range(NF):
        va=np.where(pres&(ofold==f))[0]
        if len(va): sr.append(strae(oof_tab[va,j],lo[va],hi[va],y[va]))
    print(f"[{T()}]   {iso}: {np.mean(sr):.3f}±{std1(sr):.3f}",flush=True)
# macro over per-isoform means (OOF)
macro=[]
for j,iso in enumerate(ISOS):
    y=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float)
    lo=df[f"{iso}_pIC50_direct_inhibition_conf_low"].to_numpy(float); hi=df[f"{iso}_pIC50_direct_inhibition_conf_high"].to_numpy(float)
    pres=~np.isnan(y)&oko&~np.isnan(Ftr).any(1); sr=[]
    for f in range(NF):
        va=np.where(pres&(ofold==f))[0]
        if len(va): sr.append(strae(oof_tab[va,j],lo[va],hi[va],y[va]))
    macro.append(np.mean(sr))
print(f"[{T()}]   macro TabICL = {np.mean(macro):.3f}   (vs ours LightGBM+predicted-primary 0.433, §25)")

# ---- full-model test predictions (per-isoform checkpoint) for the final submission ----
print(f"[{T()}] TabICL test predictions (full-model per isoform)...",flush=True)
for j,iso in enumerate(ISOS):
    if test_done[j]: continue
    y=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float); pres=~np.isnan(y)&oko&~np.isnan(Ftr).any(1)
    trall=np.where(pres)[0]; r=TabICLRegressor(random_state=SEED); r.fit(Ftr[trall],y[trall]); tab_test[:,j]=r.predict(Fte)
    test_done[j]=True; save_ck(); print(f"[{T()}]     {iso} test done",flush=True)
np.save("experiments/tabfm_oof.npy",oof_tab); np.save("experiments/tabfm_test.npy",tab_test)
print("saved tabfm_oof.npy, tabfm_test.npy\ndone")
