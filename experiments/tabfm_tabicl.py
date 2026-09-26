"""TabICL stage of the tabular-FM pipeline (§26), as a standalone process.

Split out from experiments/tabfm.py on purpose: importing lightgbm (and chemprop)
loads a second OpenMP runtime that deadlocks TabICL's multi-threaded torch
attention. This script imports ONLY numpy/pandas/torch/sklearn/tabicl (+ the
torch-only cyp.losses), so TabICL runs on a clean torch and fits are fast
(~40s warm). All upstream features are read from the caches that tabfm.py wrote:
  experiments/chemeleon_train.npy / _test.npy   (CheMeleon 2048-d embeddings)
  experiments/tabfm_feat.npz                    (validity mask oko, scaffold folds ofold)
  experiments/tabfm_plog.npz                    (fold-aligned predicted-log2FC, 4 cols)

Validated ONCE on our scaffold folds with TabICL defaults (no tuning toward
these folds). Per-FIT checkpointing to experiments/tabfm_ck.npz so a kill loses
at most one ~40s fit. Compared like-for-like to LightGBM+predicted-primary 0.433 (§25).
"""
import sys, os
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from cyp.losses import st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5; SEED=0
# TabICL ensemble size. The default (n_estimators=8) does not complete on this
# 24 GB M4 Pro (stalls at ~12% CPU whether offload is 'auto' or False); 4 completes
# reliably (~20s warm). This is a global compute/hardware setting chosen a priori
# and applied uniformly to every isoform and fold — NOT tuning toward our folds.
# TabICL's members are random feature-permutation replicas, so 4 vs 8 is a small
# variance difference, not a different method. Disclosed in FINDINGS §26 / REPORT.
N_EST=4
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv"); test=pd.read_csv(D+"cyp-challenge-TEST-BLINDED.csv")
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0
def T(): import time; return time.strftime("%H:%M:%S")

# ---- load cached upstream features ----
z=np.load("experiments/tabfm_feat.npz"); oko=z["oko"]; ofold=z["ofold"]
zp=np.load("experiments/tabfm_plog.npz"); plog_o=zp["o"]; plog_t=zp["t"]
Eo=np.load("experiments/chemeleon_train.npy"); Et=np.load("experiments/chemeleon_test.npy")
print(f"[{T()}] PCA 2048->256 (fit on train, unsupervised)...",flush=True)
pca=PCA(n_components=256,random_state=SEED).fit(Eo[oko])
Ftr=np.concatenate([pca.transform(Eo),plog_o],1); Fte=np.concatenate([pca.transform(Et),plog_t],1)  # [256+4]

# ---- TabICL, per-fit checkpointed ----
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
        r=TabICLRegressor(random_state=SEED,n_estimators=N_EST); r.fit(Ftr[tr],y[tr]); oof_tab[va,j]=r.predict(Ftr[va])
        oof_done[j,f]=True; save_ck(); print(f"[{T()}]     {iso} fold{f} done",flush=True)
    sr=[]
    for f in range(NF):
        va=np.where(pres&(ofold==f))[0]
        if len(va): sr.append(strae(oof_tab[va,j],lo[va],hi[va],y[va]))
    print(f"[{T()}]   {iso}: {np.mean(sr):.3f}±{std1(sr):.3f}",flush=True)

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

# ---- full-model test predictions (per-isoform checkpoint) ----
print(f"[{T()}] TabICL test predictions (full-model per isoform)...",flush=True)
for j,iso in enumerate(ISOS):
    if test_done[j]: continue
    y=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float); pres=~np.isnan(y)&oko&~np.isnan(Ftr).any(1)
    trall=np.where(pres)[0]; r=TabICLRegressor(random_state=SEED,n_estimators=N_EST); r.fit(Ftr[trall],y[trall]); tab_test[:,j]=r.predict(Fte)
    test_done[j]=True; save_ck(); print(f"[{T()}]     {iso} test done",flush=True)
np.save("experiments/tabfm_oof.npy",oof_tab); np.save("experiments/tabfm_test.npy",tab_test)
print("saved tabfm_oof.npy, tabfm_test.npy\ndone")
