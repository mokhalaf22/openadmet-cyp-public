"""Direction 2 (§31b): cross-model blend sweep with the +primary variants of both
models, judged on Spearman first, ST-RAE second.

The CYP3A4 D-MPNN/LightGBM blend beat both components before the primary-screen
feature existed (§17); dropped when it bought only 0.001 macro ST-RAE (§26a). Re-run
across all four isoforms with the current +primary OOF of both models. Different
inductive biases making different errors is one of the few things that reliably
improves *ranking*, so report the Spearman effect explicitly.

Cached OOF on GFOLD: dmpnn_primary_oof.npy, lgbm_primary_oof.npy.
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,"src")
from cyp.losses import st_rae as st_rae_torch
import torch
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
z=np.load("experiments/plog_gfold.npz"); GFOLD=z["GFOLD"]; oko=z["oko"]
dm=np.load("experiments/dmpnn_primary_oof.npy"); lg=np.load("experiments/lgbm_primary_oof.npy")
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
Y=np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS],1)
LO=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS],1)
HI=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS],1); M=~np.isnan(Y)

def iso_metrics(pred,j):
    # ST-RAE = per-fold mean; Spearman = pooled OOF present rows
    sr=[]
    for f in range(NF):
        r=np.where((GFOLD==f)&M[:,j]&~np.isnan(pred))[0]
        if len(r): sr.append(strae(pred[r,j] if pred.ndim>1 else pred[r],LO[r,j],HI[r,j],Y[r,j]))
    pool=np.where((GFOLD>=0)&M[:,j]&~np.isnan(pred if pred.ndim==1 else pred[:,j]))[0]
    col=pred[pool] if pred.ndim==1 else pred[pool,j]
    rho=spearmanr(col,Y[pool,j]).correlation
    return float(np.mean(sr)), rho

print("component OOF (GFOLD):  Spearman | ST-RAE  per isoform")
for j,iso in enumerate(ISOS):
    sd,rd=iso_metrics(dm[:,j],j); sl,rl=iso_metrics(lg[:,j],j)
    print(f"  {iso}: D-MPNN rho={rd:.3f} strae={sd:.3f} | LGBM rho={rl:.3f} strae={sl:.3f} | corr(dm,lg)={spearmanr(dm[np.where((GFOLD>=0)&M[:,j])[0],j],lg[np.where((GFOLD>=0)&M[:,j])[0],j]).correlation:.3f}")

ws=[0.0,0.25,0.5,0.75,1.0]
print("\nblend sweep (w = D-MPNN weight):  Spearman [ST-RAE]")
best_sp={}; macro_by_w={w:{"sp":[],"sr":[]} for w in ws}
for j,iso in enumerate(ISOS):
    cells=[]; bw=(None,-9)
    for w in ws:
        bl=w*dm[:,j]+(1-w)*lg[:,j]; sr,rho=iso_metrics(bl,j)
        cells.append(f"w={w}: {rho:.3f}[{sr:.3f}]"); macro_by_w[w]["sp"].append(rho); macro_by_w[w]["sr"].append(sr)
        if rho>bw[1]: bw=(w,rho)
    best_sp[iso]=bw
    print(f"  {iso}: "+"  ".join(cells)+f"   -> best Spearman w={bw[0]} ({bw[1]:.3f})")
print("\nmacro by w:  Spearman | ST-RAE")
for w in ws:
    print(f"  w={w}: rho={np.mean(macro_by_w[w]['sp']):.4f}  strae={np.mean(macro_by_w[w]['sr']):.4f}")
# per-isoform best-Spearman blend, macro
psp=np.mean([best_sp[i][1] for i in ISOS]);
print(f"\nper-isoform best-Spearman blend: macro Spearman={psp:.4f} (vs D-MPNN-only {np.mean(macro_by_w[1.0]['sp']):.4f})")
print("done")
