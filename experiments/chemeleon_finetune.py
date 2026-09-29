"""§32: CheMeleon fine-tuned END-TO-END on the CYP task (the standard use we skipped).

We only tested CheMeleon frozen (PCA'd embeddings -> TabICL, §26). Here we initialize a
BondMessagePassing encoder from the CheMeleon checkpoint (data/chemeleon/chemeleon_mp.pt,
d_h=2048 depth=6, MultiHotAtomFeaturizer.v2) and fine-tune it end-to-end with our interval
loss, multi-task across the four isoforms, plus the 4 predicted-log2FC features concatenated
onto the aggregated embedding. Same GFOLD folds, 3-seed. Judged on OOF Spearman first,
ST-RAE second, vs D-MPNN+primary (0.6037 / 0.4146).

Per-(seed,fold) checkpointing to experiments/cheme_ft_s{seed}.npz (resumable). Set env
CHEME_MAXEP for a quick timing run. Prints per-epoch seconds for seed0/fold0.
"""
import sys, os, copy, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr
from cyp.losses import interval_hinge, st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5
MAXEP=int(os.environ.get("CHEME_MAXEP","80")); PATIENCE=10; BS=128; LR=2e-4; WD=1e-4
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0
z=np.load("experiments/plog_gfold.npz"); GFOLD=z["GFOLD"]; oko=z["oko"]; PLOG=z["plog_o"]; NFEAT=PLOG.shape[1]
valid=np.where(oko)[0]
Y=np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS],1)
LO=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS],1)
HI=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS],1); M=~np.isnan(Y)
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer, MultiHotAtomFeaturizer
from chemprop.data import BatchMolGraph
from chemprop.nn import BondMessagePassing, MeanAggregation
_feat=SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v2())
print("caching MolGraphs (V2 featurizer)...",flush=True)
MG={int(i):_feat(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])
_CK=torch.load("data/chemeleon/chemeleon_mp.pt",weights_only=True)

class CheMeleonFT(nn.Module):
    def __init__(s,nfeat=NFEAT,hidden=256,drop=0.1):
        super().__init__()
        s.mp=BondMessagePassing(**_CK["hyper_parameters"]); s.mp.load_state_dict(_CK["state_dict"])
        s.agg=MeanAggregation(); d_h=_CK["hyper_parameters"]["d_h"]
        s.trunk=nn.Sequential(nn.Linear(d_h+nfeat,hidden),nn.GELU(),nn.Dropout(drop)); s.mu=nn.Linear(hidden,4)
    def forward(s,b,pf): g=s.agg(s.mp(b),b.batch); return s.mu(s.trunk(torch.cat([g,pf],1)))

def macro_iv(mu_std,rows,ym,ys):
    P=mu_std*ys+ym; vals=[]
    for j in range(4):
        mm=M[rows,j]
        if mm.sum(): vals.append(strae(P[mm,j],LO[rows,j][mm],HI[rows,j][mm],Y[rows,j][mm]))
    return float(np.mean(vals))

def run_fold(seed,f,PF):
    outer=np.where((GFOLD!=f)&(GFOLD>=0))[0]
    rng=np.random.RandomState(seed*100+f); iv=rng.permutation(len(outer))[:int(len(outer)*0.15)]
    iv_rows=outer[iv]; tr=np.setdiff1d(outer,iv_rows); va=np.where(GFOLD==f)[0]
    ym=np.array([Y[tr,j][M[tr,j]].mean() for j in range(4)]); ys=np.array([Y[tr,j][M[tr,j]].std() or 1.0 for j in range(4)])
    lo=np.nan_to_num((LO-ym)/ys); hi=np.nan_to_num((HI-ym)/ys)
    def pf(rows): return torch.tensor(PF[rows],dtype=torch.float32)
    torch.manual_seed(seed); net=CheMeleonFT(); opt=torch.optim.Adam(net.parameters(),lr=LR,weight_decay=WD)
    n=len(tr); g=torch.Generator().manual_seed(seed); best=(1e9,None,0)
    for ep in range(MAXEP):
        t0=time.time(); net.train(); perm=torch.randperm(n,generator=g).numpy()
        for i in range(0,n,BS):
            b=tr[perm[i:i+BS]]; opt.zero_grad()
            mu=net(bmg(b),pf(b))
            loss=interval_hinge(mu,torch.tensor(lo[b],dtype=torch.float32),torch.tensor(hi[b],dtype=torch.float32),torch.tensor(M[b],dtype=torch.float32))
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): ivs=macro_iv(net(bmg(iv_rows),pf(iv_rows)).numpy(),iv_rows,ym,ys)
        if seed==0 and f==0: print(f"    ep{ep+1}: iv={ivs:.4f} ({time.time()-t0:.0f}s)",flush=True)
        if ivs<best[0]-1e-4: best=(ivs,copy.deepcopy(net.state_dict()),ep)
        elif ep-best[2]>=PATIENCE: break
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad(): pred=net(bmg(va),pf(va)).numpy()*ys+ym
    print(f"  s{seed} fold{f}: best@{best[2]+1}ep",flush=True)
    return va,pred

def run_seed(seed):
    ck=f"experiments/cheme_ft_s{seed}.npz"
    if os.path.exists(ck):
        z=np.load(ck); oof=z["oof"]; done=z["done"]
    else:
        oof=np.full((len(df),4),np.nan); done=np.zeros(NF,bool)
    pm=np.nanmean(PLOG[valid],0); ps=np.nanstd(PLOG[valid],0); ps=np.where(ps<1e-6,1.0,ps)
    PF=np.nan_to_num((PLOG-pm)/ps)
    for f in range(NF):
        if done[f]: continue
        va,pred=run_fold(seed,f,PF); oof[va]=pred; done[f]=True
        np.savez(ck,oof=oof,done=done)
    return oof

def per_iso(oof,metric):
    out={}
    for j,iso in enumerate(ISOS):
        if metric=="strae":
            sr=[strae(oof[np.where((GFOLD==f)&M[:,j])[0],j],LO[np.where((GFOLD==f)&M[:,j])[0],j],HI[np.where((GFOLD==f)&M[:,j])[0],j],Y[np.where((GFOLD==f)&M[:,j])[0],j]) for f in range(NF) if ((GFOLD==f)&M[:,j]).sum()]
            out[iso]=float(np.mean(sr))
        else:
            r=np.where((GFOLD>=0)&M[:,j]&~np.isnan(oof[:,j]))[0]; out[iso]=spearmanr(oof[r,j],Y[r,j]).correlation
    return out

if __name__=="__main__":
    ens=np.nanmean([run_seed(s) for s in [0,1,2]],0)
    np.save("experiments/chemeleon_finetune_oof.npy",ens)
    base=np.load("experiments/dmpnn_primary_oof.npy")
    sB,rB=per_iso(base,"strae"),per_iso(base,"sp"); sE,rE=per_iso(ens,"strae"),per_iso(ens,"sp")
    print("\nD-MPNN+primary  vs  CheMeleon-finetune+primary  (Spearman | ST-RAE), GFOLD OOF:")
    for iso in ISOS:
        print(f"  {iso}: rho {rB[iso]:.3f}->{rE[iso]:.3f} ({rE[iso]-rB[iso]:+.3f}) | strae {sB[iso]:.3f}->{sE[iso]:.3f} ({sE[iso]-sB[iso]:+.3f})")
    print(f"  macro: rho {np.mean(list(rB.values())):.4f}->{np.mean(list(rE.values())):.4f} ({np.mean(list(rE.values()))-np.mean(list(rB.values())):+.4f}) | "
          f"strae {np.mean(list(sB.values())):.4f}->{np.mean(list(sE.values())):.4f} ({np.mean(list(sE.values()))-np.mean(list(sB.values())):+.4f})")
    print("done")
