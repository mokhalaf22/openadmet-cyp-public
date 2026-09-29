"""Direction 1 (§31a) part B: D-MPNN + predicted-primary + predicted-Emax.

Add the 8 GFOLD-OOF predicted-Emax features (emax_gfold.npz) alongside the 4
predicted-log2FC (plog_gfold.npz) -> 12 static features concatenated onto the
aggregated graph embedding. Same architecture/folds as dmpnn_primary.py (0.415
macro ST-RAE, the +primary baseline), interval loss, 3-seed OOF on GFOLD. Judged
on OOF Spearman FIRST, ST-RAE second (§31). Per-seed cached (resumable).
"""
import sys, copy, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr
from cyp.losses import interval_hinge, st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5
DH=200; MAXEP=300; PATIENCE=20; BS=512
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0
z=np.load("experiments/plog_gfold.npz"); GFOLD=z["GFOLD"]; oko=z["oko"]; PLOG=z["plog_o"]
PEMAX=np.load("experiments/emax_gfold.npz")["pemax_o"]
FEAT=np.concatenate([PLOG,PEMAX],1); NFEAT=FEAT.shape[1]  # 4 + 8 = 12
valid=np.where(oko)[0]
Y=np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS],1)
LO=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS],1)
HI=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS],1); M=~np.isnan(Y)
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.data import BatchMolGraph
from chemprop.nn import BondMessagePassing, MeanAggregation
_feat=SimpleMoleculeMolGraphFeaturizer()
print(f"caching MolGraphs... (NFEAT={NFEAT})",flush=True)
MG={int(i):_feat(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])

class DMPNN(nn.Module):
    def __init__(s,d_h=DH,depth=3,hidden=256,drop=0.1,nfeat=NFEAT):
        super().__init__(); s.mp=BondMessagePassing(d_h=d_h,depth=depth); s.agg=MeanAggregation()
        s.trunk=nn.Sequential(nn.Linear(d_h+nfeat,hidden),nn.GELU(),nn.Dropout(drop))
        s.mu=nn.Linear(hidden,4); s.dr=nn.Linear(hidden,4)
    def forward(s,b,pf): g=s.agg(s.mp(b),b.batch); h=s.trunk(torch.cat([g,pf],1)); return s.mu(h), nn.functional.softplus(s.dr(h))

def macro_iv(pred_std,rows,ym,ys):
    P=pred_std*ys+ym; vals=[]
    for j in range(4):
        mm=M[rows,j]
        if mm.sum(): vals.append(strae(P[mm,j],LO[rows,j][mm],HI[rows,j][mm],Y[rows,j][mm]))
    return float(np.mean(vals))

def run_seed(seed):
    npy=f"experiments/p2_primaryemax_s{seed}.npy"
    if os.path.exists(npy): print(f"+pe s{seed}: cached",flush=True); return np.load(npy)
    oof=np.full((len(df),4),np.nan)
    for f in range(NF):
        outer=np.where((GFOLD!=f)&(GFOLD>=0))[0]
        rng=np.random.RandomState(seed*100+f); iv=rng.permutation(len(outer))[:int(len(outer)*0.15)]
        iv_rows=outer[iv]; tr=np.setdiff1d(outer,iv_rows); va=np.where(GFOLD==f)[0]
        ym=np.array([Y[tr,j][M[tr,j]].mean() for j in range(4)]); ys=np.array([Y[tr,j][M[tr,j]].std() or 1.0 for j in range(4)])
        lo=np.nan_to_num((LO-ym)/ys); hi=np.nan_to_num((HI-ym)/ys)
        pm=np.nanmean(FEAT[tr],0); ps=np.nanstd(FEAT[tr],0); ps=np.where(ps<1e-6,1.0,ps)
        PF=np.nan_to_num((FEAT-pm)/ps)
        def pf(rows): return torch.tensor(PF[rows],dtype=torch.float32)
        torch.manual_seed(seed); m=DMPNN(); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        n=len(tr); g=torch.Generator().manual_seed(seed); best=(1e9,None,0)
        for ep in range(MAXEP):
            m.train(); perm=torch.randperm(n,generator=g).numpy()
            for i in range(0,n,BS):
                b=tr[perm[i:i+BS]]; opt.zero_grad(); mu,_=m(bmg(b),pf(b))
                loss=interval_hinge(mu,torch.tensor(lo[b],dtype=torch.float32),torch.tensor(hi[b],dtype=torch.float32),torch.tensor(M[b],dtype=torch.float32))
                loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): iv_s=macro_iv(m(bmg(iv_rows),pf(iv_rows))[0].numpy(),iv_rows,ym,ys)
            if iv_s<best[0]-1e-4: best=(iv_s,copy.deepcopy(m.state_dict()),ep)
            elif ep-best[2]>=PATIENCE: break
        m.load_state_dict(best[1]); m.eval()
        with torch.no_grad(): oof[va]=m(bmg(va),pf(va))[0].numpy()*ys+ym
        print(f"  +pe s{seed} fold{f}: best@{best[2]+1}ep",flush=True)
    np.save(npy,oof); return oof

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
    np.save("experiments/dmpnn_primary_emax_oof.npy",ens)
    base=np.load("experiments/dmpnn_primary_oof.npy")
    sB,rB=per_iso(base,"strae"),per_iso(base,"sp"); sE,rE=per_iso(ens,"strae"),per_iso(ens,"sp")
    print("\nD-MPNN +primary  vs  +primary+Emax  (Spearman | ST-RAE), GFOLD OOF:")
    for iso in ISOS:
        print(f"  {iso}: rho {rB[iso]:.3f}->{rE[iso]:.3f} ({rE[iso]-rB[iso]:+.3f}) | strae {sB[iso]:.3f}->{sE[iso]:.3f} ({sE[iso]-sB[iso]:+.3f})")
    print(f"  macro: rho {np.mean(list(rB.values())):.4f}->{np.mean(list(rE.values())):.4f} ({np.mean(list(rE.values()))-np.mean(list(rB.values())):+.4f}) | "
          f"strae {np.mean(list(sB.values())):.4f}->{np.mean(list(sE.values())):.4f} ({np.mean(list(sE.values()))-np.mean(list(sB.values())):+.4f})")
    print("done")
