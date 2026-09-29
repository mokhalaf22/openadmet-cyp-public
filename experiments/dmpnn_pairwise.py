"""§33a: add a pairwise margin RANKING term to the D-MPNN+primary interval loss.

total = interval_hinge + lambda * pairwise_margin, where pairwise_margin ranks all
within-batch, same-isoform ordered pairs on the model's mu output (standardized space,
monotone within isoform). Optimize ordering directly instead of hoping it follows from
absolute error. Sweep lambda in {0.1, 0.5, 1.0}. Same architecture/folds/features as
dmpnn_primary.py (D-MPNN + 4 predicted-log2FC, GFOLD 3-seed). Judged on OOF Spearman
first, ST-RAE second, vs D-MPNN+primary (0.6037 / 0.4146). Per-(lambda,seed) cached.
"""
import sys, copy, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr
from cyp.losses import interval_hinge, st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5
DH=200; MAXEP=300; PATIENCE=20; BS=512; MARGIN=0.1
LAMBDAS=[0.1,0.5,1.0]
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
z=np.load("experiments/plog_gfold.npz"); GFOLD=z["GFOLD"]; oko=z["oko"]; PLOG=z["plog_o"]; NFEAT=PLOG.shape[1]
valid=np.where(oko)[0]
Y=np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS],1)
LO=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS],1)
HI=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS],1); M=~np.isnan(Y)
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.data import BatchMolGraph
from chemprop.nn import BondMessagePassing, MeanAggregation
_feat=SimpleMoleculeMolGraphFeaturizer()
print("caching MolGraphs...",flush=True)
MG={int(i):_feat(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])

class DMPNN(nn.Module):
    def __init__(s,d_h=DH,depth=3,hidden=256,drop=0.1,nfeat=NFEAT):
        super().__init__(); s.mp=BondMessagePassing(d_h=d_h,depth=depth); s.agg=MeanAggregation()
        s.trunk=nn.Sequential(nn.Linear(d_h+nfeat,hidden),nn.GELU(),nn.Dropout(drop)); s.mu=nn.Linear(hidden,4)
    def forward(s,b,pf): g=s.agg(s.mp(b),b.batch); return s.mu(s.trunk(torch.cat([g,pf],1)))

def pairwise_margin(mu,tgt,mask,margin=MARGIN):
    tot=mu.new_tensor(0.0); npair=mu.new_tensor(0.0)
    for j in range(4):
        m=mask[:,j].bool()
        if int(m.sum())<2: continue
        p=mu[m,j]; t=tgt[m,j]
        dp=p.unsqueeze(0)-p.unsqueeze(1); dt=t.unsqueeze(0)-t.unsqueeze(1)
        w=(dt.abs()>1e-6).float(); sign=torch.sign(dt)
        tot=tot+(torch.relu(margin-sign*dp)*w).sum(); npair=npair+w.sum()
    return tot/npair.clamp(min=1.0)

def macro_iv(mu_std,rows,ym,ys):
    P=mu_std*ys+ym; vals=[]
    for j in range(4):
        mm=M[rows,j]
        if mm.sum(): vals.append(strae(P[mm,j],LO[rows,j][mm],HI[rows,j][mm],Y[rows,j][mm]))
    return float(np.mean(vals))

def run_seed(lam,seed,PF):
    tag=str(lam).replace('.','p'); npy=f"experiments/pw_l{tag}_s{seed}.npy"
    if os.path.exists(npy): print(f"  lam{lam} s{seed}: cached",flush=True); return np.load(npy)
    oof=np.full((len(df),4),np.nan)
    for f in range(NF):
        outer=np.where((GFOLD!=f)&(GFOLD>=0))[0]
        rng=np.random.RandomState(seed*100+f); iv=rng.permutation(len(outer))[:int(len(outer)*0.15)]
        iv_rows=outer[iv]; tr=np.setdiff1d(outer,iv_rows); va=np.where(GFOLD==f)[0]
        ym=np.array([Y[tr,j][M[tr,j]].mean() for j in range(4)]); ys=np.array([Y[tr,j][M[tr,j]].std() or 1.0 for j in range(4)])
        tgt=np.nan_to_num((Y-ym)/ys); lo=np.nan_to_num((LO-ym)/ys); hi=np.nan_to_num((HI-ym)/ys)
        torch.manual_seed(seed); m=DMPNN(); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        n=len(tr); g=torch.Generator().manual_seed(seed); best=(1e9,None,0)
        for ep in range(MAXEP):
            m.train(); perm=torch.randperm(n,generator=g).numpy()
            for i in range(0,n,BS):
                b=tr[perm[i:i+BS]]; opt.zero_grad(); mu=m(bmg(b),torch.tensor(PF[b],dtype=torch.float32))
                mk=torch.tensor(M[b],dtype=torch.float32)
                ih=interval_hinge(mu,torch.tensor(lo[b],dtype=torch.float32),torch.tensor(hi[b],dtype=torch.float32),mk)
                pw=pairwise_margin(mu,torch.tensor(tgt[b],dtype=torch.float32),mk)
                (ih+lam*pw).backward(); opt.step()
            m.eval()
            with torch.no_grad(): ivs=macro_iv(m(bmg(iv_rows),torch.tensor(PF[iv_rows],dtype=torch.float32)).numpy(),iv_rows,ym,ys)
            if ivs<best[0]-1e-4: best=(ivs,copy.deepcopy(m.state_dict()),ep)
            elif ep-best[2]>=PATIENCE: break
        m.load_state_dict(best[1]); m.eval()
        with torch.no_grad(): oof[va]=m(bmg(va),torch.tensor(PF[va],dtype=torch.float32)).numpy()*ys+ym
        print(f"  lam{lam} s{seed} fold{f}: best@{best[2]+1}ep",flush=True)
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
    pm=np.nanmean(PLOG[valid],0); ps=np.nanstd(PLOG[valid],0); ps=np.where(ps<1e-6,1.0,ps)
    PF=np.nan_to_num((PLOG-pm)/ps)
    base=np.load("experiments/dmpnn_primary_oof.npy"); rB=per_iso(base,"sp"); sB=per_iso(base,"strae")
    print(f"baseline D-MPNN+primary: macro Spearman={np.mean(list(rB.values())):.4f} ST-RAE={np.mean(list(sB.values())):.4f}")
    for lam in LAMBDAS:
        ens=np.nanmean([run_seed(lam,s,PF) for s in [0,1,2]],0)
        np.save(f"experiments/pw_l{str(lam).replace('.','p')}_oof.npy",ens)
        rE=per_iso(ens,"sp"); sE=per_iso(ens,"strae")
        print(f"\nlambda={lam}:")
        for iso in ISOS: print(f"  {iso}: rho {rB[iso]:.3f}->{rE[iso]:.3f} ({rE[iso]-rB[iso]:+.3f}) | strae {sB[iso]:.3f}->{sE[iso]:.3f} ({sE[iso]-sB[iso]:+.3f})")
        print(f"  macro: rho {np.mean(list(rB.values())):.4f}->{np.mean(list(rE.values())):.4f} ({np.mean(list(rE.values()))-np.mean(list(rB.values())):+.4f}) | strae {np.mean(list(sB.values())):.4f}->{np.mean(list(sE.values())):.4f} ({np.mean(list(sE.values()))-np.mean(list(sB.values())):+.4f})")
    print("done")
