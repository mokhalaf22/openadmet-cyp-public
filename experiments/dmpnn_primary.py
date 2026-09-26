"""Test the predicted-primary-screen feature on the D-MPNN path (§26 follow-up, #3).

Our best regression base was the D-MPNN interval model (§17), but the
predicted-primary-screen feature (§25) was only ever tested on LightGBM. This adds
the 4 fold-aligned predicted-log2FC values (GFOLD-OOF, from plog_gfold.py) to the
D-MPNN by concatenating them onto the aggregated graph embedding before the trunk,
and retrains interval mode, 3-seed OOF on the SAME GFOLD folds as the cached base
(experiments/p2_interval_s{0,1,2}.npy), so base vs +primary compare like-for-like.

Per-seed OOF cached (experiments/p2_intervalprimary_s{s}.npy) so it resumes after a
kill. Reports D-MPNN {base,+primary}; the LightGBM {base,+primary}-on-GFOLD numbers
come from plog_gfold.py. Also reports the CYP3A4 blend (§17) for the winning base.
"""
import sys, copy
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from cyp.losses import interval_hinge, st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5
DH=200; MAXEP=300; PATIENCE=20; BS=512; IVFRAC=0.15
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0

z=np.load("experiments/plog_gfold.npz"); GFOLD=z["GFOLD"]; oko=z["oko"]; PLOG=z["plog_o"]  # GFOLD-OOF predicted-log2FC
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

class DMPNN(nn.Module):  # phase2 DMPNN + 4 static predicted-log2FC features after agg
    def __init__(s,d_h=DH,depth=3,hidden=256,drop=0.1,nfeat=4):
        super().__init__(); s.mp=BondMessagePassing(d_h=d_h,depth=depth); s.agg=MeanAggregation()
        s.trunk=nn.Sequential(nn.Linear(d_h+nfeat,hidden),nn.GELU(),nn.Dropout(drop))
        s.mu=nn.Linear(hidden,4); s.dr=nn.Linear(hidden,4)
    def forward(s,b,pf):
        g=s.agg(s.mp(b),b.batch); h=s.trunk(torch.cat([g,pf],1)); return s.mu(h), nn.functional.softplus(s.dr(h))

def macro_iv(pred_std,rows,ym,ys):
    P=pred_std*ys+ym; vals=[]
    for j in range(4):
        m=M[rows,j]
        if m.sum(): vals.append(strae(P[m,j],LO[rows,j][m],HI[rows,j][m],Y[rows,j][m]))
    return float(np.mean(vals))

def run_seed(seed):
    oof=np.full((len(df),4),np.nan)
    for f in range(NF):
        outer=np.where((GFOLD!=f)&(GFOLD>=0))[0]
        rng=np.random.RandomState(seed*100+f); iv=rng.permutation(len(outer))[:int(len(outer)*IVFRAC)]
        iv_rows=outer[iv]; tr=np.setdiff1d(outer,iv_rows); va=np.where(GFOLD==f)[0]
        ym=np.array([Y[tr,j][M[tr,j]].mean() for j in range(4)]); ys=np.array([Y[tr,j][M[tr,j]].std() or 1.0 for j in range(4)])
        lo=np.nan_to_num((LO-ym)/ys); hi=np.nan_to_num((HI-ym)/ys)
        # standardize predicted-log2FC on train-fold stats; impute missing -> 0 (the mean)
        pm=np.nanmean(PLOG[tr],0); ps=np.nanstd(PLOG[tr],0); ps=np.where(ps<1e-6,1.0,ps)
        PF=np.nan_to_num((PLOG-pm)/ps)
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
        print(f"  +primary s{seed} fold{f}: best@{best[2]+1}ep",flush=True)
    return oof

def per_iso(oof):
    out={}
    for j,iso in enumerate(ISOS):
        sr=[]
        for f in range(NF):
            r=np.where((GFOLD==f)&M[:,j])[0]
            if len(r): sr.append(strae(oof[r,j],LO[r,j],HI[r,j],Y[r,j]))
        out[iso]=[float(np.mean(sr)),std1(sr)]
    return out

if __name__=="__main__":
    oofs=[]
    for s in [0,1,2]:
        npy=f"experiments/p2_intervalprimary_s{s}.npy"
        try: oof=np.load(npy); print(f"+primary s{s}: cached",flush=True)
        except Exception: oof=run_seed(s); np.save(npy,oof); print(f"+primary s{s}: computed",flush=True)
        oofs.append(oof)
    ensp=np.nanmean(oofs,0); perp=per_iso(ensp)
    np.save("experiments/dmpnn_primary_oof.npy",ensp)
    # base D-MPNN interval on same GFOLD (cached from phase2)
    base=[np.load(f"experiments/p2_interval_s{s}.npy") for s in [0,1,2]]
    ensb=np.nanmean(base,0); perb=per_iso(ensb)
    print("\nD-MPNN interval OOF ST-RAE on GFOLD (base vs +predicted-primary):")
    for iso in ISOS: print(f"  {iso}: base={perb[iso][0]:.3f}±{perb[iso][1]:.3f}  +primary={perp[iso][0]:.3f}±{perp[iso][1]:.3f}  ({perp[iso][0]-perb[iso][0]:+.3f})")
    mb=np.mean([perb[i][0] for i in ISOS]); mp=np.mean([perp[i][0] for i in ISOS])
    print(f"  macro: base={mb:.3f}  +primary={mp:.3f}  ({mp-mb:+.3f})")
    print("done")
