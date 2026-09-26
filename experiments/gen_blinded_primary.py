"""Blinded predictions from the winning regression model: D-MPNN interval +
predicted-primary-screen feature (§26 #3; 0.415 macro OOF on GFOLD, the best base).

Trains the D-MPNN+primary on ALL training rows (15% inner-val for early stopping),
3-seed ensemble, predicts the 750 blinded compounds. Mirrors gen_blinded.py but
concatenates the 4 predicted-log2FC features (GFOLD-OOF plog_o on train, full-model
plog_t on test, from plog_gfold.npz) onto the aggregated graph embedding — the exact
feature/architecture validated in dmpnn_primary.py. Per-seed cached (resumable).
"""
import sys, copy, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from cyp.losses import interval_hinge, st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]
DH=200; MAXEP=300; PATIENCE=20; BS=512; IVFRAC=0.15
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv"); test=pd.read_csv(D+"cyp-challenge-TEST-BLINDED.csv")
z=np.load("experiments/plog_gfold.npz"); oko=z["oko"]; okt=z["okt"]; PLOGo=z["plog_o"]; PLOGt=z["plog_t"]
Y=np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS],1)
LO=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS],1)
HI=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS],1); M=~np.isnan(Y)
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.data import BatchMolGraph
from chemprop.nn import BondMessagePassing, MeanAggregation
_feat=SimpleMoleculeMolGraphFeaturizer()
print("caching graphs...",flush=True)
MG=[_feat(Chem.MolFromSmiles(s)) if isinstance(s,str) and Chem.MolFromSmiles(s) else None for s in df["SMILES"]]
TG=[_feat(Chem.MolFromSmiles(s)) if isinstance(s,str) and Chem.MolFromSmiles(s) else None for s in test["SMILES"]]
valid=np.where(oko)[0]; tvalid=np.where(okt)[0]
def bmg(idx,g): return BatchMolGraph([g[int(i)] for i in idx])

class DMPNN(nn.Module):
    def __init__(s,d_h=DH,depth=3,hidden=256,drop=0.1,nfeat=4):
        super().__init__(); s.mp=BondMessagePassing(d_h=d_h,depth=depth); s.agg=MeanAggregation()
        s.trunk=nn.Sequential(nn.Linear(d_h+nfeat,hidden),nn.GELU(),nn.Dropout(drop))
        s.mu=nn.Linear(hidden,4); s.dr=nn.Linear(hidden,4)
    def forward(s,b,pf): g=s.agg(s.mp(b),b.batch); h=s.trunk(torch.cat([g,pf],1)); return s.mu(h), nn.functional.softplus(s.dr(h))

def run_seed(seed):
    npy=f"experiments/blinded_dmpnn_primary_s{seed}.npy"
    if os.path.exists(npy): print(f"  s{seed}: cached",flush=True); return np.load(npy)
    rng=np.random.RandomState(seed); iv=rng.permutation(valid)[:int(len(valid)*IVFRAC)]
    tr=np.setdiff1d(valid,iv)
    ym=np.array([Y[tr,j][M[tr,j]].mean() for j in range(4)]); ys=np.array([Y[tr,j][M[tr,j]].std() or 1.0 for j in range(4)])
    lo=np.nan_to_num((LO-ym)/ys); hi=np.nan_to_num((HI-ym)/ys)
    pm=np.nanmean(PLOGo[tr],0); ps=np.nanstd(PLOGo[tr],0); ps=np.where(ps<1e-6,1.0,ps)
    PFo=np.nan_to_num((PLOGo-pm)/ps); PFt=np.nan_to_num((PLOGt-pm)/ps)  # test uses TRAIN feature stats
    def pfo(rows): return torch.tensor(PFo[rows],dtype=torch.float32)
    def pft(rows): return torch.tensor(PFt[rows],dtype=torch.float32)
    torch.manual_seed(seed); net=DMPNN(); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3)
    n=len(tr); g=torch.Generator().manual_seed(seed); best=(1e9,None,0)
    def macro_iv():
        with torch.no_grad(): mv=net(bmg(iv,MG),pfo(iv))[0].numpy()*ys+ym
        vals=[]
        for j in range(4):
            mm=M[iv,j]
            if mm.sum(): vals.append(float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (mv[mm,j],LO[iv,j][mm],HI[iv,j][mm],Y[iv,j][mm])))))
        return float(np.mean(vals))
    for ep in range(MAXEP):
        net.train(); perm=torch.randperm(n,generator=g).numpy()
        for i in range(0,n,BS):
            b=tr[perm[i:i+BS]]; opt.zero_grad(); mu,_=net(bmg(b,MG),pfo(b))
            loss=interval_hinge(mu,torch.tensor(lo[b],dtype=torch.float32),torch.tensor(hi[b],dtype=torch.float32),torch.tensor(M[b],dtype=torch.float32))
            loss.backward(); opt.step()
        net.eval(); ivs=macro_iv()
        if ivs<best[0]-1e-4: best=(ivs,copy.deepcopy(net.state_dict()),ep)
        elif ep-best[2]>=PATIENCE: break
    net.load_state_dict(best[1]); net.eval()
    pred=np.full((len(test),4),np.nan)
    with torch.no_grad(): pred[tvalid]=net(bmg(tvalid,TG),pft(tvalid))[0].numpy()*ys+ym
    np.save(npy,pred); print(f"  s{seed}: computed best@{best[2]+1}ep",flush=True); return pred

preds=np.nanmean([run_seed(s) for s in [0,1,2]],0)
np.save("experiments/blinded_dmpnn_primary_ensemble.npy",preds)
def stats(x): x=x[~np.isnan(x)]; return f"std={x.std():.2f} range=[{x.min():.2f},{x.max():.2f}]"
for j,iso in enumerate(ISOS): print(f"  {iso}: {stats(preds[:,j])}  train tgt std={df[f'{iso}_pIC50_direct_inhibition'].std():.2f}")
print("done")
