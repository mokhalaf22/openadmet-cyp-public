"""Generate the shared D-MPNN interval model's BLINDED predictions (train-on-all).

For Q2 (§22): are the final model's blinded predictions as compressed as the
LightGBM baseline's? Trains the shared D-MPNN with interval targets on ALL
training rows (15% inner-val for early stopping), predicts the 750 blinded
compounds, 3-seed ensemble. Saves predictions and prints per-isoform std/IQR/range.
"""
import sys, copy, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from cyp.features import featurize
from cyp.losses import interval_hinge, st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; MAXEP=300; PATIENCE=20; BS=512; IVFRAC=0.15
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv"); test=pd.read_csv(D+"cyp-challenge-TEST-BLINDED.csv")
print("featurizing...",flush=True)
_,ok,_=featurize(df["SMILES"].tolist()); _,tok,_=featurize(test["SMILES"].tolist())
Y=np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS],1)
LO=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS],1)
HI=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS],1); M=~np.isnan(Y)
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.data import BatchMolGraph
_feat=SimpleMoleculeMolGraphFeaturizer()
print("caching graphs...",flush=True)
MG=[_feat(Chem.MolFromSmiles(s)) if isinstance(s,str) and Chem.MolFromSmiles(s) else None for s in df["SMILES"]]
TG=[_feat(Chem.MolFromSmiles(s)) if isinstance(s,str) and Chem.MolFromSmiles(s) else None for s in test["SMILES"]]
valid=np.where(ok)[0]; tvalid=np.where(tok)[0]
def bmg(idx,g): return BatchMolGraph([g[int(i)] for i in idx])

def run_seed(seed):
    npy=f"experiments/blinded_dmpnn_s{seed}.npy"
    if os.path.exists(npy): print(f"  s{seed}: cached",flush=True); return np.load(npy)
    rng=np.random.RandomState(seed); iv=rng.permutation(valid)[:int(len(valid)*IVFRAC)]
    iv_rows=iv; tr=np.setdiff1d(valid,iv_rows)
    ym=np.array([Y[tr,j][M[tr,j]].mean() for j in range(4)]); ys=np.array([Y[tr,j][M[tr,j]].std() or 1.0 for j in range(4)])
    lo=np.nan_to_num((LO-ym)/ys); hi=np.nan_to_num((HI-ym)/ys)
    torch.manual_seed(seed)
    class Net(nn.Module):
        def __init__(s,d_h=200,depth=3,hidden=256,drop=0.1):
            super().__init__()
            from chemprop.nn import BondMessagePassing, MeanAggregation
            s.mp=BondMessagePassing(d_h=d_h,depth=depth); s.agg=MeanAggregation()
            s.trunk=nn.Sequential(nn.Linear(d_h,hidden),nn.GELU(),nn.Dropout(drop)); s.mu=nn.Linear(hidden,4); s.dr=nn.Linear(hidden,4)
        def forward(s,b): h=s.trunk(s.agg(s.mp(b),b.batch)); return s.mu(h), nn.functional.softplus(s.dr(h))
    net=Net(); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3)
    n=len(tr); g=torch.Generator().manual_seed(seed); best=(1e9,None,0)
    def macro_iv():
        with torch.no_grad(): mv=net(bmg(iv_rows,MG))[0].numpy()*ys+ym
        vals=[]
        for j in range(4):
            mm=M[iv_rows,j]
            if mm.sum(): vals.append(float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (mv[mm,j],LO[iv_rows,j][mm],HI[iv_rows,j][mm],Y[iv_rows,j][mm])))))
        return float(np.mean(vals))
    for ep in range(MAXEP):
        net.train(); perm=torch.randperm(n,generator=g).numpy()
        for i in range(0,n,BS):
            b=tr[perm[i:i+BS]]; opt.zero_grad(); mu,delta=net(bmg(b,MG))
            loss=interval_hinge(mu,torch.tensor(lo[b],dtype=torch.float32),torch.tensor(hi[b],dtype=torch.float32),torch.tensor(M[b],dtype=torch.float32))
            loss.backward(); opt.step()
        net.eval(); ivs=macro_iv()
        if ivs<best[0]-1e-4: best=(ivs,copy.deepcopy(net.state_dict()),ep)
        elif ep-best[2]>=PATIENCE: break
    net.load_state_dict(best[1]); net.eval()
    pred=np.full((len(test),4),np.nan)
    with torch.no_grad(): pred[tvalid]=net(bmg(tvalid,TG))[0].numpy()*ys+ym
    np.save(npy,pred); print(f"  s{seed}: computed best@{best[2]+1}ep",flush=True); return pred

preds=np.nanmean([run_seed(s) for s in [0,1,2]],0)
np.save("experiments/blinded_dmpnn_ensemble.npy",preds)
def stats(x): x=x[~np.isnan(x)]; return f"std={x.std():.2f} IQR={np.percentile(x,75)-np.percentile(x,25):.2f} range=[{x.min():.2f},{x.max():.2f}]"
sub=pd.read_parquet("submissions/regression.parquet")
print("\nBLINDED prediction dispersion (final D-MPNN interval, 3-seed) vs submitted LightGBM baseline:")
for j,iso in enumerate(ISOS):
    print(f"  {iso}: D-MPNN {stats(preds[:,j])}  | LGBM(submitted) {stats(sub[f'{iso}_pIC50_direct_inhibition'].to_numpy())}  | train tgt std={df[f'{iso}_pIC50_direct_inhibition'].std():.2f}")
print("\ndone")
