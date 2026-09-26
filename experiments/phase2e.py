"""Standalone TDI classifier — own D-MPNN encoder, TDI objective only.

No regression heads competing for capacity: a D-MPNN encoder trained purely to
classify is_TDI for CYP3A4 and CYP2D6 on the trainable (both-arms) rows. Tests
whether the encoder-sharing that helped regression (-0.03) is costing the
classifier. 3-seed ensemble, global folds, per-fold checkpointing.

  python experiments/phase2e.py clf_only
  python experiments/phase2e.py report
"""
import sys, json, copy, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from sklearn.metrics import matthews_corrcoef
from cyp.splits import scaffold_folds
from cyp.features import featurize
from cyp.guards import tdi_trainable_mask
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; TDI=["CYP2D6","CYP3A4"]; NF=5; MAXEP=300; PATIENCE=20; BS=512; IVFRAC=0.15
RESULTS="experiments/phase2e_results.json"
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
print("featurizing...",flush=True); _,ok,_=featurize(df["SMILES"].tolist())
valid=np.where(ok)[0]; gf,_=scaffold_folds(df["SMILES"].to_numpy()[valid].tolist(),NF,0)
GFOLD=np.full(len(df),-1); GFOLD[valid]=gf
Y=np.zeros((len(df),2)); M=np.zeros((len(df),2))
for k,iso in enumerate(TDI):
    tm=tdi_trainable_mask(df,iso).to_numpy()
    Y[tm,k]=df[f"{iso}_is_TDI"][tm].astype(bool).astype(float); M[tm,k]=1.0
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0
def load():
    try: return json.load(open(RESULTS))
    except Exception: return {}
def save(d): json.dump(d,open(RESULTS,"w"),indent=1)
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.data import BatchMolGraph
from chemprop.nn import BondMessagePassing, MeanAggregation
_feat=SimpleMoleculeMolGraphFeaturizer(); print("caching MolGraphs...",flush=True)
MG={int(i):_feat(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])
class Clf(nn.Module):
    def __init__(s,d_h=200,depth=3,hidden=256,drop=0.1):
        super().__init__(); s.mp=BondMessagePassing(d_h=d_h,depth=depth); s.agg=MeanAggregation()
        s.trunk=nn.Sequential(nn.Linear(d_h,hidden),nn.GELU(),nn.Dropout(drop)); s.clf=nn.Linear(hidden,2)
    def forward(s,b): return s.clf(s.trunk(s.agg(s.mp(b),b.batch)))

def run_seed(name,seed):
    oof=np.full((len(df),2),np.nan)
    # rows with at least one trainable isoform label
    trainable_any=(M.sum(1)>0)
    for f in range(NF):
        va=np.where(GFOLD==f)[0]
        ff=f"experiments/p2e_{name}_s{seed}_f{f}.npz"
        if os.path.exists(ff):
            z=np.load(ff); oof[va]=z["clf"]; print(f"  {name} s{seed} f{f}: cached",flush=True); continue
        outer=np.where((GFOLD!=f)&(GFOLD>=0)&trainable_any)[0]
        rng=np.random.RandomState(seed*100+f); iv=rng.permutation(len(outer))[:int(len(outer)*IVFRAC)]
        iv_rows=outer[iv]; tr=np.setdiff1d(outer,iv_rows)
        # pos_weight per isoform from training rows
        pw=[]
        for k in range(2):
            sel=(M[tr,k]==1); pos=max(1,int(Y[tr,k][sel].sum())); neg=int((Y[tr,k][sel]==0).sum()); pw.append(neg/pos)
        pw=torch.tensor(pw,dtype=torch.float32)
        torch.manual_seed(seed); m=Clf(); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        n=len(tr); g=torch.Generator().manual_seed(seed); best=(1e9,None,0)
        for ep in range(MAXEP):
            m.train(); perm=torch.randperm(n,generator=g).numpy()
            for i in range(0,n,BS):
                b=tr[perm[i:i+BS]]; opt.zero_grad(); logit=m(bmg(b)); mk=torch.tensor(M[b],dtype=torch.float32)
                bce=nn.functional.binary_cross_entropy_with_logits(logit,torch.tensor(Y[b],dtype=torch.float32),pos_weight=pw,reduction="none")
                (( bce*mk).sum()/mk.sum().clamp(min=1)).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                lv=m(bmg(iv_rows)); mk=torch.tensor(M[iv_rows],dtype=torch.float32)
                ivbce=float((nn.functional.binary_cross_entropy_with_logits(lv,torch.tensor(Y[iv_rows],dtype=torch.float32),pos_weight=pw,reduction="none")*mk).sum()/mk.sum().clamp(min=1))
            if ivbce<best[0]-1e-4: best=(ivbce,copy.deepcopy(m.state_dict()),ep)
            elif ep-best[2]>=PATIENCE: break
        m.load_state_dict(best[1]); m.eval()
        with torch.no_grad(): oof[va]=torch.sigmoid(m(bmg(va))).numpy()
        np.savez(ff,clf=oof[va]); print(f"  {name} s{seed} f{f}: best@{best[2]+1}ep",flush=True)
    return oof

def mcc(clf):
    out={}
    for k,iso in enumerate(TDI):
        tm=(M[:,k]==1)&~np.isnan(clf[:,k]); truth=Y[tm,k].astype(int); p=clf[tm,k]
        out[iso]=float(max(matthews_corrcoef(truth,(p>=t).astype(int)) for t in np.linspace(0.05,0.95,19)))
    return out

def ensemble(name):
    clfs=[]
    for s in [0,1,2]:
        base=f"experiments/p2e_{name}_s{s}.npy"
        try: c=np.load(base); print(f"  {name} s{s}: cached",flush=True)
        except Exception: c=run_seed(name,s); np.save(base,c); print(f"  {name} s{s}: computed",flush=True)
        clfs.append(c)
    clf=np.nanmean(clfs,0)
    per_seed={iso:[mcc(c)[iso] for c in clfs] for iso in TDI}
    return {"mcc":mcc(clf),"per_seed":per_seed}

def report(R):
    print("\n===== STANDALONE TDI CLASSIFIER (own encoder, 3-seed ensemble) =====")
    print("refs: shared-model clf head 2D6 0.125 / 3A4 0.336 ; baseline LightGBM 0.097 / 0.347")
    for k in R:
        r=R[k]; ps=r["per_seed"]
        print(f"{k:10} MCC 2D6={r['mcc']['CYP2D6']:.3f} (seeds {[round(x,3) for x in ps['CYP2D6']]})  3A4={r['mcc']['CYP3A4']:.3f} (seeds {[round(x,3) for x in ps['CYP3A4']]})")

if __name__=="__main__":
    ph=sys.argv[1] if len(sys.argv)>1 else "report"
    R=load()
    if ph=="clf_only":
        R[ph]=ensemble(ph); save(R); print(f"clf_only: {R[ph]['mcc']}",flush=True)
    report(R)
