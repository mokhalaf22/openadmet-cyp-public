"""D-MPNN tuning — fixed 6-config grid, decided in advance (not exploratory).

Shared multi-task D-MPNN (the phase-1 winner). Grid: epochs {50,150,300} x
d_h {200,400}, depth fixed at 3. Early stopping monitors a held-out 15% slice of
each outer fold's TRAINING rows (seeded) — never the eval fold. Reports per-iso
OOF ST-RAE + fold std, with ridge/LightGBM references.

Resilient: MolGraphs cached once; each config checkpointed to JSON, so a rerun
skips finished configs.

  python experiments/dmpnn_tune.py grid              # 6 configs
  python experiments/dmpnn_tune.py ensemble E DH     # 3-seed ensemble of one config
  python experiments/dmpnn_tune.py report
"""
import sys, json, time, copy
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from cyp.splits import scaffold_folds
from cyp.losses import st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*")
# Multi-threaded for speed on the M4 Pro. This is a tuning grid, so exact bitwise
# reproducibility is traded for ~throughput; the 3-seed ensemble quantifies the
# resulting noise. (cyp.baseline / cyp.twohead keep set_num_threads(1) for
# reproducible submissions.)
torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5
RESULTS="experiments/dmpnn_tune_results.json"
GRID=[(e,d) for d in (200,400) for e in (50,150,300)]   # 6 configs
PATIENCE=20; BS=512; INNER_VAL_FRAC=0.15
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
def dcol(i): return f"{i}_pIC50_direct_inhibition"

from cyp.features import featurize
print("featurizing (for valid mask + folds)...", flush=True)
_,ok,_=featurize(df["SMILES"].tolist())
valid=np.where(ok)[0]
gf,_=scaffold_folds(df["SMILES"].to_numpy()[valid].tolist(),NF,0)
GFOLD=np.full(len(df),-1); GFOLD[valid]=gf

Y=np.stack([df[dcol(i)].to_numpy(float) for i in ISOS],1)
LO=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS],1)
HI=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS],1)
Wd=HI-LO; Wd=np.where(np.isnan(Wd),np.nanmedian(Wd),Wd); W=1.0/(1.0+Wd); M=~np.isnan(Y)

def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0
def load():
    try: return json.load(open(RESULTS))
    except Exception: return {}
def save(d): json.dump(d,open(RESULTS,"w"),indent=1)

from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.data import BatchMolGraph
from chemprop.nn import BondMessagePassing, MeanAggregation
_feat=SimpleMoleculeMolGraphFeaturizer()
print("caching MolGraphs...", flush=True)
MG={int(i):_feat(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])

class DMPNN(nn.Module):
    def __init__(s,d_h,n=4,depth=3,hidden=256,drop=0.1):
        super().__init__(); s.mp=BondMessagePassing(d_h=d_h,depth=depth); s.agg=MeanAggregation()
        s.trunk=nn.Sequential(nn.Linear(d_h,hidden),nn.GELU(),nn.Dropout(drop)); s.mu=nn.Linear(hidden,n); s.dr=nn.Linear(hidden,n)
    def forward(s,b):
        g=s.agg(s.mp(b),b.batch); h=s.trunk(g); return s.mu(h), nn.functional.softplus(s.dr(h))

def macro_strae(pred_std, rows, ym, ys):
    # pred_std: (len(rows),4) standardized; invert and score per-iso present rows
    P=pred_std*ys+ym; vals=[]
    for j,iso in enumerate(ISOS):
        m=M[rows,j]
        if m.sum(): vals.append(strae(P[m,j],LO[rows,j][m],HI[rows,j][m],Y[rows,j][m]))
    return float(np.mean(vals))

def train_fold(d_h,max_epochs,seed,tr_rows,iv_rows):
    ym=np.array([Y[tr_rows,j][M[tr_rows,j]].mean() for j in range(4)])
    ys=np.array([Y[tr_rows,j][M[tr_rows,j]].std() or 1.0 for j in range(4)])
    Ys=np.nan_to_num((Y-ym)/ys)
    torch.manual_seed(seed); m=DMPNN(d_h); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
    n=len(tr_rows); g=torch.Generator().manual_seed(seed)
    best=(1e9,None,0)
    for ep in range(max_epochs):
        m.train(); perm=torch.randperm(n,generator=g).numpy()
        for i in range(0,n,BS):
            b=tr_rows[perm[i:i+BS]]; opt.zero_grad(); mu,_=m(bmg(b))
            mk=torch.tensor(M[b],dtype=torch.float32)
            loss=((mu-torch.tensor(Ys[b],dtype=torch.float32)).abs()*torch.tensor(W[b],dtype=torch.float32)*mk).sum()/mk.sum().clamp(min=1)
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): iv=macro_strae(m(bmg(iv_rows))[0].numpy(),iv_rows,ym,ys)
        if iv<best[0]-1e-4: best=(iv,copy.deepcopy(m.state_dict()),ep)
        elif ep-best[2]>=PATIENCE: break
    m.load_state_dict(best[1]); m.eval()
    return m,ym,ys,best[2]

def run_config(d_h,max_epochs,seed=0):
    oof=np.full((len(df),4),np.nan); stopped=[]
    for f in range(NF):
        t0=time.time()
        outer=np.where((GFOLD!=f)&(GFOLD>=0))[0]
        rng=np.random.RandomState(seed*100+f); iv=rng.permutation(len(outer))[:int(len(outer)*INNER_VAL_FRAC)]
        iv_rows=outer[iv]; tr_rows=np.setdiff1d(outer,iv_rows); va=np.where(GFOLD==f)[0]
        m,ym,ys,best_ep=train_fold(d_h,max_epochs,seed,tr_rows,iv_rows)
        with torch.no_grad(): oof[va]=m(bmg(va))[0].numpy()*ys+ym
        stopped.append(best_ep+1); print(f"  d_h{d_h} ep{max_epochs} s{seed} fold{f}: best@{best_ep+1}ep, {time.time()-t0:.0f}s",flush=True)
    per={}
    for j,iso in enumerate(ISOS):
        sr=[]
        for f in range(NF):
            va=np.where((GFOLD==f)&M[:,j])[0]
            if len(va): sr.append(strae(oof[va,j],LO[va,j],HI[va,j],Y[va,j]))
        per[iso]=[float(np.mean(sr)),std1(sr)]
    per["_median_stop_epochs"]=int(np.median(stopped))
    return per,oof

def report(R):
    ref={"ridge":{"CYP1A2":0.590,"CYP2C9":0.387,"CYP2D6":0.676,"CYP3A4":0.304},
         "lgbm":{"CYP1A2":0.535,"CYP2C9":0.362,"CYP2D6":0.612,"CYP3A4":0.297}}
    def macro(d): return float(np.mean([d[i][0] for i in ISOS]))
    print("\n===== D-MPNN TUNING (shared, global folds, ES on inner 15%) =====")
    print(f"{'config':14} "+" ".join(f"{i:>13}" for i in ISOS)+"   macro  stop@")
    for k in sorted([k for k in R if k.startswith("dh")]):
        d=R[k]; print(f"{k:14} "+" ".join(f"{d[i][0]:.3f}±{d[i][1]:.3f}" for i in ISOS)+f"   {macro(d):.3f}  {d.get('_median_stop_epochs','?')}")
    print(f"{'ridge':14} "+" ".join(f"{ref['ridge'][i]:.3f}       " for i in ISOS)+f"   {np.mean(list(ref['ridge'].values())):.3f}")
    print(f"{'lgbm':14} "+" ".join(f"{ref['lgbm'][i]:.3f}       " for i in ISOS)+f"   {np.mean(list(ref['lgbm'].values())):.3f}")

if __name__=="__main__":
    phase=sys.argv[1] if len(sys.argv)>1 else "report"
    R=load()
    if phase=="grid":
        for e,d in GRID:
            key=f"dh{d}_ep{e}"
            if key in R: print(f"skip {key} (done)",flush=True); continue
            print(f"== {key} ==",flush=True); per,_=run_config(d,e,seed=0); R[key]=per; save(R)
        report(R)
    elif phase=="ensemble":
        e,d=int(sys.argv[2]),int(sys.argv[3]); oofs=[]; seeds=[0,1,2]; per_seed=[]
        for s in seeds:
            npy=f"experiments/ens_oof_dh{d}_ep{e}_s{s}.npy"
            try:
                oof=np.load(npy); print(f"  seed {s}: loaded cached OOF",flush=True)
            except Exception:
                _,oof=run_config(d,e,seed=s); np.save(npy,oof); print(f"  seed {s}: computed + cached",flush=True)
            oofs.append(oof)
            sm=[]
            for j in range(4):
                va=np.where((GFOLD>=0)&M[:,j])[0]; sm.append(np.mean([strae(oof[np.where((GFOLD==f)&M[:,j])[0],j],LO[np.where((GFOLD==f)&M[:,j])[0],j],HI[np.where((GFOLD==f)&M[:,j])[0],j],Y[np.where((GFOLD==f)&M[:,j])[0],j]) for f in range(NF)]))
            m=float(np.mean(sm)); per_seed.append(m); print(f"  seed {s} macro={m:.3f}",flush=True)
        ens=np.nanmean(oofs,axis=0); per={}
        for j,iso in enumerate(ISOS):
            sr=[strae(ens[np.where((GFOLD==f)&M[:,j])[0],j],LO[np.where((GFOLD==f)&M[:,j])[0],j],HI[np.where((GFOLD==f)&M[:,j])[0],j],Y[np.where((GFOLD==f)&M[:,j])[0],j]) for f in range(NF)]
            per[iso]=[float(np.mean(sr)),std1(sr)]
        R[f"ensemble_dh{d}_ep{e}"]={"per":per,"per_seed_macro":per_seed,
            "seed_spread":float(max(per_seed)-min(per_seed)),"ensemble_macro":float(np.mean([per[i][0] for i in ISOS]))}
        save(R); print("per-seed macro:",per_seed,"spread",round(max(per_seed)-min(per_seed),3),"ensemble macro",round(np.mean([per[i][0] for i in ISOS]),3),flush=True)
    else: report(R)
