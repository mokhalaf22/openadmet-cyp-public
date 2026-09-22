"""Option-A ablation: representation before loss design.

Global scaffold folds (one partition over all valid compounds) so per-isoform
and shared-encoder configs are evaluated on identical held-out sets. Deltas are
measured from the two-head control (ecfp + per-iso + point); Ridge and LightGBM
are external reference columns; fold std on everything.

Phased + resumable: each phase appends to experiments/ablation_results.json and
prints line-buffered, so partial results survive an interruption.

  python experiments/ablation.py ecfp           # refs + control + shared (fast)
  python experiments/ablation.py dmpnn-shared    # shared D-MPNN
  python experiments/ablation.py dmpnn-periso    # per-isoform D-MPNN
  python experiments/ablation.py report          # print tables from the JSON
"""
import sys, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import lightgbm as lgb
from rdkit import Chem, RDLogger
from sklearn.linear_model import RidgeCV
from cyp.features import featurize
from cyp.splits import scaffold_folds
from cyp.losses import DirectShiftHead, st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(1)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; SEED=0; NF=5
RESULTS="experiments/ablation_results.json"
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
def dcol(i): return f"{i}_pIC50_direct_inhibition"

print("featurizing ECFP + cleaning...", flush=True)
X,ok,names=featurize(df["SMILES"].tolist())
var=X[ok].var(0); drop=(var<1e-8)|~np.isfinite(X[ok]).all(0)
drop[[i for i,n in enumerate(names) if n=="Ipc"]]=True
Xc=X[:,~drop].astype(np.float32)
valid=np.where(ok)[0]
gf,_=scaffold_folds(df["SMILES"].to_numpy()[valid].tolist(),NF,SEED)
GFOLD=np.full(len(df),-1); GFOLD[valid]=gf

def bounds(i):
    lo=df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float); hi=df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float)
    w=hi-lo; w=np.where(np.isnan(w),np.nanmedian(w),w); return lo,hi,1.0/(1.0+w)
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0
def zfit(a): m=a.mean(); s=a.std() or 1.0; return m,s
def stdz(tr,va): mu=tr.mean(0); sd=tr.std(0); sd[sd==0]=1; return ((tr-mu)/sd).astype(np.float32),((va-mu)/sd).astype(np.float32)
def load():
    try: return json.load(open(RESULTS))
    except Exception: return {}
def save(d): json.dump(d,open(RESULTS,"w"),indent=1)

# ---------- ECFP per-isoform learners ----------
def ridge_pred(iso,tr,va):
    _,_,wt=bounds(iso); y=df[dcol(iso)].to_numpy(float)
    Xt,Xv=stdz(Xc[tr],Xc[va]); r=RidgeCV(alphas=[0.1,1,10,100,1000]); r.fit(Xt,y[tr],sample_weight=wt[tr]); return r.predict(Xv)
def lgbm_pred(iso,tr,va):
    _,_,wt=bounds(iso); y=df[dcol(iso)].to_numpy(float)
    m=lgb.LGBMRegressor(objective="regression_l1",n_estimators=500,learning_rate=0.03,num_leaves=31,
        min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=1.0,
        random_state=SEED,deterministic=True,force_row_wise=True,n_jobs=1,verbosity=-1)
    m.fit(Xc[tr],y[tr],sample_weight=wt[tr]); return m.predict(Xc[va])
def torch_ecfp_pred(iso,tr,va,h=256,wd=1e-3,epochs=300,bs=256):
    _,_,wt=bounds(iso); y=df[dcol(iso)].to_numpy(float); ym,ys=zfit(y[tr])
    Xt,Xv=stdz(Xc[tr],Xc[va]); torch.manual_seed(SEED)
    m=DirectShiftHead(Xt.shape[1],1,h); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=wd)
    Xt_t=torch.tensor(Xt); yt=torch.tensor(((y[tr]-ym)/ys)[:,None],dtype=torch.float32); wtt=torch.tensor(wt[tr][:,None],dtype=torch.float32)
    n=len(Xt_t); g=torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        m.train(); perm=torch.randperm(n,generator=g)
        for i in range(0,n,bs):
            b=perm[i:i+bs]; opt.zero_grad(); mu,_=m(Xt_t[b]); (((mu-yt[b]).abs()*wtt[b]).mean()).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(torch.tensor(Xv))[0].numpy()[:,0]*ys+ym

def run_shared_ecfp(h=256,wd=1e-3,epochs=300,bs=256):
    Y=np.stack([df[dcol(i)].to_numpy(float) for i in ISOS],1); W=np.stack([bounds(i)[2] for i in ISOS],1); M=~np.isnan(Y)
    oof={i:np.full(len(df),np.nan) for i in ISOS}
    for f in range(NF):
        tr=np.where((GFOLD!=f)&(GFOLD>=0))[0]; va=np.where(GFOLD==f)[0]
        Xt,Xv=stdz(Xc[tr],Xc[va])
        ym=np.array([Y[tr,j][M[tr,j]].mean() for j in range(4)]); ys=np.array([Y[tr,j][M[tr,j]].std() or 1.0 for j in range(4)])
        Ys=(Y-ym)/ys; torch.manual_seed(SEED); m=DirectShiftHead(Xt.shape[1],4,h)
        opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=wd)
        Xt_t=torch.tensor(Xt); yt=torch.tensor(np.nan_to_num(Ys[tr]),dtype=torch.float32)
        wt=torch.tensor(W[tr],dtype=torch.float32); mk=torch.tensor(M[tr],dtype=torch.float32)
        n=len(tr); g=torch.Generator().manual_seed(SEED)
        for _ in range(epochs):
            m.train(); perm=torch.randperm(n,generator=g)
            for i in range(0,n,bs):
                b=perm[i:i+bs]; opt.zero_grad(); mu,_=m(Xt_t[b])
                (((mu-yt[b]).abs()*wt[b]*mk[b]).sum()/mk[b].sum().clamp(min=1)).backward(); opt.step()
        m.eval()
        with torch.no_grad(): pr=m(torch.tensor(Xv))[0].numpy()*ys+ym
        for j,iso in enumerate(ISOS): oof[iso][va]=pr[:,j]
    return oof

# ---------- D-MPNN ----------
def _dmpnn_setup():
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.data import BatchMolGraph
    from chemprop.nn import BondMessagePassing, MeanAggregation
    feat=SimpleMoleculeMolGraphFeaturizer()
    print("building MolGraphs...", flush=True)
    MG={int(i):feat(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
    class DMPNN(nn.Module):
        def __init__(s,n,d_h=200,depth=3,hidden=256,drop=0.1):
            super().__init__(); s.mp=BondMessagePassing(d_h=d_h,depth=depth); s.agg=MeanAggregation()
            s.trunk=nn.Sequential(nn.Linear(d_h,hidden),nn.GELU(),nn.Dropout(drop)); s.mu=nn.Linear(hidden,n); s.dr=nn.Linear(hidden,n)
        def forward(s,bmg):
            g=s.agg(s.mp(bmg),bmg.batch); h=s.trunk(g); return s.mu(h), nn.functional.softplus(s.dr(h))
    def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])
    return DMPNN, bmg

def dmpnn_periso(DMPNN,bmg,iso,tr,va,epochs=50,bs=256,wd=1e-4):
    _,_,wt=bounds(iso); y=df[dcol(iso)].to_numpy(float); ym,ys=zfit(y[tr])
    torch.manual_seed(SEED); m=DMPNN(1); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=wd)
    yt=((y-ym)/ys); n=len(tr); g=torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        m.train(); perm=torch.randperm(n,generator=g).numpy()
        for i in range(0,n,bs):
            b=tr[perm[i:i+bs]]; opt.zero_grad(); mu,_=m(bmg(b))
            (((mu-torch.tensor(yt[b][:,None],dtype=torch.float32)).abs()*torch.tensor(wt[b][:,None],dtype=torch.float32)).mean()).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(bmg(va))[0].numpy()[:,0]*ys+ym

def run_shared_dmpnn(DMPNN,bmg,epochs=50,bs=256,wd=1e-4):
    Y=np.stack([df[dcol(i)].to_numpy(float) for i in ISOS],1); W=np.stack([bounds(i)[2] for i in ISOS],1); M=~np.isnan(Y)
    oof={i:np.full(len(df),np.nan) for i in ISOS}
    for f in range(NF):
        t0=time.time(); tr=np.where((GFOLD!=f)&(GFOLD>=0))[0]; va=np.where(GFOLD==f)[0]
        ym=np.array([Y[tr,j][M[tr,j]].mean() for j in range(4)]); ys=np.array([Y[tr,j][M[tr,j]].std() or 1.0 for j in range(4)])
        Ys=np.nan_to_num((Y-ym)/ys); torch.manual_seed(SEED); m=DMPNN(4); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=wd)
        n=len(tr); g=torch.Generator().manual_seed(SEED)
        for _ in range(epochs):
            m.train(); perm=torch.randperm(n,generator=g).numpy()
            for i in range(0,n,bs):
                b=tr[perm[i:i+bs]]; opt.zero_grad(); mu,_=m(bmg(b))
                mk=torch.tensor(M[b],dtype=torch.float32)
                (((mu-torch.tensor(Ys[b],dtype=torch.float32)).abs()*torch.tensor(W[b],dtype=torch.float32)*mk).sum()/mk.sum().clamp(min=1)).backward(); opt.step()
        m.eval()
        with torch.no_grad(): pr=m(bmg(va))[0].numpy()*ys+ym
        for j,iso in enumerate(ISOS): oof[iso][va]=pr[:,j]
        print(f"  shared-dmpnn fold {f} done in {time.time()-t0:.0f}s", flush=True)
    return oof

def eval_periso(pred_fn):
    out={}
    for iso in ISOS:
        present=(~np.isnan(df[dcol(iso)].to_numpy(float)))&ok; lo,hi,_=bounds(iso); y=df[dcol(iso)].to_numpy(float); sr=[]
        for f in range(NF):
            va=np.where(present&(GFOLD==f))[0]; tr=np.where(present&(GFOLD!=f)&(GFOLD>=0))[0]
            if len(va)==0 or len(tr)==0: continue
            sr.append(strae(pred_fn(iso,tr,va),lo[va],hi[va],y[va]))
        out[iso]=[float(np.mean(sr)),std1(sr)]; print(f"  {iso}: {out[iso][0]:.3f}±{out[iso][1]:.3f}", flush=True)
    return out
def eval_shared(oof):
    out={}
    for iso in ISOS:
        present=(~np.isnan(df[dcol(iso)].to_numpy(float)))&ok; lo,hi,_=bounds(iso); y=df[dcol(iso)].to_numpy(float); sr=[]
        for f in range(NF):
            va=np.where(present&(GFOLD==f))[0]
            if len(va): sr.append(strae(oof[iso][va],lo[va],hi[va],y[va]))
        out[iso]=[float(np.mean(sr)),std1(sr)]; print(f"  {iso}: {out[iso][0]:.3f}±{out[iso][1]:.3f}", flush=True)
    return out

def report(R):
    def macro(d): return float(np.mean([d[i][0] for i in ISOS]))
    order=["control","shared","dmpnn","dmpnn_sh","ridge","lgbm"]
    print("\n============ ABLATION (global scaffold folds) ============")
    print(f"{'config':10} " + " ".join(f"{i:>13}" for i in ISOS) + "   macro")
    for k in order:
        if k in R: print(f"{k:10} " + " ".join(f"{R[k][i][0]:.3f}±{R[k][i][1]:.3f}" for i in ISOS) + f"   {macro(R[k]):.3f}")
    if "control" in R:
        print("\ndeltas from control (negative = better):")
        for k in ["shared","dmpnn","dmpnn_sh"]:
            if k in R: print(f"  {k:10} " + " ".join(f"{R[k][i][0]-R['control'][i][0]:+.3f}" for i in ISOS) + f"   macro {macro(R[k])-macro(R['control']):+.3f}")

if __name__=="__main__":
    phase=sys.argv[1] if len(sys.argv)>1 else "report"
    R=load()
    if phase=="ecfp":
        print("ridge:"); R["ridge"]=eval_periso(ridge_pred); save(R)
        print("lgbm:"); R["lgbm"]=eval_periso(lgbm_pred); save(R)
        print("control (ecfp per-iso):"); R["control"]=eval_periso(torch_ecfp_pred); save(R)
        print("(b) shared ecfp:"); R["shared"]=eval_shared(run_shared_ecfp()); save(R)
    elif phase=="dmpnn-shared":
        DMPNN,bmg=_dmpnn_setup(); print("(c) shared dmpnn:"); R["dmpnn_sh"]=eval_shared(run_shared_dmpnn(DMPNN,bmg)); save(R)
    elif phase=="dmpnn-periso":
        DMPNN,bmg=_dmpnn_setup(); print("(c) per-iso dmpnn:"); R["dmpnn"]=eval_periso(lambda iso,tr,va: dmpnn_periso(DMPNN,bmg,iso,tr,va)); save(R)
    report(R)
