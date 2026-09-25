"""Phase 2 — loss switches on the tuned shared D-MPNN (dh200_ep300).

Every config is run at the 3-seed ENSEMBLE level (per-seed OOF cached to .npy,
so it resumes after a kill), to match the control's ensembling level. Control is
the point-mode ensemble from §13 (macro 0.445). Loss modes:

  point         : weighted L1 to the point estimate (= control)
  interval      : interval-hinge against [conf_low, conf_high]
  interval_pull : interval-hinge + the 1/(1+width) L1 pull (switch e)

Direct-arm regression only (switches a, e). OOF ST-RAE, global scaffold folds,
ES on a held-out 15% slice of each fold's training rows.

  python experiments/phase2.py interval
  python experiments/phase2.py interval_pull
  python experiments/phase2.py report
"""
import sys, json, copy
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from cyp.splits import scaffold_folds
from cyp.features import featurize
from cyp.losses import interval_hinge, width_weighted_l1, st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5
RESULTS="experiments/phase2_results.json"
DH=200; MAXEP=300; PATIENCE=20; BS=512; IVFRAC=0.15
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
def dcol(i): return f"{i}_pIC50_direct_inhibition"
print("featurizing (valid mask + folds)...",flush=True)
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
print("caching MolGraphs...",flush=True)
MG={int(i):_feat(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])
class DMPNN(nn.Module):
    def __init__(s,d_h=DH,depth=3,hidden=256,drop=0.1):
        super().__init__(); s.mp=BondMessagePassing(d_h=d_h,depth=depth); s.agg=MeanAggregation()
        s.trunk=nn.Sequential(nn.Linear(d_h,hidden),nn.GELU(),nn.Dropout(drop)); s.mu=nn.Linear(hidden,4); s.dr=nn.Linear(hidden,4)
    def forward(s,b):
        g=s.agg(s.mp(b),b.batch); h=s.trunk(g); return s.mu(h), nn.functional.softplus(s.dr(h))

def loss_fn(mode,mu,tgt_std,lo_std,hi_std,pt_std,wstd,wt,mk):
    if mode=="point":
        return ((mu-tgt_std).abs()*wt*mk).sum()/mk.sum().clamp(min=1)
    if mode=="interval":
        return interval_hinge(mu,lo_std,hi_std,mk)
    if mode=="interval_pull":
        return interval_hinge(mu,lo_std,hi_std,mk)+width_weighted_l1(mu,pt_std,wstd,mk)
    raise ValueError(mode)

def macro_iv(pred_std,rows,ym,ys):
    P=pred_std*ys+ym; vals=[]
    for j in range(4):
        m=M[rows,j]
        if m.sum(): vals.append(strae(P[m,j],LO[rows,j][m],HI[rows,j][m],Y[rows,j][m]))
    return float(np.mean(vals))

def run_seed(mode,seed):
    oof=np.full((len(df),4),np.nan)
    for f in range(NF):
        outer=np.where((GFOLD!=f)&(GFOLD>=0))[0]
        rng=np.random.RandomState(seed*100+f); iv=rng.permutation(len(outer))[:int(len(outer)*IVFRAC)]
        iv_rows=outer[iv]; tr=np.setdiff1d(outer,iv_rows); va=np.where(GFOLD==f)[0]
        ym=np.array([Y[tr,j][M[tr,j]].mean() for j in range(4)]); ys=np.array([Y[tr,j][M[tr,j]].std() or 1.0 for j in range(4)])
        tgt=np.nan_to_num((Y-ym)/ys); lo=np.nan_to_num((LO-ym)/ys); hi=np.nan_to_num((HI-ym)/ys); wstd=Wd/ys
        torch.manual_seed(seed); m=DMPNN(); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        n=len(tr); g=torch.Generator().manual_seed(seed); best=(1e9,None,0)
        for ep in range(MAXEP):
            m.train(); perm=torch.randperm(n,generator=g).numpy()
            for i in range(0,n,BS):
                b=tr[perm[i:i+BS]]; opt.zero_grad(); mu,_=m(bmg(b))
                loss=loss_fn(mode,mu,torch.tensor(tgt[b],dtype=torch.float32),torch.tensor(lo[b],dtype=torch.float32),
                             torch.tensor(hi[b],dtype=torch.float32),torch.tensor(tgt[b],dtype=torch.float32),
                             torch.tensor(wstd[b],dtype=torch.float32),torch.tensor(W[b],dtype=torch.float32),
                             torch.tensor(M[b],dtype=torch.float32))
                loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): iv_s=macro_iv(m(bmg(iv_rows))[0].numpy(),iv_rows,ym,ys)
            if iv_s<best[0]-1e-4: best=(iv_s,copy.deepcopy(m.state_dict()),ep)
            elif ep-best[2]>=PATIENCE: break
        m.load_state_dict(best[1]); m.eval()
        with torch.no_grad(): oof[va]=m(bmg(va))[0].numpy()*ys+ym
        print(f"  {mode} s{seed} fold{f}: best@{best[2]+1}ep",flush=True)
    return oof

def per_iso(oof):
    out={}
    for j,iso in enumerate(ISOS):
        sr=[strae(oof[np.where((GFOLD==f)&M[:,j])[0],j],LO[np.where((GFOLD==f)&M[:,j])[0],j],
                  HI[np.where((GFOLD==f)&M[:,j])[0],j],Y[np.where((GFOLD==f)&M[:,j])[0],j]) for f in range(NF)]
        out[iso]=[float(np.mean(sr)),std1(sr)]
    return out

def ensemble(mode):
    oofs=[]; per_seed=[]
    for s in [0,1,2]:
        npy=f"experiments/p2_{mode}_s{s}.npy"
        try: oof=np.load(npy); print(f"  {mode} s{s}: cached",flush=True)
        except Exception: oof=run_seed(mode,s); np.save(npy,oof); print(f"  {mode} s{s}: computed",flush=True)
        oofs.append(oof); per_seed.append(float(np.mean([per_iso(oof)[i][0] for i in ISOS])))
    ens=np.nanmean(oofs,axis=0); per=per_iso(ens)
    return {"per":per,"per_seed_macro":per_seed,"seed_spread":float(max(per_seed)-min(per_seed)),
            "ensemble_macro":float(np.mean([per[i][0] for i in ISOS]))}

CTRL={"per":{"CYP1A2":[0.532,0.028],"CYP2C9":[0.356,0.025],"CYP2D6":[0.570,0.030],"CYP3A4":[0.323,0.042]},
      "ensemble_macro":0.445}  # §13 point-mode 3-seed ensemble
LG={"CYP1A2":0.535,"CYP2C9":0.362,"CYP2D6":0.612,"CYP3A4":0.297}  # LightGBM reference

def report(R):
    R=dict(R); R["control(point)"]=CTRL
    print("\n===== PHASE 2 (shared D-MPNN dh200_ep300, 3-seed ensemble) =====")
    print(f"{'config':16} "+" ".join(f"{i:>13}" for i in ISOS)+"   macro")
    for k in ["control(point)","interval","interval_pull"]:
        if k in R:
            d=R[k]["per"]; print(f"{k:16} "+" ".join(f"{d[i][0]:.3f}±{d[i][1]:.3f}" for i in ISOS)+f"   {R[k]['ensemble_macro']:.3f}")
    print(f"{'LightGBM(ref)':16} "+" ".join(f"{LG[i]:.3f}       " for i in ISOS)+f"   {np.mean(list(LG.values())):.3f}")
    if "interval" in R and "interval_pull" in R:
        a,b=R["interval"]["per"],R["interval_pull"]["per"]
        print("\nwidth-pull isolated effect (interval_pull - interval), negative=better:")
        print("  "+" ".join(f"{i}:{b[i][0]-a[i][0]:+.3f}" for i in ISOS))

if __name__=="__main__":
    ph=sys.argv[1] if len(sys.argv)>1 else "report"
    R=load()
    if ph in ("interval","interval_pull","point"):
        R[ph]=ensemble(ph); save(R); print(f"{ph}: macro={R[ph]['ensemble_macro']:.3f} per-seed={[round(x,3) for x in R[ph]['per_seed_macro']]} spread={R[ph]['seed_spread']:.3f}",flush=True)
    report(R)
