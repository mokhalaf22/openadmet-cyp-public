"""Phase 2 (cont.) — TDI-arm two-head switches: shift_prior (f) and TDI label (d).

Two-head shared D-MPNN (dh200_ep300, interval targets from §14): mu supervised
against the direct interval, mu+delta against the TDI-condition interval (delta =
softplus >= 0). This trains delta, which switches (f)/(d) need.

Metrics per config (3-seed ensemble, global folds):
  - direct-arm macro OOF ST-RAE (regression), floor ~0.004-0.005
  - derived TDI MCC for CYP2D6/CYP3A4 via tdi_label_from_arms(mu, delta)
  - classifier-head TDI MCC (if clf=True), OOF-threshold-tuned

Configs: th_sp0 / th_sp5e-3 / th_sp2e-2 (switch f), th_clf (switch d).
Per-seed OOF cached to .npy (resumable).

  python experiments/phase2b.py th_sp5e-3
  python experiments/phase2b.py report
"""
import sys, json, copy, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from sklearn.metrics import matthews_corrcoef
from cyp.splits import scaffold_folds
from cyp.features import featurize
from cyp.guards import tdi_trainable_mask
from cyp.losses import interval_hinge, tdi_label_from_arms, st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; TDI=["CYP2D6","CYP3A4"]
NF=5; DH=200; MAXEP=300; PATIENCE=20; BS=512; IVFRAC=0.15
RESULTS="experiments/phase2b_results.json"
CONFIGS={"th_sp0":dict(sp=0.0,clf=False),"th_sp5e-3":dict(sp=5e-3,clf=False),
         "th_sp2e-2":dict(sp=2e-2,clf=False),"th_clf":dict(sp=5e-3,clf=True),
         # classifier-head diagnostics
         "clf_p15":dict(sp=5e-3,clf=True,clf_prev=0.15),   # reweight loss to 15% prevalence
         "clf_p20":dict(sp=5e-3,clf=True,clf_prev=0.20),   # reweight loss to 20% prevalence
         "clf_dlneg":dict(sp=5e-3,clf=True,clf_dlneg=True)} # include 3A4 direct-less rows as negatives
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
def dc(i): return f"{i}_pIC50_direct_inhibition"
def tc(i): return f"{i}_pIC50_TDI_condition"
print("featurizing...",flush=True); _,ok,_=featurize(df["SMILES"].tolist())
valid=np.where(ok)[0]; gf,_=scaffold_folds(df["SMILES"].to_numpy()[valid].tolist(),NF,0)
GFOLD=np.full(len(df),-1); GFOLD[valid]=gf
Yd=np.stack([df[dc(i)].to_numpy(float) for i in ISOS],1)
LOd=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS],1)
HId=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS],1)
Md=~np.isnan(Yd)
Yt=np.stack([df[tc(i)].to_numpy(float) for i in ISOS],1)
LOt=np.stack([df[f"{i}_pIC50_TDI_condition_conf_low"].to_numpy(float) for i in ISOS],1)
HIt=np.stack([df[f"{i}_pIC50_TDI_condition_conf_high"].to_numpy(float) for i in ISOS],1)
Mt=~np.isnan(Yt)
# TDI classification labels + trainable mask (both arms, excludes assigned-negatives)
ISTDI={i:df[f"{i}_is_TDI"] for i in TDI}
TRAINABLE={i:tdi_trainable_mask(df,i).to_numpy() for i in TDI}
# CYP3A4 direct-less rows (TDI arm present, direct arm absent) — normally excluded
DIRECTLESS_3A4=(df["CYP3A4_pIC50_TDI_condition"].notna()&df["CYP3A4_pIC50_direct_inhibition"].isna()).to_numpy()
def strae(p,lo,hi,y): return float(st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (p,lo,hi,y))))
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
class TwoHead(nn.Module):
    def __init__(s,clf=False,d_h=DH,depth=3,hidden=256,drop=0.1):
        super().__init__(); s.mp=BondMessagePassing(d_h=d_h,depth=depth); s.agg=MeanAggregation()
        s.trunk=nn.Sequential(nn.Linear(d_h,hidden),nn.GELU(),nn.Dropout(drop))
        s.mu=nn.Linear(hidden,4); s.dr=nn.Linear(hidden,4); s.clf=nn.Linear(hidden,2) if clf else None
    def forward(s,b):
        g=s.agg(s.mp(b),b.batch); h=s.trunk(g)
        logits=s.clf(h) if s.clf is not None else None
        return s.mu(h), nn.functional.softplus(s.dr(h)), logits

def macro_direct_strae(mu_orig,rows):
    vals=[]
    for j in range(4):
        m=Md[rows,j]
        if m.sum(): vals.append(strae(mu_orig[m,j],LOd[rows,j][m],HId[rows,j][m],Yd[rows,j][m]))
    return float(np.mean(vals))

def run_seed(name,cfg,seed):
    mu_oof=np.full((len(df),4),np.nan); dl_oof=np.full((len(df),4),np.nan); clf_oof=np.full((len(df),2),np.nan)
    for f in range(NF):
        va=np.where(GFOLD==f)[0]
        ffile=f"experiments/p2b_{name}_s{seed}_f{f}.npz"
        if os.path.exists(ffile):
            z=np.load(ffile); mu_oof[va]=z["mu"]; dl_oof[va]=z["dl"]; clf_oof[va]=z["clf"]
            print(f"  {name} s{seed} fold{f}: cached",flush=True); continue
        outer=np.where((GFOLD!=f)&(GFOLD>=0))[0]
        rng=np.random.RandomState(seed*100+f); iv=rng.permutation(len(outer))[:int(len(outer)*IVFRAC)]
        iv_rows=outer[iv]; tr=np.setdiff1d(outer,iv_rows)
        ym=np.array([Yd[tr,j][Md[tr,j]].mean() for j in range(4)]); ys=np.array([Yd[tr,j][Md[tr,j]].std() or 1.0 for j in range(4)])
        dlo=np.nan_to_num((LOd-ym)/ys); dhi=np.nan_to_num((HId-ym)/ys)
        tlo=np.nan_to_num((LOt-ym)/ys); thi=np.nan_to_num((HIt-ym)/ys)
        # classifier targets (2D6,3A4) as 0/1, mask = trainable, weights for prevalence sim
        clf_y=np.zeros((len(df),2)); clf_m=np.zeros((len(df),2)); clf_w=np.ones((len(df),2))
        for k,iso in enumerate(TDI):
            tm=TRAINABLE[iso]
            clf_y[tm,k]=ISTDI[iso][tm].astype(bool).astype(float); clf_m[tm,k]=1.0
        if cfg.get("clf_dlneg"):   # diagnostic: 3A4 direct-less rows as extra negatives
            k=TDI.index("CYP3A4"); clf_m[DIRECTLESS_3A4,k]=1.0; clf_y[DIRECTLESS_3A4,k]=0.0
        if cfg.get("clf_prev"):    # reweight each isoform's BCE to a target prevalence p
            p=cfg["clf_prev"]
            for k in range(2):
                sel=clf_m[:,k]==1; prev=clf_y[sel,k].mean()
                clf_w[sel&(clf_y[:,k]==1),k]=p/prev; clf_w[sel&(clf_y[:,k]==0),k]=(1-p)/(1-prev)
        torch.manual_seed(seed); m=TwoHead(clf=cfg["clf"]); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        n=len(tr); g=torch.Generator().manual_seed(seed); best=(1e9,None,0)
        for ep in range(MAXEP):
            m.train(); perm=torch.randperm(n,generator=g).numpy()
            for i in range(0,n,BS):
                b=tr[perm[i:i+BS]]; opt.zero_grad(); mu,delta,logits=m(bmg(b))
                loss=interval_hinge(mu,torch.tensor(dlo[b],dtype=torch.float32),torch.tensor(dhi[b],dtype=torch.float32),torch.tensor(Md[b],dtype=torch.float32))
                loss=loss+interval_hinge(mu+delta,torch.tensor(tlo[b],dtype=torch.float32),torch.tensor(thi[b],dtype=torch.float32),torch.tensor(Mt[b],dtype=torch.float32))
                if cfg["sp"]>0: loss=loss+cfg["sp"]*delta.abs().mean()
                if cfg["clf"]:
                    cm=torch.tensor(clf_m[b],dtype=torch.float32); cw=torch.tensor(clf_w[b],dtype=torch.float32)
                    bce=nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor(clf_y[b],dtype=torch.float32),reduction="none")
                    loss=loss+(bce*cm*cw).sum()/(cm*cw).sum().clamp(min=1)
                loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                mv,_,_=m(bmg(iv_rows)); iv_s=macro_direct_strae(mv.numpy()*ys+ym,iv_rows)
            if iv_s<best[0]-1e-4: best=(iv_s,copy.deepcopy(m.state_dict()),ep)
            elif ep-best[2]>=PATIENCE: break
        m.load_state_dict(best[1]); m.eval()
        with torch.no_grad():
            mv,dv,lv=m(bmg(va))
            mu_oof[va]=mv.numpy()*ys+ym; dl_oof[va]=dv.numpy()*ys   # delta in pIC50 units
            clf_oof[va]=torch.sigmoid(lv).numpy() if cfg["clf"] else np.nan
        np.savez(ffile, mu=mu_oof[va], dl=dl_oof[va], clf=clf_oof[va])
        print(f"  {name} s{seed} fold{f}: best@{best[2]+1}ep",flush=True)
    return mu_oof,dl_oof,clf_oof

def derived_mcc(mu,dl):
    out={}
    for iso in TDI:
        j=ISOS.index(iso); tm=TRAINABLE[iso] & ~np.isnan(mu[:,j])
        pred=tdi_label_from_arms(torch.tensor(mu[tm,j]),torch.tensor(dl[tm,j])).numpy()
        truth=ISTDI[iso][tm].astype(bool).to_numpy()
        out[iso]=float(matthews_corrcoef(truth,pred))
    return out

def clf_mcc(clf):
    out={}
    for k,iso in enumerate(TDI):
        tm=TRAINABLE[iso] & ~np.isnan(clf[:,k]); truth=ISTDI[iso][tm].astype(bool).to_numpy(); p=clf[tm,k]
        grid=np.linspace(0.05,0.95,19); best=max(matthews_corrcoef(truth,(p>=t).astype(int)) for t in grid)
        out[iso]=float(best)
    return out

def ensemble(name):
    cfg=CONFIGS[name]; mus=[]; dls=[]; clfs=[]
    for s in [0,1,2]:
        base=f"experiments/p2b_{name}_s{s}"
        try:
            mu=np.load(base+"_mu.npy"); dl=np.load(base+"_dl.npy"); clf=np.load(base+"_clf.npy"); print(f"  {name} s{s}: cached",flush=True)
        except Exception:
            mu,dl,clf=run_seed(name,cfg,s); np.save(base+"_mu.npy",mu); np.save(base+"_dl.npy",dl); np.save(base+"_clf.npy",clf); print(f"  {name} s{s}: computed",flush=True)
        mus.append(mu); dls.append(dl); clfs.append(clf)
    mu=np.nanmean(mus,0); dl=np.nanmean(dls,0); clf=np.nanmean(clfs,0)
    # direct ST-RAE macro (ensemble)
    per={}
    for j,iso in enumerate(ISOS):
        sr=[strae(mu[np.where((GFOLD==f)&Md[:,j])[0],j],LOd[np.where((GFOLD==f)&Md[:,j])[0],j],HId[np.where((GFOLD==f)&Md[:,j])[0],j],Yd[np.where((GFOLD==f)&Md[:,j])[0],j]) for f in range(NF)]
        per[iso]=[float(np.mean(sr)),std1(sr)]
    res={"per":per,"macro":float(np.mean([per[i][0] for i in ISOS])),"derived_mcc":derived_mcc(mu,dl)}
    if cfg["clf"]: res["clf_mcc"]=clf_mcc(clf)
    return res

def report(R):
    print("\n===== PHASE 2b (two-head TDI arm, 3-seed ensemble) =====")
    print("baseline LightGBM classifier MCC: CYP2D6 0.097, CYP3A4 0.347")
    for k in ["th_sp0","th_sp5e-3","th_sp2e-2","th_clf","clf_p15","clf_p20","clf_dlneg"]:
        if k in R:
            r=R[k]; dm=r["derived_mcc"]
            line=f"{k:10} directST-RAE macro={r['macro']:.3f}  derivedMCC 2D6={dm['CYP2D6']:.3f} 3A4={dm['CYP3A4']:.3f}"
            if "clf_mcc" in r: line+=f"  clfMCC 2D6={r['clf_mcc']['CYP2D6']:.3f} 3A4={r['clf_mcc']['CYP3A4']:.3f}"
            print(line)

if __name__=="__main__":
    ph=sys.argv[1] if len(sys.argv)>1 else "report"
    R=load()
    if ph in CONFIGS:
        R[ph]=ensemble(ph); save(R)
        r=R[ph]; print(f"{ph}: directST-RAE={r['macro']:.3f} derivedMCC={r['derived_mcc']}"+(f" clfMCC={r['clf_mcc']}" if 'clf_mcc' in r else ""),flush=True)
    report(R)
