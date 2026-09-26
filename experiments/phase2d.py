"""Phase 2 (cont.) — AID 1851 (Veith) as auxiliary multi-task heads (lever B).

Two-head shared D-MPNN (interval direct + interval TDI + our classifier head,
sp=5e-3 = the th_clf config) PLUS 5 auxiliary binary heads predicting Veith
active/inactive for CYP1A2/2C9/2C19/2D6/3A4. The aux data enters ONLY through
these separate heads — never merged into the scored pIC50 columns, no cross-assay
rescaling (CLAUDE.md rule). Aux compounds overlapping our train/blinded were
removed in prep. Aux is subsampled to the our-data size each epoch (balances the
multi-task and bounds cost).

Reports direct-arm macro ST-RAE and our classifier MCC vs th_clf (0.433,
2D6 0.125 / 3A4 0.336). 3-seed ensemble, per-fold checkpointing.

  python experiments/phase2d.py aux_on
  python experiments/phase2d.py report
"""
import sys, json, copy, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from sklearn.metrics import matthews_corrcoef
from cyp.splits import scaffold_folds
from cyp.features import featurize
from cyp.guards import tdi_trainable_mask
from cyp.losses import interval_hinge, st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; TDI=["CYP2D6","CYP3A4"]
AUX=["CYP1A2","CYP2C9","CYP2C19","CYP2D6","CYP3A4"]
NF=5; MAXEP=300; PATIENCE=20; BS=512; IVFRAC=0.15; SP=5e-3; AUXW=0.5
RESULTS="experiments/phase2d_results.json"
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
def dc(i): return f"{i}_pIC50_direct_inhibition"
def tc(i): return f"{i}_pIC50_TDI_condition"
print("featurizing ours...",flush=True); _,ok,_=featurize(df["SMILES"].tolist())
valid=np.where(ok)[0]; gf,_=scaffold_folds(df["SMILES"].to_numpy()[valid].tolist(),NF,0)
GFOLD=np.full(len(df),-1); GFOLD[valid]=gf
Yd=np.stack([df[dc(i)].to_numpy(float) for i in ISOS],1)
LOd=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS],1)
HId=np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS],1); Md=~np.isnan(Yd)
Yt=np.stack([df[tc(i)].to_numpy(float) for i in ISOS],1)
LOt=np.stack([df[f"{i}_pIC50_TDI_condition_conf_low"].to_numpy(float) for i in ISOS],1)
HIt=np.stack([df[f"{i}_pIC50_TDI_condition_conf_high"].to_numpy(float) for i in ISOS],1); Mt=~np.isnan(Yt)
ISTDI={i:df[f"{i}_is_TDI"] for i in TDI}; TRAINABLE={i:tdi_trainable_mask(df,i).to_numpy() for i in TDI}
# AID aux
aux=pd.read_parquet("data/pubchem/aid1851_clean.parquet")
AY=aux[AUX].to_numpy(float); AM=~np.isnan(AY); AY=np.nan_to_num(AY)
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
print("caching our MolGraphs...",flush=True)
MG={int(i):_feat(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
print("caching AID MolGraphs...",flush=True)
AMG=[]; AOK=[]
for s in aux["SMILES"]:
    m=Chem.MolFromSmiles(s) if isinstance(s,str) else None
    AMG.append(_feat(m) if m else None); AOK.append(m is not None)
AOK=np.array(AOK)
def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])
def abmg(rows): return BatchMolGraph([AMG[int(r)] for r in rows])
class Net(nn.Module):
    def __init__(s,d_h=200,depth=3,hidden=256,drop=0.1):
        super().__init__(); s.mp=BondMessagePassing(d_h=d_h,depth=depth); s.agg=MeanAggregation()
        s.trunk=nn.Sequential(nn.Linear(d_h,hidden),nn.GELU(),nn.Dropout(drop))
        s.mu=nn.Linear(hidden,4); s.dr=nn.Linear(hidden,4); s.clf=nn.Linear(hidden,2); s.aux=nn.Linear(hidden,5)
    def enc(s,b): return s.trunk(s.agg(s.mp(b),b.batch))
    def forward(s,b):
        h=s.enc(b); return s.mu(h), nn.functional.softplus(s.dr(h)), s.clf(h)
    def aux_logits(s,b): return s.aux(s.enc(b))

def macro_direct(mu_orig,rows):
    vals=[]
    for j in range(4):
        m=Md[rows,j]
        if m.sum(): vals.append(strae(mu_orig[m,j],LOd[rows,j][m],HId[rows,j][m],Yd[rows,j][m]))
    return float(np.mean(vals))

def run_seed(name,seed):
    mu_o=np.full((len(df),4),np.nan); dl_o=np.full((len(df),4),np.nan); clf_o=np.full((len(df),2),np.nan)
    aidx_all=np.where(AOK)[0]
    for f in range(NF):
        va=np.where(GFOLD==f)[0]
        ff=f"experiments/p2d_{name}_s{seed}_f{f}.npz"
        if os.path.exists(ff):
            z=np.load(ff); mu_o[va]=z["mu"]; dl_o[va]=z["dl"]; clf_o[va]=z["clf"]; print(f"  {name} s{seed} f{f}: cached",flush=True); continue
        outer=np.where((GFOLD!=f)&(GFOLD>=0))[0]
        rng=np.random.RandomState(seed*100+f); iv=rng.permutation(len(outer))[:int(len(outer)*IVFRAC)]
        iv_rows=outer[iv]; tr=np.setdiff1d(outer,iv_rows)
        ym=np.array([Yd[tr,j][Md[tr,j]].mean() for j in range(4)]); ys=np.array([Yd[tr,j][Md[tr,j]].std() or 1.0 for j in range(4)])
        dlo=np.nan_to_num((LOd-ym)/ys); dhi=np.nan_to_num((HId-ym)/ys); tlo=np.nan_to_num((LOt-ym)/ys); thi=np.nan_to_num((HIt-ym)/ys)
        clf_y=np.zeros((len(df),2)); clf_m=np.zeros((len(df),2))
        for k,iso in enumerate(TDI):
            tm=TRAINABLE[iso]; clf_y[tm,k]=ISTDI[iso][tm].astype(bool).astype(float); clf_m[tm,k]=1.0
        torch.manual_seed(seed); m=Net(); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        n=len(tr); g=torch.Generator().manual_seed(seed); ga=torch.Generator().manual_seed(seed+7); best=(1e9,None,0)
        for ep in range(MAXEP):
            m.train(); perm=torch.randperm(n,generator=g).numpy()
            for i in range(0,n,BS):
                b=tr[perm[i:i+BS]]; opt.zero_grad(); mu,delta,logits=m(bmg(b))
                loss=interval_hinge(mu,torch.tensor(dlo[b],dtype=torch.float32),torch.tensor(dhi[b],dtype=torch.float32),torch.tensor(Md[b],dtype=torch.float32))
                loss=loss+interval_hinge(mu+delta,torch.tensor(tlo[b],dtype=torch.float32),torch.tensor(thi[b],dtype=torch.float32),torch.tensor(Mt[b],dtype=torch.float32))
                loss=loss+SP*delta.abs().mean()
                cm=torch.tensor(clf_m[b],dtype=torch.float32)
                bce=nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor(clf_y[b],dtype=torch.float32),reduction="none")
                loss=loss+(bce*cm).sum()/cm.sum().clamp(min=1)
                loss.backward(); opt.step()
            # aux pass (subsample to ~n rows/epoch)
            asub=aidx_all[torch.randperm(len(aidx_all),generator=ga).numpy()[:n]]
            for i in range(0,len(asub),BS):
                b=asub[i:i+BS]; opt.zero_grad(); al=m.aux_logits(abmg(b))
                am=torch.tensor(AM[b],dtype=torch.float32)
                abce=nn.functional.binary_cross_entropy_with_logits(al,torch.tensor(AY[b],dtype=torch.float32),reduction="none")
                (AUXW*(abce*am).sum()/am.sum().clamp(min=1)).backward(); opt.step()
            m.eval()
            with torch.no_grad(): iv_s=macro_direct(m(bmg(iv_rows))[0].numpy()*ys+ym,iv_rows)
            if iv_s<best[0]-1e-4: best=(iv_s,copy.deepcopy(m.state_dict()),ep)
            elif ep-best[2]>=PATIENCE: break
        m.load_state_dict(best[1]); m.eval()
        with torch.no_grad():
            mv,dv,lv=m(bmg(va)); mu_o[va]=mv.numpy()*ys+ym; dl_o[va]=dv.numpy()*ys; clf_o[va]=torch.sigmoid(lv).numpy()
        np.savez(ff,mu=mu_o[va],dl=dl_o[va],clf=clf_o[va]); print(f"  {name} s{seed} f{f}: best@{best[2]+1}ep",flush=True)
    return mu_o,dl_o,clf_o

def clf_mcc(clf):
    out={}
    for k,iso in enumerate(TDI):
        tm=TRAINABLE[iso]&~np.isnan(clf[:,k]); truth=ISTDI[iso][tm].astype(bool).to_numpy(); p=clf[tm,k]
        out[iso]=float(max(matthews_corrcoef(truth,(p>=t).astype(int)) for t in np.linspace(0.05,0.95,19)))
    return out

def ensemble(name):
    mus=[]; clfs=[]
    for s in [0,1,2]:
        base=f"experiments/p2d_{name}_s{s}"
        try: mu=np.load(base+"_mu.npy"); clf=np.load(base+"_clf.npy"); print(f"  {name} s{s}: cached",flush=True)
        except Exception:
            mu,dl,clf=run_seed(name,s); np.save(base+"_mu.npy",mu); np.save(base+"_clf.npy",clf); print(f"  {name} s{s}: computed",flush=True)
        mus.append(mu); clfs.append(clf)
    mu=np.nanmean(mus,0); clf=np.nanmean(clfs,0)
    per={}
    for j,iso in enumerate(ISOS):
        sr=[strae(mu[np.where((GFOLD==f)&Md[:,j])[0],j],LOd[np.where((GFOLD==f)&Md[:,j])[0],j],HId[np.where((GFOLD==f)&Md[:,j])[0],j],Yd[np.where((GFOLD==f)&Md[:,j])[0],j]) for f in range(NF)]
        per[iso]=[float(np.mean(sr)),std1(sr)]
    return {"per":per,"macro":float(np.mean([per[i][0] for i in ISOS])),"clf_mcc":clf_mcc(clf)}

def report(R):
    print("\n===== PHASE 2d (AID 1851 aux heads, 3-seed ensemble) =====")
    print("vs th_clf (no aux): directST-RAE 0.433, clfMCC 2D6 0.125 / 3A4 0.336")
    for k in R:
        r=R[k]; print(f"{k:10} directST-RAE macro={r['macro']:.3f}  clfMCC 2D6={r['clf_mcc']['CYP2D6']:.3f} 3A4={r['clf_mcc']['CYP3A4']:.3f}")

if __name__=="__main__":
    ph=sys.argv[1] if len(sys.argv)>1 else "report"
    R=load()
    if ph=="aux_on":
        R[ph]=ensemble(ph); save(R); r=R[ph]; print(f"aux_on: ST-RAE={r['macro']:.3f} clfMCC={r['clf_mcc']}",flush=True)
    report(R)
