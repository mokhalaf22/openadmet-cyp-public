"""§35 step 2 (warm-start): pretrain the D-MPNN encoder on computed physicochemical
properties over the 324-compound near-neighbour corpus (NO assay data), then fine-tune
D-MPNN+primary on the challenge data (GFOLD 3-seed). Report OOF Spearman/ST-RAE vs
D-MPNN+primary (0.6037 / 0.4146).

Corpus is small (324, §35), so the expected effect is small; run for the empirical number.
Pretrain once (deterministic), reuse the encoder init across all folds/seeds.
"""
import sys, copy, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from scipy.stats import spearmanr
from cyp.losses import interval_hinge, st_rae as st_rae_torch
sys.stdout.reconfigure(line_buffering=True)
RDLogger.DisableLog("rdApp.*"); torch.set_num_threads(10)
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; NF=5; DH=200
MAXEP=300; PATIENCE=20; BS=512
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
MG={int(i):_feat(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])

DESCS=[Descriptors.MolWt,Crippen.MolLogP,rdMolDescriptors.CalcTPSA,rdMolDescriptors.CalcNumHBD,
       rdMolDescriptors.CalcNumHBA,rdMolDescriptors.CalcNumRotatableBonds,rdMolDescriptors.CalcNumAromaticRings,
       rdMolDescriptors.CalcFractionCSP3]
def physchem(m): return [f(m) for f in DESCS]

def pretrain_encoder():
    ck="experiments/warmstart_encoder.pt"
    corpus=pd.read_csv("experiments/warmstart_corpus.csv")["smiles"].tolist()
    mols=[Chem.MolFromSmiles(s) for s in corpus]; mols=[m for m in mols if m]
    P=np.array([physchem(m) for m in mols],float); pm=P.mean(0); psd=P.std(0)+1e-8; Pz=(P-pm)/psd
    graphs=[_feat(m) for m in mols]
    class Pre(nn.Module):
        def __init__(s): super().__init__(); s.mp=BondMessagePassing(d_h=DH,depth=3); s.agg=MeanAggregation(); s.head=nn.Sequential(nn.Linear(DH,128),nn.GELU(),nn.Linear(128,len(DESCS)))
        def forward(s,b): return s.head(s.agg(s.mp(b),b.batch))
    torch.manual_seed(0); net=Pre(); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5)
    idx=np.arange(len(mols)); g=torch.Generator().manual_seed(0)
    for ep in range(150):
        net.train(); perm=torch.randperm(len(idx),generator=g).numpy()
        for i in range(0,len(idx),128):
            b=perm[i:i+128]; opt.zero_grad()
            pred=net(BatchMolGraph([graphs[k] for k in b]))
            loss=((pred-torch.tensor(Pz[b],dtype=torch.float32))**2).mean(); loss.backward(); opt.step()
    torch.save(net.mp.state_dict(),ck); print(f"pretrained encoder on {len(mols)} corpus compounds -> {ck}",flush=True)
    return ck

class DMPNN(nn.Module):
    def __init__(s,enc_sd=None,d_h=DH,depth=3,hidden=256,drop=0.1,nfeat=NFEAT):
        super().__init__(); s.mp=BondMessagePassing(d_h=d_h,depth=depth)
        if enc_sd is not None: s.mp.load_state_dict(enc_sd)
        s.agg=MeanAggregation(); s.trunk=nn.Sequential(nn.Linear(d_h+nfeat,hidden),nn.GELU(),nn.Dropout(drop)); s.mu=nn.Linear(hidden,4)
    def forward(s,b,pf): return s.mu(s.trunk(torch.cat([s.agg(s.mp(b),b.batch),pf],1)))

def macro_iv(mu_std,rows,ym,ys):
    P=mu_std*ys+ym; vals=[]
    for j in range(4):
        mm=M[rows,j]
        if mm.sum(): vals.append(strae(P[mm,j],LO[rows,j][mm],HI[rows,j][mm],Y[rows,j][mm]))
    return float(np.mean(vals))

def run_seed(seed,enc_sd,PF):
    npy=f"experiments/ws_s{seed}.npy"
    if os.path.exists(npy): print(f"ws s{seed}: cached",flush=True); return np.load(npy)
    oof=np.full((len(df),4),np.nan)
    for f in range(NF):
        outer=np.where((GFOLD!=f)&(GFOLD>=0))[0]
        rng=np.random.RandomState(seed*100+f); iv=rng.permutation(len(outer))[:int(len(outer)*0.15)]
        iv_rows=outer[iv]; tr=np.setdiff1d(outer,iv_rows); va=np.where(GFOLD==f)[0]
        ym=np.array([Y[tr,j][M[tr,j]].mean() for j in range(4)]); ys=np.array([Y[tr,j][M[tr,j]].std() or 1.0 for j in range(4)])
        lo=np.nan_to_num((LO-ym)/ys); hi=np.nan_to_num((HI-ym)/ys)
        torch.manual_seed(seed); m=DMPNN(enc_sd=enc_sd); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        n=len(tr); g=torch.Generator().manual_seed(seed); best=(1e9,None,0)
        for ep in range(MAXEP):
            m.train(); perm=torch.randperm(n,generator=g).numpy()
            for i in range(0,n,BS):
                b=tr[perm[i:i+BS]]; opt.zero_grad(); mu=m(bmg(b),torch.tensor(PF[b],dtype=torch.float32))
                interval_hinge(mu,torch.tensor(lo[b],dtype=torch.float32),torch.tensor(hi[b],dtype=torch.float32),torch.tensor(M[b],dtype=torch.float32)).backward(); opt.step()
            m.eval()
            with torch.no_grad(): ivs=macro_iv(m(bmg(iv_rows),torch.tensor(PF[iv_rows],dtype=torch.float32)).numpy(),iv_rows,ym,ys)
            if ivs<best[0]-1e-4: best=(ivs,copy.deepcopy(m.state_dict()),ep)
            elif ep-best[2]>=PATIENCE: break
        m.load_state_dict(best[1]); m.eval()
        with torch.no_grad(): oof[va]=m(bmg(va),torch.tensor(PF[va],dtype=torch.float32)).numpy()*ys+ym
        print(f"  ws s{seed} fold{f}: best@{best[2]+1}ep",flush=True)
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
    pm=np.nanmean(PLOG[valid],0); ps=np.nanstd(PLOG[valid],0); ps=np.where(ps<1e-6,1.0,ps); PF=np.nan_to_num((PLOG-pm)/ps)
    enc_sd=torch.load(pretrain_encoder() if not os.path.exists("experiments/warmstart_encoder.pt") else "experiments/warmstart_encoder.pt")
    ens=np.nanmean([run_seed(s,enc_sd,PF) for s in [0,1,2]],0)
    np.save("experiments/warmstart_oof.npy",ens)
    base=np.load("experiments/dmpnn_primary_oof.npy"); rB,sB=per_iso(base,"sp"),per_iso(base,"strae"); rE,sE=per_iso(ens,"sp"),per_iso(ens,"strae")
    print("\nD-MPNN+primary  vs  warm-start+primary  (Spearman | ST-RAE), GFOLD OOF:")
    for iso in ISOS: print(f"  {iso}: rho {rB[iso]:.3f}->{rE[iso]:.3f} ({rE[iso]-rB[iso]:+.3f}) | strae {sB[iso]:.3f}->{sE[iso]:.3f} ({sE[iso]-sB[iso]:+.3f})")
    print(f"  macro: rho {np.mean(list(rB.values())):.4f}->{np.mean(list(rE.values())):.4f} ({np.mean(list(rE.values()))-np.mean(list(rB.values())):+.4f}) | strae {np.mean(list(sB.values())):.4f}->{np.mean(list(sE.values())):.4f} ({np.mean(list(sE.values()))-np.mean(list(sB.values())):+.4f})")
    print("done")
