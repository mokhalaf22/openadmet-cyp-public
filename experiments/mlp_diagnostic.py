"""Bounded diagnostic: is the torch MLP's gap optimization or model class?"""
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from cyp.features import featurize
from cyp.splits import scaffold_folds
from cyp.losses import DirectShiftHead, st_rae as st_rae_torch

torch.set_num_threads(1)
D="data/cyp-challenge-train-test/"
SP="/private/tmp/claude-501/-Users-mohamedmahmoud-github-projects-challenge-public-cyp/6b6cc8ab-cfb6-46f2-b843-204b47dc2156/scratchpad"
ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; SEED=0; NF=5
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
print("featurizing..."); X,ok,names=featurize(df["SMILES"].tolist())

# ---- 1. feature cleaning (before standardizing) ----
var=X[ok].var(0); finite=np.isfinite(X[ok]).all(0)
drop=np.zeros(X.shape[1],bool)
ipc=[i for i,n in enumerate(names) if n=="Ipc"]
drop[ipc]=True; nzv=var<1e-8; nonfin=~finite
drop|=nzv; drop|=nonfin
keep=np.where(~drop)[0]
print("\n=== 1. FEATURE CLEANING ===")
print(f"  start: {X.shape[1]} features")
print(f"  dropped Ipc: {len(ipc)}; near-zero-variance: {int(nzv.sum())}; non-finite: {int(nonfin.sum())}")
print(f"  remaining: {len(keep)} features")
Xc=X[:,keep].astype(np.float32)

def stdz(tr,va):
    mu=tr.mean(0); sd=tr.std(0); sd[sd==0]=1
    return ((tr-mu)/sd).astype(np.float32), ((va-mu)/sd).astype(np.float32)

def strae(pred,lo,hi,y):
    return st_rae_torch(*(torch.tensor(a,dtype=torch.float64) for a in (pred,lo,hi,y)))

def std1(v): v=np.asarray(v,float); return float(v.std(ddof=1)) if len(v)>1 else 0.0

def lgbm(Xtr,ytr,wtr,Xva):
    m=lgb.LGBMRegressor(objective="regression_l1",n_estimators=500,learning_rate=0.03,
        num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,
        reg_lambda=1.0,random_state=SEED,deterministic=True,force_row_wise=True,n_jobs=1,verbosity=-1)
    m.fit(Xtr,ytr,sample_weight=wtr); return m.predict(Xva)

def ridge(Xtr,ytr,wtr,Xva):
    Xt,Xv=stdz(Xtr,Xva)
    m=RidgeCV(alphas=[0.1,1,10,100,1000]); m.fit(Xt,ytr,sample_weight=wtr); return m.predict(Xv)

def skmlp(Xtr,ytr,wtr,Xva):
    Xt,Xv=stdz(Xtr,Xva)
    m=MLPRegressor(hidden_layer_sizes=(256,),alpha=1e-4,max_iter=300,early_stopping=True,
        random_state=SEED)  # no sample_weight support
    m.fit(Xt,ytr); return m.predict(Xv)

def torchmlp(Xtr,ytr,wtr,Xva,curves=None):
    Xt,Xv=stdz(Xtr,Xva)
    torch.manual_seed(SEED)
    model=DirectShiftHead(Xt.shape[1],1,512)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    Xt_t=torch.tensor(Xt); y_t=torch.tensor(ytr[:,None],dtype=torch.float32); w_t=torch.tensor(wtr[:,None],dtype=torch.float32)
    for ep in range(300):
        model.train(); opt.zero_grad()
        mu,_=model(Xt_t); loss=((mu-y_t).abs()*w_t).mean()
        loss.backward(); opt.step()
        if curves is not None:
            with torch.no_grad():
                vp=model(torch.tensor(Xv))[0].numpy()[:,0]
            curves["train"].append(float(loss)); curves["val_strae"].append(float(curves["_srae"](vp)))
    with torch.no_grad(): return model(torch.tensor(Xv))[0].numpy()[:,0]

# ---- 2. four learners on identical cleaned features + folds ----
print("\n=== 2. OOF ST-RAE (mean +/- fold std), identical cleaned features + folds ===")
print(f"  {'iso':7} {'LightGBM':>16} {'Ridge':>16} {'sklearn-MLP':>16} {'torch-MLP':>16}")
for iso in ISOS:
    y_all=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float)
    idx=np.where(~np.isnan(y_all)&ok)[0]
    smi=df["SMILES"].to_numpy()[idx]; fold,_=scaffold_folds(smi.tolist(),NF,SEED)
    y=y_all[idx]; Xs=Xc[idx]
    lo=df[f"{iso}_pIC50_direct_inhibition_conf_low"].to_numpy(float)[idx]
    hi=df[f"{iso}_pIC50_direct_inhibition_conf_high"].to_numpy(float)[idx]
    w=hi-lo; w=np.where(np.isnan(w),np.nanmedian(w),w); weight=1.0/(1.0+w)
    res={k:[] for k in ["lgbm","ridge","skmlp","torch"]}
    for f in range(NF):
        va=fold==f; tr=~va
        for name,fn in [("lgbm",lgbm),("ridge",ridge),("skmlp",skmlp),("torch",torchmlp)]:
            pred=fn(Xs[tr],y[tr],weight[tr],Xs[va])
            res[name].append(float(strae(pred,lo[va],hi[va],y[va])))
    def cell(k): return f"{np.mean(res[k]):.3f}+/-{std1(res[k]):.3f}"
    print(f"  {iso:7} {cell('lgbm'):>16} {cell('ridge'):>16} {cell('skmlp'):>16} {cell('torch'):>16}")

# ---- 3. loss curves for CYP2D6 ----
print("\n=== 3. torch MLP loss curves for CYP2D6 (per fold) -> PNG ===")
iso="CYP2D6"
y_all=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float)
idx=np.where(~np.isnan(y_all)&ok)[0]
smi=df["SMILES"].to_numpy()[idx]; fold,_=scaffold_folds(smi.tolist(),NF,SEED)
y=y_all[idx]; Xs=Xc[idx]
lo=df[f"{iso}_pIC50_direct_inhibition_conf_low"].to_numpy(float)[idx]
hi=df[f"{iso}_pIC50_direct_inhibition_conf_high"].to_numpy(float)[idx]
w=hi-lo; w=np.where(np.isnan(w),np.nanmedian(w),w); weight=1.0/(1.0+w)
fig,axes=plt.subplots(1,NF,figsize=(19,3.6),sharey=True)
for f in range(NF):
    va=fold==f; tr=~va
    curves={"train":[],"val_strae":[],"_srae":lambda vp,lo=lo[va],hi=hi[va],yy=y[va]:strae(vp,lo,hi,yy)}
    torchmlp(Xs[tr],y[tr],weight[tr],Xs[va],curves=curves)
    ax=axes[f]; ep=range(len(curves["train"]))
    ax.plot(ep,curves["train"],label="train wL1"); ax.plot(ep,curves["val_strae"],label="val ST-RAE")
    ax.set_title(f"fold {f}"); ax.set_xlabel("epoch")
    bestep=int(np.argmin(curves["val_strae"]))
    ax.axvline(bestep,color="k",ls=":",lw=1)
    print(f"  fold {f}: final val ST-RAE={curves['val_strae'][-1]:.3f}, min={min(curves['val_strae']):.3f} @ep{bestep}")
axes[0].legend(fontsize=8)
fig.suptitle("CYP2D6 torch MLP: train weighted-L1 and val ST-RAE (dotted = min val)")
fig.tight_layout(); fig.savefig(SP+"/diag_curves.png",dpi=90); plt.close(fig)
print("saved diag_curves.png")
