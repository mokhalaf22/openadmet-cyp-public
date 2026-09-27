"""Dispersion variant (§28): expand each isoform's predictions to 0.85x the
training-target std, about the per-isoform predicted mean, ranking preserved.

Operates on regression_final.parquet (D-MPNN+primary + CYP2D6 -0.5 shift already
applied), so it differs ONLY in dispersion. Per isoform: new = mean + f*(pred-mean),
f = 0.85*train_std / current_std. f>0 and the map is affine, so within-isoform
ordering (Spearman) is exactly preserved. Modest on purpose: 0.85x, not full match,
and far below the x4.03 CYP2D6 blow-up the old gamble implied. Writes
submissions/regression_disp.parquet for a live-board A/B against regression_final.
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,"src")
from cyp.submit import write_submission, load_blinded_ids, REGRESSION_ENDPOINTS
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; TARGET=0.85
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
fin=pd.read_parquet("submissions/regression_final.parquet")
out=fin[["SMILES","Molecule_Name"]].copy()
print(f"dispersion target = {TARGET}x training-target std, per isoform, about predicted mean:")
print(f"{'iso':7} {'f':>6} {'old std':>8} {'new std':>8} {'old ratio':>9} {'new ratio':>9} {'new range':>18} {'Spearman':>9}")
for j,iso in enumerate(ISOS):
    c=f"{iso}_pIC50_direct_inhibition"; ts=df[c].std()
    p=fin[c].to_numpy(float); m=p.mean(); s=p.std()
    f=(TARGET*ts)/s
    np_=m+f*(p-m)
    rho=spearmanr(p,np_).correlation
    out[c]=np_
    print(f"{iso:7} {f:>6.2f} {s:>8.2f} {np_.std():>8.2f} {s/ts:>9.2f} {np_.std()/ts:>9.2f} {f'[{np_.min():.2f},{np_.max():.2f}]':>18} {rho:>9.5f}")
out=out[["SMILES","Molecule_Name"]+REGRESSION_ENDPOINTS]
assert not out[REGRESSION_ENDPOINTS].isna().any().any()
path=write_submission(out,"regression","submissions/regression_disp.parquet",load_blinded_ids())
print(f"\nwrote {path}  rows={len(out)}")
