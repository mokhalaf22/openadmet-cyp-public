"""Dispersion variant v3 (§30a): expand ALL four isoforms to 1.0x the training-target
std, about the per-isoform predicted mean, ranking preserved.

Round 2 (§30) confirmed dispersion helps on blind (v2's 0.85x/CYP2D6-0.48x cut
ST-RAE 0.7114->0.6683). v3 pushes to full match, including CYP2D6 (0.48 -> 1.0) via
the SAME mean-centred affine map used throughout (new = mean + f*(pred-mean),
f = 1.0*train_std/current_std) -- not a hand-built construction. Reports per-isoform
factors, ranges, and the CYP2D6 upper tail (>6.0/>6.5) since §28a flagged CYP2D6 as
shifted DOWN. Built from regression_final.parquet (D-MPNN+primary + CYP2D6 -0.5).
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,"src")
from cyp.submit import write_submission, load_blinded_ids, REGRESSION_ENDPOINTS
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]; TARGET=1.0
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
fin=pd.read_parquet("submissions/regression_final.parquet")
out=fin[["SMILES","Molecule_Name"]].copy()
print(f"target = {TARGET}x training-target std, per isoform, about predicted mean:")
print(f"{'iso':7} {'f':>6} {'old std':>8} {'new std':>8} {'old ratio':>9} {'new ratio':>9} {'new range':>18} {'Spearman':>9}")
for iso in ISOS:
    c=f"{iso}_pIC50_direct_inhibition"; ts=df[c].std()
    p=fin[c].to_numpy(float); m=p.mean(); s=p.std()
    f=(TARGET*ts)/s
    np_=m+f*(p-m); rho=spearmanr(p,np_).correlation; out[c]=np_
    print(f"{iso:7} {f:>6.2f} {s:>8.2f} {np_.std():>8.2f} {s/ts:>9.2f} {np_.std()/ts:>9.2f} {f'[{np_.min():.2f},{np_.max():.2f}]':>18} {rho:>9.5f}")
# CYP2D6 upper-tail detail
c="CYP2D6_pIC50_direct_inhibition"; ts=df[c].std(); p=fin[c].to_numpy(float); m=p.mean()
d6=m+((TARGET*ts)/p.std())*(p-m)
tr=df[c].dropna().to_numpy()
print(f"\nCYP2D6 v3: >6.0 = {(d6>6).sum()} ({100*(d6>6).mean():.1f}%)  >6.5 = {(d6>6.5).sum()} ({100*(d6>6.5).mean():.1f}%)  max = {d6.max():.2f}")
print(f"  (train CYP2D6: >6.0 {100*(tr>6).mean():.1f}%  >6.5 {100*(tr>6.5).mean():.1f}%  max {tr.max():.2f}; test known LESS potent, §23)")
out=out[["SMILES","Molecule_Name"]+REGRESSION_ENDPOINTS]
assert not out[REGRESSION_ENDPOINTS].isna().any().any()
path=write_submission(out,"regression","submissions/regression_disp_v3.parquet",load_blinded_ids())
print(f"\nwrote {path}  rows={len(out)}")
