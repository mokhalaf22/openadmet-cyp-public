"""Dispersion variant v3b (§30a): isolate the confirmed hypothesis from the
unconfirmed one.

v3 changed two things at once (three no-shift isoforms 0.85->1.0 AND CYP2D6
0.48->1.0), so its result would not attribute. v3b pushes only the CONFIRMED lever:
CYP1A2/CYP2C9/CYP3A4 -> 1.0x training spread (more dispersion helps where there is
no known distribution shift), while holding CYP2D6 at 0.85x (peer level) rather than
the full training-width tail the organizers' 'less potent' statement (§23) argues
against. Same mean-centred affine map, same CYP2D6 -0.5 location shift underneath.
Built from regression_final.parquet.
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,"src")
from cyp.submit import write_submission, load_blinded_ids, REGRESSION_ENDPOINTS
D="data/cyp-challenge-train-test/"
TARGET={"CYP1A2":1.0,"CYP2C9":1.0,"CYP3A4":1.0,"CYP2D6":0.85}
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
fin=pd.read_parquet("submissions/regression_final.parquet")
out=fin[["SMILES","Molecule_Name"]].copy()
print(f"{'iso':7} {'target':>6} {'f':>6} {'old std':>8} {'new std':>8} {'old ratio':>9} {'new ratio':>9} {'new range':>18} {'Spearman':>9}")
for iso in ["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]:
    c=f"{iso}_pIC50_direct_inhibition"; ts=df[c].std()
    p=fin[c].to_numpy(float); m=p.mean(); s=p.std(); tgt=TARGET[iso]
    f=(tgt*ts)/s
    np_=m+f*(p-m); rho=spearmanr(p,np_).correlation; out[c]=np_
    print(f"{iso:7} {tgt:>6.2f} {f:>6.2f} {s:>8.2f} {np_.std():>8.2f} {s/ts:>9.2f} {np_.std()/ts:>9.2f} {f'[{np_.min():.2f},{np_.max():.2f}]':>18} {rho:>9.5f}")
# CYP2D6 upper-tail detail
c="CYP2D6_pIC50_direct_inhibition"; ts=df[c].std(); p=fin[c].to_numpy(float); m=p.mean()
d6=m+((TARGET['CYP2D6']*ts)/p.std())*(p-m); tr=df[c].dropna().to_numpy()
print(f"\nCYP2D6 v3b (0.85x): >6.0 = {(d6>6).sum()} ({100*(d6>6).mean():.1f}%)  >6.5 = {(d6>6.5).sum()} ({100*(d6>6.5).mean():.1f}%)  max = {d6.max():.2f}")
print(f"  (train CYP2D6: >6.0 {100*(tr>6).mean():.1f}%  >6.5 {100*(tr>6.5).mean():.1f}%  max {tr.max():.2f}; test known LESS potent, §23)")
out=out[["SMILES","Molecule_Name"]+REGRESSION_ENDPOINTS]
assert not out[REGRESSION_ENDPOINTS].isna().any().any()
path=write_submission(out,"regression","submissions/regression_disp_v3b.parquet",load_blinded_ids())
print(f"\nwrote {path}  rows={len(out)}")
