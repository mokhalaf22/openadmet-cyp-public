"""Final regression submission (§26 #3 winner): D-MPNN interval + predicted-primary.

Like-for-like on our scaffold folds (GFOLD), macro OOF:
  LightGBM base 0.451 | LightGBM+primary 0.436 | D-MPNN base 0.434 |
  D-MPNN+primary 0.415  <- winner (the predicted-primary feature helps the D-MPNN
  more than LightGBM; a CYP3A4 blend adds only 0.001, below the seed floor, so we
  keep pure D-MPNN+primary).

Base test predictions come from gen_blinded_primary.py (D-MPNN+primary trained on
all data, 3-seed, blinded_dmpnn_primary_ensemble.npy). We apply the validated
CYP2D6 -0.5 location shift (§24) and write a schema-valid parquet.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from cyp.submit import write_submission, load_blinded_ids, REGRESSION_ENDPOINTS
D="data/cyp-challenge-train-test/"; ISOS=["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]
CYP2D6_SHIFT=-0.5  # validated location shift (§24)
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv"); test=pd.read_csv(D+"cyp-challenge-TEST-BLINDED.csv")
pred=np.load("experiments/blinded_dmpnn_primary_ensemble.npy").astype(float)  # (750,4), NaN for invalid test rows

for j,iso in enumerate(ISOS):
    y=df[f"{iso}_pIC50_direct_inhibition"].to_numpy(float); tmean=float(np.nanmean(y))
    col=pred[:,j]
    col[np.isnan(col)]=tmean  # invalid test rows (featurize failed) -> train mean
    if iso=="CYP2D6": col=col+CYP2D6_SHIFT
    pred[:,j]=col
    print(f"{iso}: test mean={col.mean():.3f} std={col.std():.3f} (train mean={tmean:.3f}){'  [CYP2D6 -0.5]' if iso=='CYP2D6' else ''}",flush=True)

sub=test[["SMILES","Molecule_Name"]].copy()
for j,ep in enumerate(REGRESSION_ENDPOINTS): sub[ep]=pred[:,j]
assert not sub[REGRESSION_ENDPOINTS].isna().any().any(), "NaN in predictions"
out=write_submission(sub,"regression","submissions/regression_final.parquet",load_blinded_ids())
print(f"\nwrote {out}  rows={len(sub)}  cols={list(sub.columns)}")
