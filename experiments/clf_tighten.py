"""Tighten CYP3A4 TDI threshold (§29a): trade recall for precision.

Live board (§29) says we over-call positives — precision 0.39 at recall 0.50, vs the
field's 0.53-0.71 precision at similar recall. CYP3A4 @ 0.45 predicts 32.7% positive
vs an implied field rate ~17%. Sweep CYP3A4 @ {0.45(current), 0.55, 0.65}, CYP2D6
held @ 0.30, and report: predicted-positive rate (OOF + blinded) and OOF MCC cost, so
the simulated give-up is visible against a possible real precision gain on blind.

Uses baseline's stored probabilities (data/baseline_{oof,test}_predictions.csv);
OOF evaluation set = the trainable rows baseline scored (non-NaN proba_oof =
assigned-negatives already excluded, per guards.py). MCC via sklearn, matching baseline.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from sklearn.metrics import matthews_corrcoef
from cyp.submit import write_submission, load_blinded_ids
D="data/cyp-challenge-train-test/"
df=pd.read_csv(D+"cyp-challenge-TRAIN_TDI.csv")
oof=pd.read_csv("data/baseline_oof_predictions.csv")
test=pd.read_csv("data/baseline_test_predictions.csv")
ID="Molecule_Name"
def mcc(yt,yp): return matthews_corrcoef(yt,yp)

def oof_eval(iso):
    m=oof.merge(df[[ID,f"{iso}_is_TDI"]],on=ID,how="left")
    p=m[f"{iso}_tdi_proba_oof"].to_numpy(float); y=m[f"{iso}_is_TDI"]
    ev=~np.isnan(p) & y.notna().to_numpy()
    return p[ev], y[ev].astype(bool).astype(int).to_numpy()

p3,y3=oof_eval("CYP3A4"); p2,y2=oof_eval("CYP2D6")
t3=test["CYP3A4_tdi_proba"].to_numpy(float); t2=test["CYP2D6_tdi_proba"].to_numpy(float)
print(f"OOF eval n: CYP3A4={len(y3)} (pos {y3.mean()*100:.1f}%)  CYP2D6={len(y2)} (pos {y2.mean()*100:.1f}%)")
print(f"blinded n=750\n")

CYP2D6_THR=0.30
m2=mcc(y2,(p2>=CYP2D6_THR).astype(int))
print(f"CYP2D6 @ {CYP2D6_THR} (held): OOF MCC={m2:.3f}  OOF pos%={100*(p2>=CYP2D6_THR).mean():.1f}  blinded pos%={100*(t2>=CYP2D6_THR).mean():.1f}")
print(f"\nCYP3A4 threshold sweep (CYP2D6 held @ {CYP2D6_THR}):")
print(f"{'thr':>5} {'OOF MCC 3A4':>11} {'OOF pos%':>9} {'blinded pos%':>12} {'macro OOF MCC':>14}")
rows={}
for thr in [0.45,0.55,0.65]:
    m3=mcc(y3,(p3>=thr).astype(int))
    macro=(m2+m3)/2
    rows[thr]=dict(mcc3=m3,oofpos=100*(p3>=thr).mean(),blindpos=100*(t3>=thr).mean(),macro=macro)
    print(f"{thr:>5.2f} {m3:>11.3f} {rows[thr]['oofpos']:>8.1f}% {rows[thr]['blindpos']:>11.1f}% {macro:>14.3f}")

# build tightened submissions for 0.55 and 0.65
for thr in [0.55,0.65]:
    sub=test[[ID,"SMILES"]].copy() if "SMILES" in test.columns else pd.read_csv(D+"cyp-challenge-TEST-BLINDED.csv")[["Molecule_Name","SMILES"]].copy()
    sub=sub.rename(columns={})
    out=pd.DataFrame({"SMILES":sub["SMILES"],"Molecule_Name":sub[ID]})
    out["CYP2D6_is_TDI"]=(t2>=CYP2D6_THR)
    out["CYP3A4_is_TDI"]=(t3>=thr)
    tag=str(thr).replace("0.","0")
    path=write_submission(out,"classification",f"submissions/classification_3a4_{tag}.parquet",load_blinded_ids())
    print(f"wrote {path}  CYP2D6 pos {int(out['CYP2D6_is_TDI'].sum())}  CYP3A4 pos {int(out['CYP3A4_is_TDI'].sum())}")
print("done")
