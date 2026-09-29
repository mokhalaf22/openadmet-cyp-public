"""§35 step 2B: build the warm-start corpus from retrieved neighbours.

Aggregate ChEMBL+PubChem neighbour SMILES, canonicalize, drop challenge train/test and the
§34 quarantine (by InChIKey), admit those with max-Tanimoto-to-blind > 0.50 (train mean NN).
Report admitted corpus size and blind anchor density at Tanimoto 0.7 (the step-3 SQRL gate).
"""
import os, json, glob, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, inchi
RDLogger.DisableLog("rdApp.*")
bl=pd.read_csv("data/cyp-challenge-train-test/cyp-challenge-TEST-BLINDED.csv")
tr=pd.read_csv("data/cyp-challenge-train-test/cyp-challenge-TRAIN_TDI.csv")
def ikset(smis):
    s=set()
    for x in smis:
        m=Chem.MolFromSmiles(x) if isinstance(x,str) else None
        if m: s.add(inchi.MolToInchiKey(m))
    return s
excl=ikset(bl["SMILES"])|ikset(tr["SMILES"])  # challenge test+train (test-exclusion also covers the 6 quarantined)
def fp(s):
    m=Chem.MolFromSmiles(s) if isinstance(s,str) else None
    return (m, AllChem.GetMorganFingerprintAsBitVect(m,2,2048)) if m else (None,None)
blfps=[fp(s)[1] for s in bl["SMILES"]]; blfps=[f for f in blfps if f]
# gather + dedupe candidates
cand={}
for fpath in glob.glob("experiments/neighbors_cache/*.json"):
    r=json.load(open(fpath))
    for s in (r.get("chembl") or [])+(r.get("pubchem") or []):
        m=Chem.MolFromSmiles(s) if isinstance(s,str) else None
        if not m: continue
        ik=inchi.MolToInchiKey(m)
        if ik in excl or ik in cand: continue
        cand[ik]=(Chem.MolToSmiles(m), AllChem.GetMorganFingerprintAsBitVect(m,2,2048))
print(f"raw unique neighbour structures (excl. challenge train/test): {len(cand)}")
# admit by max-Tanimoto-to-blind > 0.50; record per-candidate maxsim
adm=[]
for ik,(cs,cfp) in cand.items():
    mx=max(DataStructs.BulkTanimotoSimilarity(cfp,blfps))
    if mx>0.50: adm.append((cs,cfp,mx))
print(f"admitted (max-blind-Tanimoto > 0.50): {len(adm)}")
for thr in [0.5,0.6,0.7,0.8]:
    print(f"  admitted with max-blind-Tanimoto >= {thr}: {sum(1 for _,_,mx in adm if mx>=thr)}")
# anchor density: fraction of the 750 blinded with >=1 admitted neighbour at Tanimoto>=thr
admfps=[f for _,f,_ in adm]
for thr in [0.7]:
    cov=sum(1 for bf in blfps if admfps and max(DataStructs.BulkTanimotoSimilarity(bf,admfps))>=thr)
    print(f"BLIND ANCHOR DENSITY @ Tanimoto {thr}: {cov}/{len(blfps)} = {100*cov/len(blfps):.1f}%  (step-3 gate ~30%)")
pd.DataFrame({"smiles":[cs for cs,_,_ in adm],"max_blind_tanimoto":[round(mx,3) for _,_,mx in adm]}).to_csv("experiments/warmstart_corpus.csv",index=False)
print(f"saved experiments/warmstart_corpus.csv ({len(adm)} compounds)")
