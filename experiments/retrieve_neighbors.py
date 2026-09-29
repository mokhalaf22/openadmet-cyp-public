"""§35 step 2A: retrieve structural near-neighbours of the 750 blinded compounds from
ChEMBL and PubChem (structures only; NO assay data pulled). Resumable per-compound cache.

For each blinded SMILES: ChEMBL similarity API @70% and PubChem fastsimilarity_2d @70%.
Collect neighbour SMILES only. Downstream (build_corpus) canonicalizes, excludes challenge
train/test + the §34 quarantine, and admits those with max-Tanimoto-to-blind > 0.50
(the training set's own mean NN, so admitted are closer to the blind set than we are).
"""
import sys, os, json, time, urllib.parse, urllib.request
import pandas as pd
sys.stdout.reconfigure(line_buffering=True)
bl=pd.read_csv("data/cyp-challenge-train-test/cyp-challenge-TEST-BLINDED.csv")
CACHE="experiments/neighbors_cache"; os.makedirs(CACHE,exist_ok=True)
def get(url,timeout=30):
    try:
        with urllib.request.urlopen(url,timeout=timeout) as r: return json.loads(r.read().decode())
    except Exception as e: return {"__err__":f"{type(e).__name__}:{str(e)[:80]}"}
def chembl(smi):
    u=f"https://www.ebi.ac.uk/chembl/api/data/similarity/{urllib.parse.quote(smi,safe='')}/70.json?limit=100&only=molecule_structures"
    d=get(u); out=[]
    for m in (d.get("molecules") or []):
        st=m.get("molecule_structures") or {}; cs=st.get("canonical_smiles")
        if cs: out.append(cs)
    return out
def pubchem(smi):
    u=f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{urllib.parse.quote(smi,safe='')}/property/CanonicalSMILES/JSON?Threshold=70&MaxRecords=100"
    d=get(u); props=(d.get("PropertyTable") or {}).get("Properties") or []
    return [p.get("CanonicalSMILES") for p in props if p.get("CanonicalSMILES")]
for i,(name,smi) in enumerate(zip(bl["Molecule_Name"],bl["SMILES"])):
    fp=f"{CACHE}/{name}.json"
    if os.path.exists(fp): continue
    rec={"name":name,"smiles":smi,"chembl":chembl(smi)}; time.sleep(0.25)
    rec["pubchem"]=pubchem(smi); time.sleep(0.25)
    json.dump(rec,open(fp,"w"))
    if i%50==0: print(f"[{i}/{len(bl)}] {name}: chembl {len(rec['chembl'])} pubchem {len(rec['pubchem'])}",flush=True)
done=len([f for f in os.listdir(CACHE) if f.endswith('.json')])
print(f"done: {done}/{len(bl)} compounds cached")
