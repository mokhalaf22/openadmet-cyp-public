"""Phase 2: fetch the Tox21 CYP panel from PubChem as SEPARATE auxiliary heads.

Tox21 covers CYP1A2, CYP2C9 and CYP2D6 (no CYP3A4 assay under this panel). Each isoform
becomes its own column / own head -- never merged into or rescaled onto a scored challenge
column (project hard rule 2). Values are pAC50 (-log10 of AC50 in molar) where the assay
reports a potency, which is a different assay and scale from the challenge DRC pIC50; the
separate head is what makes that safe.

Writes data/external/tox21_cyp_auxhead.csv with an InChIKey column plus tox21_<ISO> columns.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/external/tox21_cyp_auxhead.csv"
AIDS = {"CYP1A2": [2284393, 1671199], "CYP2C9": [2284391, 1671198], "CYP2D6": [2284394, 1671196]}
POTENCY_COLS = ("Potency", "AC50", "EC50", "IC50", "Activity_Potency")


def _get(url: str, tries: int = 3, timeout: int = 90):
    for a in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(3 * (a + 1))
    return None


def assay_table(aid: int) -> pd.DataFrame | None:
    r = _get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV")
    if r is None:
        print(f"  AID {aid}: download failed", flush=True)
        return None
    try:
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    except Exception as exc:
        print(f"  AID {aid}: parse failed ({exc})", flush=True)
        return None
    df = df[pd.to_numeric(df.get("PUBCHEM_CID"), errors="coerce").notna()]
    print(f"  AID {aid}: {len(df)} rows, cols incl "
          f"{[c for c in df.columns if c in POTENCY_COLS] or 'no potency col'}", flush=True)
    return df


def cid_smiles(cids: list[int], batch: int = 200) -> dict[int, str]:
    """CID -> SMILES via POST (a GET URL with hundreds of CIDs is rejected as too long)."""
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/SMILES/CSV"
    out: dict[int, str] = {}
    fails = 0
    for i in range(0, len(cids), batch):
        body = {"cid": ",".join(str(c) for c in cids[i:i + batch])}
        got = False
        for attempt in range(3):
            try:
                r = requests.post(url, data=body, timeout=90)
                if r.status_code == 200:
                    t = pd.read_csv(io.StringIO(r.text))
                    scol = next((c for c in t.columns if "SMILES" in c.upper()), None)
                    if scol:
                        out.update(dict(zip(t["CID"].astype(int), t[scol])))
                        got = True
                    break
                if attempt == 2:
                    print(f"    batch {i}: HTTP {r.status_code} {r.text[:80]}", flush=True)
            except Exception as exc:
                if attempt == 2:
                    print(f"    batch {i}: {type(exc).__name__}", flush=True)
            time.sleep(3 * (attempt + 1))
        fails += 0 if got else 1
        if (i // batch) % 10 == 0:
            print(f"    smiles {len(out)}/{len(cids)} (failed batches {fails})", flush=True)
        time.sleep(0.25)
    return out


def main() -> None:
    per_iso: dict[str, dict[int, float]] = {}
    for iso, aids in AIDS.items():
        vals: dict[int, float] = {}
        for aid in aids:
            df = assay_table(aid)
            if df is None:
                continue
            pcol = next((c for c in POTENCY_COLS if c in df.columns), None)
            for _, r in df.iterrows():
                cid = int(r["PUBCHEM_CID"])
                v = None
                if pcol is not None and pd.notna(r.get(pcol)):
                    try:  # PubChem potency is micromolar -> pAC50
                        um = float(r[pcol])
                        if um > 0:
                            v = -(__import__("math").log10(um * 1e-6))
                    except Exception:
                        v = None
                if v is None:
                    out = str(r.get("PUBCHEM_ACTIVITY_OUTCOME", ""))
                    if out.lower() == "inactive":
                        v = 4.0  # inactive floor on the pAC50 scale (own head, own scale)
                if v is not None:
                    vals.setdefault(cid, v)
        per_iso[iso] = vals
        print(f"{iso}: {len(vals)} compounds with a value", flush=True)

    all_cids = sorted({c for v in per_iso.values() for c in v})
    print(f"resolving SMILES for {len(all_cids)} CIDs...", flush=True)
    smi = cid_smiles(all_cids)
    print(f"  resolved {len(smi)}", flush=True)

    rows = []
    for cid, s in smi.items():
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            continue
        rec = {"ik": inchi.MolToInchiKey(m), "cid": cid}
        for iso in AIDS:
            rec[f"tox21_{iso}"] = per_iso[iso].get(cid)
        rows.append(rec)
    out = pd.DataFrame(rows).dropna(subset=[f"tox21_{i}" for i in AIDS], how="all")
    out = out.drop_duplicates("ik")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}: {len(out)} compounds")
    for iso in AIDS:
        print(f"  tox21_{iso}: {int(out[f'tox21_{iso}'].notna().sum())} values")

    # how much of it actually touches our training chemistry?
    tr = pd.read_csv(ROOT / "data/cyp-challenge-train-test/cyp-challenge-TRAIN_TDI.csv")
    tr_ik = {inchi.MolToInchiKey(m) for m in
             (Chem.MolFromSmiles(s) for s in tr["SMILES"]) if m is not None}
    print(f"  overlap with challenge train rows: {int(out['ik'].isin(tr_ik).sum())}")


if __name__ == "__main__":
    main()
