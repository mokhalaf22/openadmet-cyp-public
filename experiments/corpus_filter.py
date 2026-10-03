"""Phase 1: filter retrieved neighbours into the admitted corpus, and report GATE 1.

Pipeline
  1. gather SmallWorld hits (structures only) from experiments/sw_cache/
  2. canonicalize, dedupe by InChIKey
  3. exclude challenge train/test structures and the 6 quarantined Octant overlaps
  4. keep candidates inside the blinded set's physicochemical envelope (1st-99th pct)
  5. alert screen with REOS under the SUBTRACTIVE VETO: any alert that fires on even one
     blinded compound is dropped from the filter, so we never prune chemistry the test set
     itself contains (iterated to convergence, since REOS reports the first match only)
  6. write experiments/corpus_phase1.csv and report GATE 1 metrics

Similarity for the gate is computed locally (RDKit Morgan r=2, 2048 bits) for comparability
with the prior ChEMBL/PubChem measurement (324 compounds / 13.3% density).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, DataStructs, Descriptors, inchi, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "experiments" / "sw_cache"
OUT = ROOT / "experiments" / "corpus_phase1.csv"
GATE_JSON = ROOT / "experiments" / "gate1.json"
D = ROOT / "data/cyp-challenge-train-test"
ANCHOR_T = 0.7
MIN_CORPUS, MIN_DENSITY = 10_000, 0.40  # gate thresholds from the plan

DESCS = {
    "mw": Descriptors.MolWt, "clogp": Crippen.MolLogP, "tpsa": rdMolDescriptors.CalcTPSA,
    "hbd": rdMolDescriptors.CalcNumHBD, "hba": rdMolDescriptors.CalcNumHBA,
    "rotb": rdMolDescriptors.CalcNumRotatableBonds, "rings": rdMolDescriptors.CalcNumRings,
}


def _fp(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def _mols(smis):
    for s in smis:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            yield s, m


def _ikset(smis) -> set[str]:
    return {inchi.MolToInchiKey(m) for _, m in _mols(smis)}


def _envelope(mols) -> dict[str, tuple[float, float]]:
    vals = {k: [] for k in DESCS}
    for m in mols:
        for k, f in DESCS.items():
            vals[k].append(float(f(m)))
    return {k: (float(np.percentile(v, 1)), float(np.percentile(v, 99))) for k, v in vals.items()}


def _in_envelope(m, env) -> bool:
    return all(env[k][0] <= float(f(m)) <= env[k][1] for k, f in DESCS.items())


def _veto_rules(blinded_mols):
    """REOS with every alert that fires on a blinded compound removed (iterated)."""
    import useful_rdkit_utils as uru

    reos = uru.REOS()
    reos.set_active_rule_sets(list(reos.get_available_rule_sets()))
    dropped: list[str] = []
    # process_mol returns (rule_set_name, description) when an alert fires, and ('ok','ok')
    # when it passes. drop_rule() keys on the DESCRIPTION, so the veto must use res[1] --
    # passing res[0] (the set name) silently drops nothing and vetoes no alert at all.
    for _ in range(500):  # iterate: only the first matching alert per molecule is reported
        firing = set()
        for m in blinded_mols:
            try:
                res = reos.process_mol(m)
            except Exception:
                continue
            if not (isinstance(res, (tuple, list)) and len(res) >= 2):
                continue
            rule_set, desc = str(res[0]), str(res[1])
            if desc.lower() in ("ok", "none") or rule_set.lower() == "ok":
                continue
            firing.add(desc)
        if not firing:
            break
        for desc in firing:
            try:
                reos.drop_rule(desc)
                dropped.append(desc)
            except Exception:
                pass
    return reos, dropped


def build() -> dict:
    bl = pd.read_csv(D / "cyp-challenge-TEST-BLINDED.csv")
    tr = pd.read_csv(D / "cyp-challenge-TRAIN_TDI.csv")
    bl_pairs = list(_mols(bl["SMILES"]))
    bl_mols = [m for _, m in bl_pairs]
    bl_fps = [_fp(m) for m in bl_mols]
    print(f"blinded: {len(bl_mols)} parsed", flush=True)

    # 1-2. gather + dedupe
    cand: dict[str, tuple[str, float]] = {}  # ik -> (canonical smiles, best service ecfp4)
    files = sorted(CACHE.glob("OCNT-*.json"))
    raw = 0
    for f in files:
        for h in (json.loads(f.read_text()) or {}).get("hits") or []:
            s = h.get("smiles")
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            if m is None:
                continue
            raw += 1
            ik = inchi.MolToInchiKey(m)
            e = h.get("ecfp4") or 0.0
            if ik not in cand or e > cand[ik][1]:
                cand[ik] = (Chem.MolToSmiles(m), float(e))
    print(f"queries cached: {len(files)} | raw hits: {raw} | unique structures: {len(cand)}", flush=True)

    # 3. exclusions
    excl = _ikset(bl["SMILES"]) | _ikset(tr["SMILES"])
    qpath = ROOT / "data" / "octant_quarantine.csv"
    n_quar = 0
    if qpath.exists():
        qn = set(pd.read_csv(qpath)["Molecule_Name"])
        excl |= _ikset(bl[bl["Molecule_Name"].isin(qn)]["SMILES"])
        n_quar = len(qn)
    cand = {k: v for k, v in cand.items() if k not in excl}
    print(f"after excluding challenge train/test + {n_quar} quarantined: {len(cand)}", flush=True)

    # 4. physicochemical envelope of the blinded set
    env = _envelope(bl_mols)
    keep = {}
    for ik, (s, e) in cand.items():
        m = Chem.MolFromSmiles(s)
        if m is not None and _in_envelope(m, env):
            keep[ik] = (s, e, m)
    print(f"after envelope filter: {len(keep)}", flush=True)

    # 5. subtractive-veto alert screen
    reos, dropped = _veto_rules(bl_mols)
    print(f"subtractive veto dropped {len(dropped)} alerts that fire on blinded compounds"
          f"{': ' + ', '.join(sorted(set(dropped))[:12]) if dropped else ''}", flush=True)
    admitted = []
    for ik, (s, e, m) in keep.items():
        try:
            res = reos.process_mol(m)
            rule = res[0] if isinstance(res, (tuple, list)) and res else None
            if rule and str(rule).lower() not in ("ok", "none"):
                continue
        except Exception:
            pass
        admitted.append((s, e))
    print(f"after alert screen: {len(admitted)}", flush=True)

    # 6. gate metrics (local ECFP4, comparable to the prior 13.3% measurement)
    adm_fps = [_fp(Chem.MolFromSmiles(s)) for s, _ in admitted]
    nn = []
    for bf in bl_fps:
        nn.append(max(DataStructs.BulkTanimotoSimilarity(bf, adm_fps)) if adm_fps else 0.0)
    nn = np.asarray(nn)
    density = float((nn >= ANCHOR_T).mean())
    pd.DataFrame({"smiles": [s for s, _ in admitted],
                  "service_ecfp4": [e for _, e in admitted]}).to_csv(OUT, index=False)

    gate = {
        "corpus_size": len(admitted),
        "anchor_density_0.7": round(density, 4),
        "median_nn_tanimoto": round(float(np.median(nn)), 4),
        "mean_nn_tanimoto": round(float(nn.mean()), 4),
        "frac_nn_ge_0.5": round(float((nn >= 0.5).mean()), 4),
        "queries_cached": len(files), "raw_hits": raw, "unique_structures": len(cand),
        "dropped_alerts": sorted(set(dropped)),
        "passes_gate": bool(len(admitted) >= MIN_CORPUS and density >= MIN_DENSITY),
    }
    GATE_JSON.write_text(json.dumps(gate, indent=1))
    return gate


def gate1_metrics() -> dict:
    g = json.loads(GATE_JSON.read_text())
    return {k: g[k] for k in ("corpus_size", "anchor_density_0.7", "median_nn_tanimoto",
                              "frac_nn_ge_0.5", "passes_gate") if k in g}


def gate1_report() -> str:
    g = json.loads(GATE_JSON.read_text())
    verdict = ("PASS — proceed to phase 2 on review" if g["passes_gate"] else
               f"FAIL — under the {MIN_CORPUS:,}-compound / {MIN_DENSITY:.0%} thresholds; "
               "the rest of the plan is premised on retrieval working")
    return (f"  admitted corpus size        : {g['corpus_size']:,}\n"
            f"  blind anchor density @0.7   : {g['anchor_density_0.7']:.1%}\n"
            f"  median NN Tanimoto (blind→corpus): {g['median_nn_tanimoto']:.3f}\n"
            f"  (prior ChEMBL/PubChem attempt: 324 compounds / 13.3% density)\n"
            f"  verdict: {verdict}")


if __name__ == "__main__":
    g = build()
    print("\n=== GATE 1 ===")
    print(gate1_report())
