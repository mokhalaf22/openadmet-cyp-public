"""Where does the Phase-1 corpus sit: near the blinded set only, or near training too?

The warm-start premise needs the corpus to BRIDGE both. If it neighbours only the blinded
set, pretraining moves the encoder toward a region with no labels and the DRC fine-tune has
to undo it; if it neighbours both, the representation it learns is one the labelled data can
actually refine. Reports nearest-neighbour ECFP4 (Morgan r=2, 2048) from each TRAINING
compound to the corpus, alongside the blinded-set numbers, plus the internal baselines.

Writes experiments/corpus_similarity.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/cyp-challenge-train-test"
OUT = ROOT / "experiments" / "corpus_similarity.json"


def fps(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    return out


def nn_stats(query_fps, pool_fps, label: str) -> dict:
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(q, pool_fps)) for q in query_fps])
    s = {
        "n_query": len(nn),
        "median_nn": round(float(np.median(nn)), 4),
        "mean_nn": round(float(nn.mean()), 4),
        "frac_ge_0.7": round(float((nn >= 0.7).mean()), 4),
        "frac_ge_0.5": round(float((nn >= 0.5).mean()), 4),
        "p10_nn": round(float(np.percentile(nn, 10)), 4),
        "p90_nn": round(float(np.percentile(nn, 90)), 4),
    }
    print(f"{label}: median {s['median_nn']:.3f} mean {s['mean_nn']:.3f} "
          f"| >=0.7 {s['frac_ge_0.7']:.1%} | >=0.5 {s['frac_ge_0.5']:.1%} "
          f"| p10 {s['p10_nn']:.3f} p90 {s['p90_nn']:.3f}", flush=True)
    return s


def main() -> dict:
    corpus = pd.read_csv(ROOT / "experiments/corpus_phase1.csv")["smiles"].tolist()
    tr = pd.read_csv(D / "cyp-challenge-TRAIN_TDI.csv")["SMILES"].tolist()
    bl = pd.read_csv(D / "cyp-challenge-TEST-BLINDED.csv")["SMILES"].tolist()
    print(f"fingerprinting corpus ({len(corpus)})...", flush=True)
    cf = fps(corpus)
    tf, bf = fps(tr), fps(bl)
    print(f"train {len(tf)} | blinded {len(bf)} | corpus {len(cf)}\n", flush=True)

    res = {
        "corpus_size": len(cf),
        "train_to_corpus": nn_stats(tf, cf, "train  -> corpus "),
        "blind_to_corpus": nn_stats(bf, cf, "blinded-> corpus "),
        # internal baselines for scale
        "train_to_blind": nn_stats(tf, bf, "train  -> blinded"),
        "blind_to_train": nn_stats(bf, tf, "blinded-> train  "),
    }
    OUT.write_text(json.dumps(res, indent=1))
    t, b = res["train_to_corpus"], res["blind_to_corpus"]
    gap = b["median_nn"] - t["median_nn"]
    print(f"\nbridge check: blinded-NN {b['median_nn']:.3f} vs training-NN {t['median_nn']:.3f} "
          f"(gap {gap:+.3f})")
    print("  -> corpus bridges both" if abs(gap) < 0.10 and t["frac_ge_0.7"] >= 0.5 else
          "  -> corpus is markedly closer to the blinded set than to training")
    return res


if __name__ == "__main__":
    main()
