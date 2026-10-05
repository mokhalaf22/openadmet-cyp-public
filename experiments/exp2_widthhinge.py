"""Experiment 2: width-weighted interval hinge.

    python experiments/exp2_widthhinge.py --leg {w1,w2}

§46b localised 85% of ST-RAE to the three narrowest credible-interval quartiles, while the
hinge weights every row equally. Here the per-row loss is weighted by 1/(1+width)^p so learning
concentrates where the metric actually charges us. `interval_hinge`'s mask multiplies the
per-element loss and then normalises by the mask sum, so passing a fractional mask IS a weighted
mean -- no change to the loss function is needed.

  w1 : weight = 1/(1+width)
  w2 : weight = 1/(1+width)^2   (sharper, to see whether the effect scales)

We already use 1/(1+width) for LightGBM sample weights (src/cyp/baseline.py) but never inside
the hinge. Architecture/folds/seeds/features are identical to the Phase-2 control, so
experiments/p2_baseline_oof.npy (0.6059 / 0.4131) is the in-run control.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

from cyp.losses import interval_hinge, st_rae as st_rae_torch

RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(10)
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/cyp-challenge-train-test"
ISOS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
NF, DH, MAXEP, PATIENCE, BS = 5, 200, 300, 20, 512

from chemprop.data import BatchMolGraph  # noqa: E402
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer  # noqa: E402
from chemprop.nn import BondMessagePassing, MeanAggregation  # noqa: E402

FEAT = SimpleMoleculeMolGraphFeaturizer()


def strae(p, lo, hi, y):
    return float(st_rae_torch(*(torch.tensor(a, dtype=torch.float64) for a in (p, lo, hi, y))))


class Net(nn.Module):
    def __init__(self, nfeat: int, hidden=256, drop=0.1):
        super().__init__()
        self.mp = BondMessagePassing(d_h=DH, depth=3)
        self.agg = MeanAggregation()
        self.trunk = nn.Sequential(nn.Linear(DH + nfeat, hidden), nn.GELU(), nn.Dropout(drop))
        self.mu = nn.Linear(hidden, 4)

    def forward(self, b, pf):
        return self.mu(self.trunk(torch.cat([self.agg(self.mp(b), b.batch), pf], 1)))


POWERS = {"w1": 1.0, "w2": 2.0, "wm05": -0.5, "wm1": -1.0}
# Negative p inverts the intent: weight = (1+width)^|p|, i.e. wide rows carry MORE weight.
# §49 found harm monotone in p over {0,1,2}, implying the optimum is at or below p=0.


def run(leg: str) -> None:
    power = POWERS[leg]
    df = pd.read_csv(D / "cyp-challenge-TRAIN_TDI.csv")
    z = np.load(ROOT / "experiments/plog_gfold.npz")
    GFOLD, oko, PLOG = z["GFOLD"], z["oko"], z["plog_o"]
    Y = np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS], 1)
    LO = np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS], 1)
    HI = np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS], 1)
    M = ~np.isnan(Y)
    WID = HI - LO
    med = np.nanmedian(WID)
    WID = np.where(np.isnan(WID), med, WID)
    WT = M.astype(np.float32) / np.power(1.0 + WID, power).astype(np.float32)
    print(f"leg={leg} (power={power}) | weight range on observed cells: "
          f"{WT[M].min():.4f}-{WT[M].max():.4f} mean {WT[M].mean():.4f}", flush=True)

    valid = np.where(oko)[0]
    MG = {int(i): FEAT(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
    def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])
    pmn, psd = np.nanmean(PLOG[valid], 0), np.nanstd(PLOG[valid], 0)
    PF = np.nan_to_num((PLOG - np.where(np.isnan(pmn), 0, pmn)) / np.where(psd < 1e-6, 1, psd))

    oofs = []
    for seed in (0, 1, 2):
        npy = ROOT / f"experiments/e2_{leg}_s{seed}.npy"
        if npy.exists():
            print(f"  s{seed}: cached", flush=True); oofs.append(np.load(npy)); continue
        oof = np.full((len(df), 4), np.nan)
        for f in range(NF):
            outer = np.where((GFOLD != f) & (GFOLD >= 0))[0]
            rng = np.random.RandomState(seed * 100 + f)
            iv = outer[rng.permutation(len(outer))[:int(len(outer) * 0.15)]]
            tr = np.setdiff1d(outer, iv)
            va = np.where(GFOLD == f)[0]
            ym = np.array([Y[tr, j][M[tr, j]].mean() for j in range(4)])
            ys = np.array([Y[tr, j][M[tr, j]].std() or 1.0 for j in range(4)])
            lo_s, hi_s = np.nan_to_num((LO - ym) / ys), np.nan_to_num((HI - ym) / ys)
            torch.manual_seed(seed)
            net = Net(PF.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            g = torch.Generator().manual_seed(seed); best = (1e9, None, 0)
            for ep in range(MAXEP):
                net.train(); perm = torch.randperm(len(tr), generator=g).numpy()
                for i in range(0, len(tr), BS):
                    b = tr[perm[i:i + BS]]
                    opt.zero_grad()
                    mu = net(bmg(b), torch.tensor(PF[b], dtype=torch.float32))
                    interval_hinge(mu, torch.tensor(lo_s[b], dtype=torch.float32),
                                   torch.tensor(hi_s[b], dtype=torch.float32),
                                   torch.tensor(WT[b])).backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    mv = net(bmg(iv), torch.tensor(PF[iv], dtype=torch.float32)).numpy() * ys + ym
                vals = [strae(mv[M[iv, j], j], LO[iv, j][M[iv, j]], HI[iv, j][M[iv, j]],
                              Y[iv, j][M[iv, j]]) for j in range(4) if M[iv, j].sum()]
                ivs = float(np.mean(vals))
                if ivs < best[0] - 1e-4: best = (ivs, copy.deepcopy(net.state_dict()), ep)
                elif ep - best[2] >= PATIENCE: break
            net.load_state_dict(best[1]); net.eval()
            with torch.no_grad():
                oof[va] = net(bmg(va), torch.tensor(PF[va], dtype=torch.float32)).numpy() * ys + ym
            print(f"  {leg} s{seed} fold{f}: best@{best[2]+1}ep", flush=True)
        np.save(npy, oof); oofs.append(oof)
    np.save(ROOT / f"experiments/e2_{leg}_oof.npy", np.nanmean(oofs, 0))
    print(f"saved experiments/e2_{leg}_oof.npy", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", required=True, choices=list(POWERS))
    run(ap.parse_args().leg)
