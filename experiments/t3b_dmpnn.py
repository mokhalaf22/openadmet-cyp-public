"""Decorrelated members (b) and (c): CYP2D6 D-MPNN specialists.

    python experiments/t3b_dmpnn.py --leg {bag,noplog}

bag     -- same outer GFOLD fold, but trained on a BAGGED 70% subsample of that fold's
           training rows with a different seed stream. The brief asked for "a different fold
           partition"; a genuinely different OUTER partition would train on GFOLD fold-f rows
           and then be scored on them, which leaks. Bagging changes training composition while
           keeping the OOF valid.
noplog  -- no predicted-primary feature at all (nfeat=0). §31b identified that feature as what
           made our D-MPNN and LightGBM converge, so removing it is the most direct lever on
           member correlation.

Both are CYP2D6-only, chosen for decorrelation rather than individual strength.
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
ISO, NF, DH, MAXEP, PATIENCE, BS = "CYP2D6", 5, 200, 300, 20, 512
BAG = 0.70

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
        self.nfeat = nfeat
        self.trunk = nn.Sequential(nn.Linear(DH + nfeat, hidden), nn.GELU(), nn.Dropout(drop))
        self.mu = nn.Linear(hidden, 1)

    def forward(self, b, pf):
        g = self.agg(self.mp(b), b.batch)
        h = self.trunk(torch.cat([g, pf], 1) if self.nfeat else g)
        return self.mu(h)


def run(leg: str) -> None:
    df = pd.read_csv(D / "cyp-challenge-TRAIN_TDI.csv")
    z = np.load(ROOT / "experiments/plog_gfold.npz")
    GFOLD, oko, PLOG = z["GFOLD"], z["oko"], z["plog_o"]
    y = df[f"{ISO}_pIC50_direct_inhibition"].to_numpy(float)
    lo = df[f"{ISO}_pIC50_direct_inhibition_conf_low"].to_numpy(float)
    hi = df[f"{ISO}_pIC50_direct_inhibition_conf_high"].to_numpy(float)
    M = ~np.isnan(y)
    pres = M & oko
    valid = np.where(oko)[0]
    MG = {int(i): FEAT(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
    def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])
    use_pf = leg != "noplog"
    pmn, psd = np.nanmean(PLOG[valid], 0), np.nanstd(PLOG[valid], 0)
    PF = np.nan_to_num((PLOG - np.where(np.isnan(pmn), 0, pmn)) / np.where(psd < 1e-6, 1, psd))
    nfeat = PF.shape[1] if use_pf else 0
    def pf(rows):
        return torch.tensor(PF[rows], dtype=torch.float32) if use_pf else torch.zeros(len(rows), 0)
    print(f"leg={leg} | rows={int(pres.sum())} | nfeat={nfeat} | bag={BAG if leg=='bag' else '-'}", flush=True)

    oofs = []
    for seed in (0, 1, 2):
        npy = ROOT / f"experiments/t3b_{leg}_s{seed}.npy"
        if npy.exists():
            print(f"  s{seed}: cached", flush=True); oofs.append(np.load(npy)); continue
        oof = np.full(len(df), np.nan)
        for f in range(NF):
            outer = np.where(pres & (GFOLD != f) & (GFOLD >= 0))[0]
            # different seed stream for bagging, so members see different compositions
            rng = np.random.RandomState((seed + 1) * 7919 + f if leg == "bag" else seed * 100 + f)
            if leg == "bag":
                outer = rng.choice(outer, size=int(len(outer) * BAG), replace=False)
            iv = outer[rng.permutation(len(outer))[:int(len(outer) * 0.15)]]
            tr = np.setdiff1d(outer, iv)
            va = np.where(pres & (GFOLD == f))[0]
            if not len(tr) or not len(va):
                continue
            ym, ys = y[tr].mean(), (y[tr].std() or 1.0)
            lo_s, hi_s = np.nan_to_num((lo - ym) / ys), np.nan_to_num((hi - ym) / ys)
            torch.manual_seed(seed if leg != "bag" else seed + 101)
            net = Net(nfeat)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            gg = torch.Generator().manual_seed(seed); best = (1e9, None, 0)
            for ep in range(MAXEP):
                net.train(); perm = torch.randperm(len(tr), generator=gg).numpy()
                for i in range(0, len(tr), BS):
                    b = tr[perm[i:i + BS]]
                    opt.zero_grad()
                    mu = net(bmg(b), pf(b))[:, 0]
                    interval_hinge(mu, torch.tensor(lo_s[b], dtype=torch.float32),
                                   torch.tensor(hi_s[b], dtype=torch.float32),
                                   torch.ones(len(b))).backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    mv = net(bmg(iv), pf(iv))[:, 0].numpy() * ys + ym
                ivs = strae(mv, lo[iv], hi[iv], y[iv])
                if ivs < best[0] - 1e-4: best = (ivs, copy.deepcopy(net.state_dict()), ep)
                elif ep - best[2] >= PATIENCE: break
            net.load_state_dict(best[1]); net.eval()
            with torch.no_grad():
                oof[va] = net(bmg(va), pf(va))[:, 0].numpy() * ys + ym
            print(f"  {leg} s{seed} fold{f}: best@{best[2]+1}ep", flush=True)
        np.save(npy, oof); oofs.append(oof)
    np.save(ROOT / f"experiments/t3b_{leg}_oof.npy", np.nanmean(oofs, 0))
    print(f"saved experiments/t3b_{leg}_oof.npy", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", required=True, choices=["bag", "noplog"])
    run(ap.parse_args().leg)
