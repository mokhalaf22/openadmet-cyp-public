"""Experiment 3: CYP2D6 target transform (prior pre-registered in FINDINGS §50).

    python experiments/exp3_cyp2d6_target.py --leg {rank,ordinal}

rank     -- CYP2D6's target replaced by its within-isoform rank (mapped to [0,1] on the fold's
            training rows), head trained with L1. Ordering is what Spearman measures, so train
            on order directly.
ordinal  -- CYP2D6 binned into K quantile bins; the head emits K-1 cumulative logits and is
            trained with BCE against the indicators 1[y > t_k] (a cumulative-link / ordinal
            model). The ranking score is sum(sigmoid(logits)) = expected number of thresholds
            exceeded.

Both keep the SHARED 4-isoform model and change only CYP2D6's head; the other three isoforms keep
the interval hinge (§45: replacing CYP2D6 with a specialist costs −0.038, more than a transform is
likely to win).

Both transforms discard CYP2D6's pIC50 scale, so ST-RAE is not directly defined. We recover it by
mapping the predicted rank back through the empirical quantile function of that fold's CYP2D6
training labels — reported, but the honest primary metric here is Spearman.
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
from scipy.stats import rankdata, spearmanr

from cyp.losses import interval_hinge, st_rae as st_rae_torch

RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(10)
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/cyp-challenge-train-test"
ISOS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
J2D6 = 2
NF, DH, MAXEP, PATIENCE, BS = 5, 200, 300, 20, 512
NBINS = 10

from chemprop.data import BatchMolGraph  # noqa: E402
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer  # noqa: E402
from chemprop.nn import BondMessagePassing, MeanAggregation  # noqa: E402

FEAT = SimpleMoleculeMolGraphFeaturizer()


def strae(p, lo, hi, y):
    return float(st_rae_torch(*(torch.tensor(a, dtype=torch.float64) for a in (p, lo, hi, y))))


class Net(nn.Module):
    """3 interval heads (other isoforms) + a CYP2D6 head of n2d6 outputs."""

    def __init__(self, nfeat: int, n2d6: int, hidden=256, drop=0.1):
        super().__init__()
        self.mp = BondMessagePassing(d_h=DH, depth=3)
        self.agg = MeanAggregation()
        self.trunk = nn.Sequential(nn.Linear(DH + nfeat, hidden), nn.GELU(), nn.Dropout(drop))
        self.mu = nn.Linear(hidden, 4)        # all four on the pIC50 scale (CYP2D6 col unused)
        self.d6 = nn.Linear(hidden, n2d6)     # CYP2D6 replacement head

    def forward(self, b, pf):
        h = self.trunk(torch.cat([self.agg(self.mp(b), b.batch), pf], 1))
        return self.mu(h), self.d6(h)


def run(leg: str) -> None:
    n_out = 1 if leg == "rank" else NBINS - 1
    df = pd.read_csv(D / "cyp-challenge-TRAIN_TDI.csv")
    z = np.load(ROOT / "experiments/plog_gfold.npz")
    GFOLD, oko, PLOG = z["GFOLD"], z["oko"], z["plog_o"]
    Y = np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS], 1)
    LO = np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS], 1)
    HI = np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS], 1)
    M = ~np.isnan(Y)
    valid = np.where(oko)[0]
    MG = {int(i): FEAT(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
    def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])
    pmn, psd = np.nanmean(PLOG[valid], 0), np.nanstd(PLOG[valid], 0)
    PF = np.nan_to_num((PLOG - np.where(np.isnan(pmn), 0, pmn)) / np.where(psd < 1e-6, 1, psd))
    bce = nn.BCEWithLogitsLoss(reduction="none")
    print(f"leg={leg} | CYP2D6 head outputs={n_out}", flush=True)

    oof_score = []   # ranking score (for Spearman)
    oof_pic50 = []   # mapped back to pIC50 (for ST-RAE)
    for seed in (0, 1, 2):
        sc = np.full(len(df), np.nan); pc = np.full(len(df), np.nan)
        for f in range(NF):
            outer = np.where((GFOLD != f) & (GFOLD >= 0))[0]
            rng = np.random.RandomState(seed * 100 + f)
            iv = outer[rng.permutation(len(outer))[:int(len(outer) * 0.15)]]
            tr = np.setdiff1d(outer, iv)
            va = np.where(GFOLD == f)[0]
            ym = np.array([Y[tr, j][M[tr, j]].mean() for j in range(4)])
            ys = np.array([Y[tr, j][M[tr, j]].std() or 1.0 for j in range(4)])
            lo_s, hi_s = np.nan_to_num((LO - ym) / ys), np.nan_to_num((HI - ym) / ys)
            # --- CYP2D6 transformed target, fit on this fold's training rows only ---
            d6tr = tr[M[tr, J2D6]]
            yt = Y[d6tr, J2D6]
            if leg == "rank":
                tgt = np.full(len(df), np.nan)
                tgt[d6tr] = (rankdata(yt) - 0.5) / len(yt)     # -> (0,1)
            else:
                edges = np.quantile(yt, np.linspace(0, 1, NBINS + 1)[1:-1])
                tgt = np.full((len(df), n_out), np.nan)
                tgt[d6tr] = (yt[:, None] > edges[None, :]).astype(float)
            qgrid = np.quantile(yt, np.linspace(0.001, 0.999, 999))   # rank -> pIC50 map
            torch.manual_seed(seed)
            net = Net(PF.shape[1], n_out)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            g = torch.Generator().manual_seed(seed); best = (-1e9, None, 0)
            m3 = M.copy(); m3[:, J2D6] = False        # CYP2D6 excluded from the hinge heads
            for ep in range(MAXEP):
                net.train(); perm = torch.randperm(len(tr), generator=g).numpy()
                for i in range(0, len(tr), BS):
                    b = tr[perm[i:i + BS]]
                    opt.zero_grad()
                    mu, d6 = net(bmg(b), torch.tensor(PF[b], dtype=torch.float32))
                    loss = interval_hinge(mu, torch.tensor(lo_s[b], dtype=torch.float32),
                                          torch.tensor(hi_s[b], dtype=torch.float32),
                                          torch.tensor(m3[b], dtype=torch.float32))
                    msk = torch.tensor(M[b, J2D6].astype(np.float32))
                    if float(msk.sum()):
                        if leg == "rank":
                            t = torch.tensor(np.nan_to_num(tgt[b]), dtype=torch.float32)
                            l2 = ((d6[:, 0] - t).abs() * msk).sum() / msk.sum()
                        else:
                            t = torch.tensor(np.nan_to_num(tgt[b]), dtype=torch.float32)
                            l2 = (bce(d6, t).mean(1) * msk).sum() / msk.sum()
                        loss = loss + l2
                    loss.backward(); opt.step()
                net.eval()
                # select on CYP2D6 Spearman (the metric these transforms target)
                with torch.no_grad():
                    _, d6v = net(bmg(iv), torch.tensor(PF[iv], dtype=torch.float32))
                s = d6v[:, 0].numpy() if leg == "rank" else torch.sigmoid(d6v).sum(1).numpy()
                mm = M[iv, J2D6]
                r = spearmanr(s[mm], Y[iv, J2D6][mm]).correlation if mm.sum() > 2 else -1e9
                if r > best[0] + 1e-4: best = (r, copy.deepcopy(net.state_dict()), ep)
                elif ep - best[2] >= PATIENCE: break
            net.load_state_dict(best[1]); net.eval()
            with torch.no_grad():
                _, d6v = net(bmg(va), torch.tensor(PF[va], dtype=torch.float32))
            s = d6v[:, 0].numpy() if leg == "rank" else torch.sigmoid(d6v).sum(1).numpy()
            sc[va] = s
            u = np.clip((rankdata(s) - 0.5) / len(s), 0.001, 0.999)
            pc[va] = np.interp(u, np.linspace(0.001, 0.999, 999), qgrid)
            print(f"  {leg} s{seed} fold{f}: best@{best[2]+1}ep (iv rho={best[0]:.3f})", flush=True)
        oof_score.append(sc); oof_pic50.append(pc)
    np.save(ROOT / f"experiments/e3_{leg}_score.npy", np.nanmean(oof_score, 0))
    np.save(ROOT / f"experiments/e3_{leg}_pic50.npy", np.nanmean(oof_pic50, 0))
    y = Y[:, J2D6]; m = (~np.isnan(y)) & (GFOLD >= 0)
    s = np.nanmean(oof_score, 0); p = np.nanmean(oof_pic50, 0)
    m = m & ~np.isnan(s)
    print(f"\n{leg}: CYP2D6 OOF Spearman = {spearmanr(s[m], y[m]).correlation:.4f}  (control 0.4421)")
    print(f"{leg}: CYP2D6 OOF ST-RAE (rank->pIC50 remap) = "
          f"{strae(p[m], LO[m, J2D6], HI[m, J2D6], y[m]):.4f}  (control 0.5681)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", required=True, choices=["rank", "ordinal"])
    run(ap.parse_args().leg)
