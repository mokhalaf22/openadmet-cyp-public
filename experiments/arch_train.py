"""Architecture tests: does the TDI head cost regression, and does isoform sharing still help?

    python experiments/arch_train.py --leg {allheads,per_isoform}

PREMISE CORRECTION (see FINDINGS §45). The question "does the TDI head cost the regression
side?" was posed as removing heads from the control. But the control is ALREADY
regression-only: its network has the four `mu` interval heads and nothing else supervised
(the reference `dmpnn_primary` carries a `dr` module whose output never enters the loss, so
it receives no gradient). There is nothing to strip. The test therefore runs the other way:

  (a) allheads     regression interval heads + a supervised delta head (TDI-arm intervals,
                   mu+delta) + a TDI classifier head, vs the regression-only control. If this
                   is worse on regression, the §20 "sharing helps classification" benefit is
                   paid for by the regression side -- and shipping regression from a
                   regression-only model (which is already what we do) is confirmed correct.

  (b) per_isoform  four single-task models, each its own encoder and one interval head,
                   trained only on that isoform's rows. Re-tests the -0.032 macro benefit of
                   shared multi-task heads, which was measured on the ECFP control before the
                   D-MPNN encoder and the predicted-primary feature existed.

Folds, seeds, optimizer and features match the Phase-2 baseline leg, so
experiments/p2_baseline_oof.npy (0.6059 / 0.4131) is a valid in-run control.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cyp.guards import tdi_trainable_mask  # noqa: E402
from cyp.losses import interval_hinge, st_rae as st_rae_torch  # noqa: E402

RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(10)
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/cyp-challenge-train-test"
ISOS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
NF, DH, MAXEP, PATIENCE, BS = 5, 200, 300, 20, 512
W_TDI, W_CLF = 1.0, 0.3  # delta/TDI-arm and classifier loss weights

from chemprop.data import BatchMolGraph  # noqa: E402
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer  # noqa: E402
from chemprop.nn import BondMessagePassing, MeanAggregation  # noqa: E402

FEAT = SimpleMoleculeMolGraphFeaturizer()


def strae(p, lo, hi, y):
    return float(st_rae_torch(*(torch.tensor(a, dtype=torch.float64) for a in (p, lo, hi, y))))


class Net(nn.Module):
    """n_out interval heads, plus optional supervised delta and classifier heads."""

    def __init__(self, nfeat: int, n_out: int = 4, heads: bool = False, hidden=256, drop=0.1):
        super().__init__()
        self.mp = BondMessagePassing(d_h=DH, depth=3)
        self.agg = MeanAggregation()
        self.trunk = nn.Sequential(nn.Linear(DH + nfeat, hidden), nn.GELU(), nn.Dropout(drop))
        self.mu = nn.Linear(hidden, n_out)
        self.dr = nn.Linear(hidden, n_out) if heads else None
        self.clf = nn.Linear(hidden, n_out) if heads else None

    def forward(self, b, pf):
        h = self.trunk(torch.cat([self.agg(self.mp(b), b.batch), pf], 1))
        delta = nn.functional.softplus(self.dr(h)) if self.dr is not None else None
        logit = self.clf(h) if self.clf is not None else None
        return self.mu(h), delta, logit


def load():
    df = pd.read_csv(D / "cyp-challenge-TRAIN_TDI.csv")
    z = np.load(ROOT / "experiments/plog_gfold.npz")
    g = {
        "df": df, "GFOLD": z["GFOLD"], "oko": z["oko"], "PLOG": z["plog_o"],
        "Y": np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS], 1),
        "LO": np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS], 1),
        "HI": np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS], 1),
    }
    # TDI arm (for the delta head) and the boolean label (for the classifier head)
    g["TLO"] = np.stack([df[f"{i}_pIC50_TDI_condition_conf_low"].to_numpy(float) for i in ISOS], 1)
    g["THI"] = np.stack([df[f"{i}_pIC50_TDI_condition_conf_high"].to_numpy(float) for i in ISOS], 1)
    g["TY"] = np.stack([df[f"{i}_pIC50_TDI_condition"].to_numpy(float) for i in ISOS], 1)
    g["CLF"] = np.stack([df[f"{i}_is_TDI"].to_numpy() for i in ISOS], 1).astype(float)
    # guards: never supervise TDI on direct-arm-never-assayed rows
    g["CLFM"] = np.stack([tdi_trainable_mask(df, i).to_numpy() for i in ISOS], 1)
    g["M"] = ~np.isnan(g["Y"])
    g["TM"] = ~np.isnan(g["TY"])
    return g


def run(leg: str) -> None:
    g = load()
    df, GFOLD, oko, PLOG = g["df"], g["GFOLD"], g["oko"], g["PLOG"]
    Y, LO, HI, M = g["Y"], g["LO"], g["HI"], g["M"]
    valid = np.where(oko)[0]
    MG = {int(i): FEAT(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}

    def bmg(rows):
        return BatchMolGraph([MG[int(r)] for r in rows])

    pmn, psd = np.nanmean(PLOG[valid], 0), np.nanstd(PLOG[valid], 0)
    PF = np.nan_to_num((PLOG - np.where(np.isnan(pmn), 0, pmn)) / np.where(psd < 1e-6, 1, psd))
    bce = nn.BCEWithLogitsLoss(reduction="none")

    oofs = []
    for seed in (0, 1, 2):
        npy = ROOT / f"experiments/arch_{leg}_s{seed}.npy"
        if npy.exists():
            print(f"  s{seed}: cached", flush=True)
            oofs.append(np.load(npy))
            continue
        oof = np.full((len(df), 4), np.nan)
        # per_isoform: one single-task model per isoform; allheads: one multi-task model
        iso_sets = [[j] for j in range(4)] if leg == "per_isoform" else [[0, 1, 2, 3]]
        for cols in iso_sets:
            for f in range(NF):
                present = M[:, cols].any(1)
                outer = np.where((GFOLD != f) & (GFOLD >= 0) & present)[0]
                rng = np.random.RandomState(seed * 100 + f)
                iv = outer[rng.permutation(len(outer))[:int(len(outer) * 0.15)]]
                tr = np.setdiff1d(outer, iv)
                va = np.where((GFOLD == f) & present)[0]
                if not len(tr) or not len(va):
                    continue
                ym = np.array([Y[tr, j][M[tr, j]].mean() if M[tr, j].any() else 0.0 for j in cols])
                ys = np.array([Y[tr, j][M[tr, j]].std() or 1.0 for j in cols])
                lo = np.nan_to_num((LO[:, cols] - ym) / ys)
                hi = np.nan_to_num((HI[:, cols] - ym) / ys)
                tlo = np.nan_to_num((g["TLO"][:, cols] - ym) / ys)
                thi = np.nan_to_num((g["THI"][:, cols] - ym) / ys)
                torch.manual_seed(seed)
                net = Net(PF.shape[1], n_out=len(cols), heads=(leg == "allheads"))
                opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
                gg = torch.Generator().manual_seed(seed)
                best = (1e9, None, 0)
                for ep in range(MAXEP):
                    net.train()
                    perm = torch.randperm(len(tr), generator=gg).numpy()
                    for i in range(0, len(tr), BS):
                        b = tr[perm[i:i + BS]]
                        opt.zero_grad()
                        mu, delta, logit = net(bmg(b), torch.tensor(PF[b], dtype=torch.float32))
                        loss = interval_hinge(mu,
                                              torch.tensor(lo[b], dtype=torch.float32),
                                              torch.tensor(hi[b], dtype=torch.float32),
                                              torch.tensor(M[b][:, cols], dtype=torch.float32))
                        if leg == "allheads":
                            # TDI-arm pIC50 is mu+delta by construction
                            loss = loss + W_TDI * interval_hinge(
                                mu + delta,
                                torch.tensor(tlo[b], dtype=torch.float32),
                                torch.tensor(thi[b], dtype=torch.float32),
                                torch.tensor(g["TM"][b][:, cols], dtype=torch.float32))
                            cm = torch.tensor(g["CLFM"][b][:, cols], dtype=torch.float32)
                            if float(cm.sum()):
                                ct = torch.tensor(np.nan_to_num(g["CLF"][b][:, cols]),
                                                  dtype=torch.float32)
                                loss = loss + W_CLF * ((bce(logit, ct) * cm).sum() / cm.sum())
                        loss.backward()
                        opt.step()
                    net.eval()
                    with torch.no_grad():
                        mv = net(bmg(iv), torch.tensor(PF[iv], dtype=torch.float32))[0].numpy() * ys + ym
                    vals = [strae(mv[M[iv, j], k], LO[iv, j][M[iv, j]], HI[iv, j][M[iv, j]],
                                  Y[iv, j][M[iv, j]])
                            for k, j in enumerate(cols) if M[iv, j].sum()]
                    ivs = float(np.mean(vals)) if vals else 1e9
                    if ivs < best[0] - 1e-4:
                        best = (ivs, copy.deepcopy(net.state_dict()), ep)
                    elif ep - best[2] >= PATIENCE:
                        break
                net.load_state_dict(best[1])
                net.eval()
                with torch.no_grad():
                    pv = net(bmg(va), torch.tensor(PF[va], dtype=torch.float32))[0].numpy() * ys + ym
                for k, j in enumerate(cols):
                    oof[va, j] = pv[:, k]
                tag = ISOS[cols[0]] if leg == "per_isoform" else "shared"
                print(f"  {leg} {tag} s{seed} fold{f}: best@{best[2]+1}ep", flush=True)
        np.save(npy, oof)
        oofs.append(oof)
    np.save(ROOT / f"experiments/arch_{leg}_oof.npy", np.nanmean(oofs, 0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", required=True, choices=["allheads", "per_isoform"])
    run(ap.parse_args().leg)
