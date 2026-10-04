"""Phase 3: multi-fidelity and metric-aligned training. Two legs, final experiment.

    python experiments/phase3_train.py --leg {proxy,ci_mc}

(a) proxy  -- extra DOWN-WEIGHTED supervision from the single-concentration screen.
    Retargeted from the brief: there are no screen-only *molecules* (all 4,376 screen
    compounds are already in the DRC table), but the DRC matrix is sparse, so 11,505
    (compound, isoform) CELLS have a measured log2FC and no DRC pIC50. Those get per-fold
    proxy targets (experiments/proxy_targets.py; held-out r 0.76-0.93) and enter as point
    targets at weight 0.3. Distinct from the predicted-primary FEATURE (§25): this is
    additional supervision, not an input column.

(b) ci_mc  -- credible-interval Monte Carlo. Each epoch, draw the target uniformly from
    within each compound's reported [lo, hi] and fit L1 to that draw, instead of the
    interval hinge. Tests whether sampling inside the interval beats scoring distance to
    its nearest bound.

Architecture, folds, seeds, optimizer and feature handling are identical to the Phase-2
baseline leg, so experiments/p2_baseline_oof.npy (0.6059 / 0.4131) is a valid in-run control.
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
PROXY_W = 0.3
TRAJ_EPOCHS = 10

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


def run(leg: str) -> None:
    df = pd.read_csv(D / "cyp-challenge-TRAIN_TDI.csv")
    z = np.load(ROOT / "experiments/plog_gfold.npz")
    GFOLD, oko, PLOG = z["GFOLD"], z["oko"], z["plog_o"]
    Y = np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS], 1)
    LO = np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS], 1)
    HI = np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS], 1)
    M = ~np.isnan(Y)
    valid = np.where(oko)[0]
    MG = {int(i): FEAT(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}

    def bmg(rows):
        return BatchMolGraph([MG[int(r)] for r in rows])

    pmn, psd = np.nanmean(PLOG[valid], 0), np.nanstd(PLOG[valid], 0)
    PF = np.nan_to_num((PLOG - np.where(np.isnan(pmn), 0, pmn)) / np.where(psd < 1e-6, 1, psd))

    PX = FILL = None
    if leg == "proxy":
        pz = np.load(ROOT / "experiments/proxy_targets.npz")
        PX, FILL = pz["proxy"], pz["fillable"]
        print(f"proxy cells: {int(FILL.sum())} at weight {PROXY_W}", flush=True)

    traj: dict[str, list[float]] = {}
    oofs = []
    for seed in (0, 1, 2):
        npy = ROOT / f"experiments/p3_{leg}_s{seed}.npy"
        if npy.exists():
            print(f"  s{seed}: cached", flush=True)
            oofs.append(np.load(npy))
            continue
        oof = np.full((len(df), 4), np.nan)
        for f in range(NF):
            outer = np.where((GFOLD != f) & (GFOLD >= 0))[0]
            rng = np.random.RandomState(seed * 100 + f)
            iv = outer[rng.permutation(len(outer))[:int(len(outer) * 0.15)]]
            tr = np.setdiff1d(outer, iv)
            va = np.where(GFOLD == f)[0]
            ym = np.array([Y[tr, j][M[tr, j]].mean() for j in range(4)])
            ys = np.array([Y[tr, j][M[tr, j]].std() or 1.0 for j in range(4)])

            lo, hi = LO.copy(), HI.copy()
            W = M.astype(np.float32)
            if leg == "proxy":  # point targets on proxy-able cells, down-weighted
                pf_ = PX[:, :, f]
                use = FILL & ~np.isnan(pf_)
                lo = np.where(use, pf_, lo)
                hi = np.where(use, pf_, hi)
                W = np.where(use, PROXY_W, W).astype(np.float32)
            lo_s, hi_s = np.nan_to_num((lo - ym) / ys), np.nan_to_num((hi - ym) / ys)
            tgt_s = np.nan_to_num((Y - ym) / ys)

            torch.manual_seed(seed)
            net = Net(PF.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            g = torch.Generator().manual_seed(seed)
            drawer = np.random.RandomState(seed * 977 + f)
            best, key = (1e9, None, 0), f"s{seed}f{f}"
            for ep in range(MAXEP):
                if leg == "ci_mc":  # one fresh draw inside [lo, hi] per epoch
                    u = drawer.uniform(size=lo_s.shape)
                    draw = lo_s + u * np.clip(hi_s - lo_s, 0, None)
                    draw = np.where(M, draw, tgt_s)
                net.train()
                perm = torch.randperm(len(tr), generator=g).numpy()
                for i in range(0, len(tr), BS):
                    b = tr[perm[i:i + BS]]
                    opt.zero_grad()
                    mu = net(bmg(b), torch.tensor(PF[b], dtype=torch.float32))
                    if leg == "ci_mc":
                        d = torch.tensor(draw[b], dtype=torch.float32)
                        loss = interval_hinge(mu, d, d, torch.tensor(W[b]))  # == weighted L1
                    else:
                        loss = interval_hinge(mu,
                                              torch.tensor(lo_s[b], dtype=torch.float32),
                                              torch.tensor(hi_s[b], dtype=torch.float32),
                                              torch.tensor(W[b]))
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    mv = net(bmg(iv), torch.tensor(PF[iv], dtype=torch.float32)).numpy() * ys + ym
                vals = [strae(mv[M[iv, j], j], LO[iv, j][M[iv, j]], HI[iv, j][M[iv, j]],
                              Y[iv, j][M[iv, j]]) for j in range(4) if M[iv, j].sum()]
                ivs = float(np.mean(vals))
                if ep < TRAJ_EPOCHS:
                    traj.setdefault(key, []).append(round(ivs, 5))
                if ivs < best[0] - 1e-4:
                    best = (ivs, copy.deepcopy(net.state_dict()), ep)
                elif ep - best[2] >= PATIENCE:
                    break
            net.load_state_dict(best[1])
            net.eval()
            with torch.no_grad():
                oof[va] = net(bmg(va), torch.tensor(PF[va], dtype=torch.float32)).numpy() * ys + ym
            print(f"  {leg} s{seed} fold{f}: best@{best[2]+1}ep", flush=True)
        np.save(npy, oof)
        oofs.append(oof)

    np.save(ROOT / f"experiments/p3_{leg}_oof.npy", np.nanmean(oofs, 0))
    if traj:
        import json
        (ROOT / f"experiments/p3_{leg}_traj.json").write_text(json.dumps(traj, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", required=True, choices=["proxy", "ci_mc"])
    run(ap.parse_args().leg)
