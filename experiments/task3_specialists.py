"""Task 3: CYP2D6 specialists averaged INTO the shared model (not substituted for it).

    python experiments/task3_specialists.py --leg {spec_all,spec_drc}

(a) spec_all  -- every CYP2D6 readout, nothing from the other isoforms: direct pIC50
                 (interval-hinge, the scored target), TDI-arm pIC50 (interval, own head),
                 Emax both arms (own heads), single-concentration log2FC (own head).
(b) spec_drc  -- CYP2D6 direct pIC50 alone, single head.

§45 showed per-isoform models are worse than the shared one, so these are NOT replacements;
they are ensemble members. The published comparison is 0.445 for the best single member vs
0.503 for a four-member average, i.e. the gain is expected from averaging, not from any
specialist winning outright.

Leakage rule: a fold's validation rows are excluded from training ENTIRELY, whichever readout
they carry. Every auxiliary readout of a compound is correlated with its scored label, so
supervising any head on a validation-fold compound would contaminate that fold.

Same GFOLD folds, 3 seeds, same optimizer/features as the Phase-2 control.
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
from rdkit.Chem import inchi

from cyp.losses import interval_hinge, st_rae as st_rae_torch

RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(10)
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/cyp-challenge-train-test"
ISO = "CYP2D6"
NF, DH, MAXEP, PATIENCE, BS = 5, 200, 300, 20, 512
W_AUX = 0.3

from chemprop.data import BatchMolGraph  # noqa: E402
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer  # noqa: E402
from chemprop.nn import BondMessagePassing, MeanAggregation  # noqa: E402

FEAT = SimpleMoleculeMolGraphFeaturizer()


def strae(p, lo, hi, y):
    return float(st_rae_torch(*(torch.tensor(a, dtype=torch.float64) for a in (p, lo, hi, y))))


def load():
    df = pd.read_csv(D / "cyp-challenge-TRAIN_TDI.csv")
    emx = pd.read_csv(D / "cyp-challenge-TRAIN_Emax.csv")
    sc = pd.read_csv(D / "cyp-challenge-single-concentration-TRAIN.csv")
    z = np.load(ROOT / "experiments/plog_gfold.npz")
    emx = df[["Molecule_Name"]].merge(emx, on="Molecule_Name", how="left")

    def ik(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return inchi.MolToInchiKey(m) if m else None

    scp = sc[sc["enzyme"] == ISO].groupby("SMILES")["log2fc_estimate"].mean()
    smap = {ik(s): v for s, v in scp.items() if ik(s)}
    g = {
        "df": df, "GFOLD": z["GFOLD"], "oko": z["oko"], "PLOG": z["plog_o"],
        "y": df[f"{ISO}_pIC50_direct_inhibition"].to_numpy(float),
        "lo": df[f"{ISO}_pIC50_direct_inhibition_conf_low"].to_numpy(float),
        "hi": df[f"{ISO}_pIC50_direct_inhibition_conf_high"].to_numpy(float),
        "tlo": df[f"{ISO}_pIC50_TDI_condition_conf_low"].to_numpy(float),
        "thi": df[f"{ISO}_pIC50_TDI_condition_conf_high"].to_numpy(float),
        "ty": df[f"{ISO}_pIC50_TDI_condition"].to_numpy(float),
        "emax_d": emx[f"{ISO}_EmaxVsPosCtrl_direct_inhibition"].to_numpy(float),
        "emax_t": emx[f"{ISO}_EmaxVsPosCtrl_TDI_condition"].to_numpy(float),
        "log2fc": np.array([smap.get(ik(s), np.nan) for s in df["SMILES"]], float),
    }
    return g


class Net(nn.Module):
    def __init__(self, nfeat: int, n_aux: int, hidden=256, drop=0.1):
        super().__init__()
        self.mp = BondMessagePassing(d_h=DH, depth=3)
        self.agg = MeanAggregation()
        self.trunk = nn.Sequential(nn.Linear(DH + nfeat, hidden), nn.GELU(), nn.Dropout(drop))
        self.mu = nn.Linear(hidden, 1)                       # scored: CYP2D6 direct pIC50
        self.tdi = nn.Linear(hidden, 1) if n_aux else None    # TDI-arm interval
        self.pts = nn.Linear(hidden, 3) if n_aux else None    # emax_d, emax_t, log2fc
    def forward(self, b, pf):
        h = self.trunk(torch.cat([self.agg(self.mp(b), b.batch), pf], 1))
        return (self.mu(h),
                self.tdi(h) if self.tdi is not None else None,
                self.pts(h) if self.pts is not None else None)


def run(leg: str) -> None:
    g = load()
    df, GFOLD, oko = g["df"], g["GFOLD"], g["oko"]
    y, lo, hi = g["y"], g["lo"], g["hi"]
    M = ~np.isnan(y)
    allrd = leg == "spec_all"
    PTS = np.stack([g["emax_d"], g["emax_t"], g["log2fc"]], 1)
    TM = ~np.isnan(g["ty"])
    PM = ~np.isnan(PTS)
    # rows usable at all: any CYP2D6 readout (spec_all) or a direct label (spec_drc)
    usable = oko & (M | (TM | PM.any(1)) if allrd else M)
    valid = np.where(oko)[0]
    MG = {int(i): FEAT(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}
    def bmg(rows): return BatchMolGraph([MG[int(r)] for r in rows])
    pmn, psd = np.nanmean(g["PLOG"][valid], 0), np.nanstd(g["PLOG"][valid], 0)
    PF = np.nan_to_num((g["PLOG"] - np.where(np.isnan(pmn), 0, pmn)) / np.where(psd < 1e-6, 1, psd))
    print(f"leg={leg} | usable rows={int(usable.sum())} | with direct label={int((M&oko).sum())}", flush=True)

    oofs = []
    for seed in (0, 1, 2):
        npy = ROOT / f"experiments/t3_{leg}_s{seed}.npy"
        if npy.exists():
            print(f"  s{seed}: cached", flush=True); oofs.append(np.load(npy)); continue
        oof = np.full(len(df), np.nan)
        for f in range(NF):
            # validation-fold rows are excluded from training ENTIRELY (leakage rule)
            outer = np.where(usable & (GFOLD != f) & (GFOLD >= 0))[0]
            rng = np.random.RandomState(seed * 100 + f)
            iv = outer[rng.permutation(len(outer))[:int(len(outer) * 0.15)]]
            tr = np.setdiff1d(outer, iv)
            va = np.where((GFOLD == f) & M)[0]
            if not len(tr) or not len(va): continue
            ym, ys = y[tr][M[tr]].mean(), (y[tr][M[tr]].std() or 1.0)
            lo_s, hi_s = np.nan_to_num((lo - ym) / ys), np.nan_to_num((hi - ym) / ys)
            tlo_s, thi_s = np.nan_to_num((g["tlo"] - ym) / ys), np.nan_to_num((g["thi"] - ym) / ys)
            pm2, ps2 = np.nanmean(PTS[tr], 0), np.nanstd(PTS[tr], 0)
            ps2 = np.where((ps2 < 1e-6) | np.isnan(ps2), 1.0, ps2)
            PTS_s = np.nan_to_num((PTS - np.where(np.isnan(pm2), 0, pm2)) / ps2)
            torch.manual_seed(seed)
            net = Net(PF.shape[1], 1 if allrd else 0)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            gg = torch.Generator().manual_seed(seed); best = (1e9, None, 0)
            for ep in range(MAXEP):
                net.train(); perm = torch.randperm(len(tr), generator=gg).numpy()
                for i in range(0, len(tr), BS):
                    b = tr[perm[i:i + BS]]
                    opt.zero_grad()
                    mu, tdi, pts = net(bmg(b), torch.tensor(PF[b], dtype=torch.float32))
                    loss = interval_hinge(mu[:, 0],
                                          torch.tensor(lo_s[b], dtype=torch.float32),
                                          torch.tensor(hi_s[b], dtype=torch.float32),
                                          torch.tensor(M[b], dtype=torch.float32))
                    if allrd:
                        loss = loss + W_AUX * interval_hinge(
                            tdi[:, 0], torch.tensor(tlo_s[b], dtype=torch.float32),
                            torch.tensor(thi_s[b], dtype=torch.float32),
                            torch.tensor(TM[b], dtype=torch.float32))
                        msk = torch.tensor(PM[b], dtype=torch.float32)
                        if float(msk.sum()):
                            tgt = torch.tensor(PTS_s[b], dtype=torch.float32)
                            loss = loss + W_AUX * (((pts - tgt) ** 2) * msk).sum() / msk.sum()
                    loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    mv = net(bmg(iv), torch.tensor(PF[iv], dtype=torch.float32))[0].numpy()[:, 0] * ys + ym
                mm = M[iv]
                ivs = strae(mv[mm], lo[iv][mm], hi[iv][mm], y[iv][mm]) if mm.sum() else 1e9
                if ivs < best[0] - 1e-4: best = (ivs, copy.deepcopy(net.state_dict()), ep)
                elif ep - best[2] >= PATIENCE: break
            net.load_state_dict(best[1]); net.eval()
            with torch.no_grad():
                oof[va] = net(bmg(va), torch.tensor(PF[va], dtype=torch.float32))[0].numpy()[:, 0] * ys + ym
            print(f"  {leg} s{seed} fold{f}: best@{best[2]+1}ep", flush=True)
        np.save(npy, oof); oofs.append(oof)
    np.save(ROOT / f"experiments/t3_{leg}_oof.npy", np.nanmean(oofs, 0))
    print(f"saved experiments/t3_{leg}_oof.npy", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", required=True, choices=["spec_all", "spec_drc"])
    run(ap.parse_args().leg)
