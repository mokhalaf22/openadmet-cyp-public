"""Phase 2: encoder warm start + external auxiliary heads, four legs for attribution.

    python experiments/phase2_train.py --leg {baseline,warmstart,octant,tox21,combined}

Legs
  baseline   D-MPNN + predicted-primary, from scratch  (reproduces 0.6037 / 0.4146)
  warmstart  + encoder pretrained on the 111,361-compound corpus (physchem only)
  octant     + Octant CYP3A4 auxiliary head (separate head, never merged)
  tox21      + Tox21 CYP auxiliary heads (separate heads, never merged)
  combined   warmstart + octant + tox21

LEAKAGE-SAFE DEVIATION FROM THE BRIEF (deliberate, documented).
The brief puts *assay* heads in the pretraining stage alongside the down-weighted
physicochemical head. Done once and shared across folds, that leaks: the Octant CYP3A4
readout is a different condition but is strongly correlated with the scored CYP3A4 pIC50, so
pretraining on a compound that later sits in a validation fold contaminates that fold. Doing
it correctly would need per-fold pretraining over ~117k molecules (15x), which is not
affordable. So:
  * pretraining uses the physicochemical head ONLY -- label-free, hence fold-independent and
    legitimately shared across all folds and seeds;
  * every assay head (challenge DRC + Octant + Tox21) lives in the fine-tune, masked to that
    fold's TRAINING rows only.
This preserves both questions the gate asks (does corpus coverage help? does external
supervision help?) without contaminating the folds.

No measured label is ever attached to a retrieved neighbour: the corpus carries computed
descriptors only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from scipy.stats import spearmanr

from cyp.losses import interval_hinge, st_rae as st_rae_torch

RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(10)
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/cyp-challenge-train-test"
ISOS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
NF, DH, MAXEP, PATIENCE, BS = 5, 200, 300, 20, 512
PRE_EPOCHS, PRE_BS, PHYSCHEM_W = 6, 256, 0.1
TRAJ_EPOCHS = 10  # inner-val trajectory logged for the first N fine-tune epochs
ENC_CKPT = ROOT / "experiments" / "phase2_encoder.pt"

DESCS = [Descriptors.MolWt, Crippen.MolLogP, rdMolDescriptors.CalcTPSA,
         rdMolDescriptors.CalcNumHBD, rdMolDescriptors.CalcNumHBA,
         rdMolDescriptors.CalcNumRotatableBonds, rdMolDescriptors.CalcNumAromaticRings,
         rdMolDescriptors.CalcFractionCSP3]

from chemprop.data import BatchMolGraph  # noqa: E402
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer  # noqa: E402
from chemprop.nn import BondMessagePassing, MeanAggregation  # noqa: E402

FEAT = SimpleMoleculeMolGraphFeaturizer()


def strae(p, lo, hi, y):
    return float(st_rae_torch(*(torch.tensor(a, dtype=torch.float64) for a in (p, lo, hi, y))))


# ------------------------------------------------------------------ data ---
def challenge():
    df = pd.read_csv(D / "cyp-challenge-TRAIN_TDI.csv")
    z = np.load(ROOT / "experiments/plog_gfold.npz")
    Y = np.stack([df[f"{i}_pIC50_direct_inhibition"].to_numpy(float) for i in ISOS], 1)
    LO = np.stack([df[f"{i}_pIC50_direct_inhibition_conf_low"].to_numpy(float) for i in ISOS], 1)
    HI = np.stack([df[f"{i}_pIC50_direct_inhibition_conf_high"].to_numpy(float) for i in ISOS], 1)
    return df, Y, LO, HI, ~np.isnan(Y), z["GFOLD"], z["oko"], z["plog_o"]


def aux_targets(df, leg: str) -> tuple[np.ndarray, list[str]]:
    """External auxiliary targets aligned to challenge rows (NaN where absent).

    Each external source is its OWN column/head -- never merged into a scored column and
    never rescaled onto the challenge pIC50 scale.
    """
    from rdkit.Chem import inchi

    def ik(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return inchi.MolToInchiKey(m) if m else None

    cols, names = [], []
    row_ik = [ik(s) for s in df["SMILES"]]
    if leg in ("octant", "combined"):
        p = ROOT / "data/external/octant_cyp3a4_auxhead.csv"
        m = dict(zip(pd.read_csv(p)["ik"], pd.read_csv(p)["CYP3A4_pIC50"]))
        cols.append(np.array([m.get(k, np.nan) for k in row_ik], float))
        names.append("octant_CYP3A4_combined_condition")
    if leg in ("tox21", "combined"):
        p = ROOT / "data/external/tox21_cyp_auxhead.csv"
        if p.exists():
            t = pd.read_csv(p)
            for c in [c for c in t.columns if c.startswith("tox21_")]:
                m = dict(zip(t["ik"], t[c]))
                cols.append(np.array([m.get(k, np.nan) for k in row_ik], float))
                names.append(c)
        else:
            print("  (tox21 aux file absent -- leg contributes no heads)", flush=True)
    A = np.stack(cols, 1) if cols else np.zeros((len(df), 0))
    return A, names


def external_rows(df, leg: str, aux_names: list[str]) -> tuple[list[str], np.ndarray]:
    """External-source compounds absent from the challenge set, as extra training rows.

    Returns their SMILES and an aux-target matrix aligned to `aux_names`. Challenge test
    compounds and the 6 quarantined overlaps are excluded upstream (Octant) or by InChIKey
    here, so no blinded structure can enter training.
    """
    from rdkit.Chem import inchi

    def ik(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return inchi.MolToInchiKey(m) if m else None

    if not aux_names:
        return [], np.zeros((0, 0))
    te = pd.read_csv(D / "cyp-challenge-TEST-BLINDED.csv")
    blocked = {ik(s) for s in te["SMILES"]} | {ik(s) for s in df["SMILES"]}
    rows: dict[str, dict] = {}
    srcs = []
    if leg in ("octant", "combined"):
        srcs.append((ROOT / "data/external/octant_cyp3a4_auxhead.csv",
                     "standardized_smiles", {"octant_CYP3A4_combined_condition": "CYP3A4_pIC50"}))
    if leg in ("tox21", "combined"):
        p = ROOT / "data/external/tox21_cyp_auxhead.csv"
        if p.exists():
            t = pd.read_csv(p)
            srcs.append((p, None, {c: c for c in t.columns if c.startswith("tox21_")}))
    for path, smi_col, mapping in srcs:
        t = pd.read_csv(path)
        if smi_col is None:  # tox21 has no SMILES column; recover from CID
            cid2smi = _tox21_smiles(t)
            t = t.assign(_smi=[cid2smi.get(int(c)) for c in t["cid"]])
            smi_col = "_smi"
        for _, r in t.iterrows():
            s = r.get(smi_col)
            if not isinstance(s, str):
                continue
            k = r.get("ik") or ik(s)
            if k in blocked or k is None:
                continue
            rec = rows.setdefault(k, {"smiles": s})
            for head, src_col in mapping.items():
                if pd.notna(r.get(src_col)):
                    rec[head] = float(r[src_col])
    smis = [v["smiles"] for v in rows.values()]
    A = np.array([[v.get(h, np.nan) for h in aux_names] for v in rows.values()], float)
    return smis, A if len(smis) else np.zeros((0, len(aux_names)))


def _tox21_smiles(t: pd.DataFrame) -> dict[int, str]:
    """Tox21 aux file stores CIDs; resolve to SMILES via the cached fetch output."""
    cache = ROOT / "data/external/tox21_cid_smiles.csv"
    if cache.exists():
        c = pd.read_csv(cache)
        return dict(zip(c["cid"].astype(int), c["smiles"]))
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tox21_fetch import cid_smiles  # reuse the POST-batched resolver

    m = cid_smiles([int(c) for c in t["cid"]])
    pd.DataFrame({"cid": list(m), "smiles": [m[k] for k in m]}).to_csv(cache, index=False)
    return m


# -------------------------------------------------------------- pretrain ---
def pretrain_encoder(corpus_smiles: list[str], extra_smiles: list[str]) -> dict:
    """Physchem-only pretraining over corpus + challenge structures (label-free).

    Graphs are featurized per batch rather than cached: 111k cached MolGraphs would not fit
    comfortably in 24 GB.
    """
    if ENC_CKPT.exists():
        print(f"encoder checkpoint exists -> {ENC_CKPT.name}", flush=True)
        return torch.load(ENC_CKPT)
    smis = [s for s in corpus_smiles + extra_smiles if isinstance(s, str)]
    print(f"pretraining physchem head on {len(smis)} structures "
          f"({len(corpus_smiles)} corpus + {len(extra_smiles)} challenge)", flush=True)

    class Pre(nn.Module):
        def __init__(s):
            super().__init__()
            s.mp = BondMessagePassing(d_h=DH, depth=3)
            s.agg = MeanAggregation()
            s.head = nn.Sequential(nn.Linear(DH, 128), nn.GELU(), nn.Linear(128, len(DESCS)))

        def forward(s, b):
            return s.head(s.agg(s.mp(b), b.batch))

    torch.manual_seed(0)
    net = Pre()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    # target standardization from a sample (full pass is unnecessary and slow)
    rng = np.random.RandomState(0)
    samp = [smis[i] for i in rng.choice(len(smis), min(5000, len(smis)), replace=False)]
    P = np.array([[f(Chem.MolFromSmiles(s)) for f in DESCS] for s in samp
                  if Chem.MolFromSmiles(s)], float)
    pm, ps = P.mean(0), P.std(0) + 1e-8

    idx = np.arange(len(smis))
    for ep in range(PRE_EPOCHS):
        net.train()
        rng.shuffle(idx)
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(idx), PRE_BS):
            chunk = [smis[k] for k in idx[i:i + PRE_BS]]
            mols = [Chem.MolFromSmiles(s) for s in chunk]
            pairs = [(m, [f(m) for f in DESCS]) for m in mols if m is not None]
            if len(pairs) < 2:
                continue
            bmg = BatchMolGraph([FEAT(m) for m, _ in pairs])
            tgt = torch.tensor((np.array([v for _, v in pairs], float) - pm) / ps,
                               dtype=torch.float32)
            opt.zero_grad()
            loss = PHYSCHEM_W * ((net(bmg) - tgt) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
            if nb % 100 == 0:
                print(f"  ep{ep+1} batch {nb}: loss {tot/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
        print(f"  ep{ep+1}/{PRE_EPOCHS}: physchem loss {tot/max(nb,1):.4f} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
    torch.save(net.mp.state_dict(), ENC_CKPT)
    print(f"saved {ENC_CKPT.name}", flush=True)
    return net.mp.state_dict()


# ------------------------------------------------------------- fine-tune ---
class Net(nn.Module):
    def __init__(self, nfeat: int, n_aux: int, enc_sd=None, hidden=256, drop=0.1):
        super().__init__()
        self.mp = BondMessagePassing(d_h=DH, depth=3)
        if enc_sd is not None:
            self.mp.load_state_dict(enc_sd)
        self.agg = MeanAggregation()
        self.trunk = nn.Sequential(nn.Linear(DH + nfeat, hidden), nn.GELU(), nn.Dropout(drop))
        self.mu = nn.Linear(hidden, 4)
        self.aux = nn.Linear(hidden, n_aux) if n_aux else None

    def forward(self, b, pf):
        h = self.trunk(torch.cat([self.agg(self.mp(b), b.batch), pf], 1))
        return self.mu(h), (self.aux(h) if self.aux is not None else None)


def run_leg(leg: str) -> dict:
    df, Y, LO, HI, M, GFOLD, oko, PLOG = challenge()
    valid = np.where(oko)[0]
    MG = {int(i): FEAT(Chem.MolFromSmiles(df["SMILES"].iloc[int(i)])) for i in valid}

    def bmg(rows):
        return BatchMolGraph([MG[int(r)] for r in rows])

    pm, ps = np.nanmean(PLOG[valid], 0), np.nanstd(PLOG[valid], 0)
    PF = np.nan_to_num((PLOG - np.where(np.isnan(pm), 0, pm)) / np.where(ps < 1e-6, 1, ps))

    A, aux_names = aux_targets(df, leg)
    # External compounds that do NOT overlap the challenge set enter as EXTRA TRAINING ROWS
    # with the challenge heads masked off and only their own aux head supervised. Without
    # this, Tox21 would contribute just 210 of its 7,879 compounds (it is distant chemistry),
    # and a null result would measure our wiring rather than the data.
    ext_smiles, ext_A = external_rows(df, leg, aux_names)
    n_ch = len(df)
    if len(ext_smiles):
        ext_graphs = {}
        for k, s in enumerate(ext_smiles):
            m = Chem.MolFromSmiles(s)
            if m is not None:
                ext_graphs[n_ch + k] = FEAT(m)
        MG.update(ext_graphs)
        ok_ext = np.array([(n_ch + k) in MG for k in range(len(ext_smiles))])
        ext_A = ext_A[ok_ext]
        ext_idx = np.array([n_ch + k for k in range(len(ext_smiles)) if (n_ch + k) in MG])
        # pad challenge arrays so external rows exist but are unsupervised for scored heads
        pad = len(ext_idx)
        Y = np.vstack([Y, np.full((pad, 4), np.nan)])
        LO = np.vstack([LO, np.full((pad, 4), np.nan)])
        HI = np.vstack([HI, np.full((pad, 4), np.nan)])
        M = np.vstack([M, np.zeros((pad, 4), bool)])
        PF = np.vstack([PF, np.zeros((pad, PF.shape[1]))])  # 0 == fold-train mean once standardized
        A = np.vstack([A, ext_A]) if A.shape[1] else A
        GFOLD = np.concatenate([GFOLD, np.full(pad, -2)])  # -2: external, never a validation fold
    else:
        ext_idx = np.array([], dtype=int)
    print(f"leg={leg} | aux heads: {aux_names or 'none'} | external rows: {len(ext_idx)}", flush=True)

    enc_sd = None
    if leg in ("warmstart", "combined", "ws_lowlr", "ws_freeze"):
        corpus = pd.read_csv(ROOT / "experiments/corpus_phase1.csv")["smiles"].tolist()
        enc_sd = pretrain_encoder(corpus, df["SMILES"].tolist())

    traj = {}
    oofs = []
    for seed in (0, 1, 2):
        npy = ROOT / f"experiments/p2_{leg}_s{seed}.npy"
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
            if len(ext_idx):
                tr = np.concatenate([tr, ext_idx])  # external rows train, never validate
            va = np.where(GFOLD == f)[0]
            ym = np.array([Y[tr, j][M[tr, j]].mean() for j in range(4)])
            ys = np.array([Y[tr, j][M[tr, j]].std() or 1.0 for j in range(4)])
            lo, hi = np.nan_to_num((LO - ym) / ys), np.nan_to_num((HI - ym) / ys)
            # aux targets standardized on this fold's TRAIN rows only, masked elsewhere
            if A.shape[1]:
                am = np.nanmean(A[tr], 0)
                asd = np.nanstd(A[tr], 0)
                asd = np.where((asd < 1e-6) | np.isnan(asd), 1.0, asd)
                Az = (A - np.where(np.isnan(am), 0, am)) / asd
                AM = ~np.isnan(A)
                AM[va] = False  # never supervise on validation-fold rows
            torch.manual_seed(seed)
            net = Net(PF.shape[1], A.shape[1], enc_sd=enc_sd)
            # Gate-2b configurations: hold the pretrained encoder in place so the corpus
            # representation can survive past epoch 1 (§40a showed it is erased by epoch 2).
            HEAD_LR, ENC_LR, FREEZE_EP = 1e-3, 1e-4, 5
            if leg in ("ws_lowlr", "ws_freeze"):
                enc_p = list(net.mp.parameters())
                enc_ids = {id(p) for p in enc_p}
                head_p = [p for p in net.parameters() if id(p) not in enc_ids]
                if leg == "ws_freeze":
                    net.mp.requires_grad_(False)  # genuinely frozen, not lr=0
                opt = torch.optim.Adam(
                    [{"params": head_p, "lr": HEAD_LR},
                     {"params": enc_p, "lr": ENC_LR}], weight_decay=1e-4)
            else:
                opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            g = torch.Generator().manual_seed(seed)
            best, key = (1e9, None, 0), f"s{seed}f{f}"
            for ep in range(MAXEP):
                if leg == "ws_freeze" and ep == FREEZE_EP:
                    net.mp.requires_grad_(True)  # unfreeze at the reduced encoder rate
                net.train()
                perm = torch.randperm(len(tr), generator=g).numpy()
                for i in range(0, len(tr), BS):
                    b = tr[perm[i:i + BS]]
                    opt.zero_grad()
                    mu, aux = net(bmg(b), torch.tensor(PF[b], dtype=torch.float32))
                    loss = interval_hinge(mu,
                                          torch.tensor(lo[b], dtype=torch.float32),
                                          torch.tensor(hi[b], dtype=torch.float32),
                                          torch.tensor(M[b], dtype=torch.float32))
                    if aux is not None and A.shape[1]:
                        amask = torch.tensor(AM[b], dtype=torch.float32)
                        at = torch.tensor(np.nan_to_num(Az[b]), dtype=torch.float32)
                        if float(amask.sum()):
                            loss = loss + 0.3 * (((aux - at) ** 2) * amask).sum() / amask.sum()
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    mv = net(bmg(iv), torch.tensor(PF[iv], dtype=torch.float32))[0].numpy() * ys + ym
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
                oof[va] = net(bmg(va), torch.tensor(PF[va], dtype=torch.float32))[0].numpy() * ys + ym
            print(f"  {leg} s{seed} fold{f}: best@{best[2]+1}ep", flush=True)
        np.save(npy, oof)
        oofs.append(oof)

    ens = np.nanmean(oofs, 0)
    np.save(ROOT / f"experiments/p2_{leg}_oof.npy", ens)
    if traj:
        (ROOT / f"experiments/p2_{leg}_traj.json").write_text(json.dumps(traj, indent=1))
    return {"leg": leg, "aux_heads": aux_names}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", required=True,
                    choices=["baseline", "warmstart", "octant", "tox21", "combined", "ws_lowlr", "ws_freeze"])
    run_leg(ap.parse_args().leg)
