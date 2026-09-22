"""Two-head direct/shift model with ablation switches exposed from the outset.

Every design axis we want to study is a field on `TwoHeadConfig`, toggled
independently:

  (a) interval_targets   — interval-hinge targets vs point (weighted-L1) targets
  (b) shared_encoder     — one multi-task encoder vs a separate model per isoform
  (c) encoder            — "ecfp" (ECFP4+descriptors) vs "dmpnn" input features
  (d) tdi_mode           — "derived" label from the two arms vs a "classifier" head
  (e) width_pull         — the 1/(1+width) L1 pull toward the point (interval mode)
  (f) shift_prior        — L1 pull on delta toward 0

Evaluation uses the SAME scaffold folds as `cyp.baseline` (per isoform,
`scaffold_folds(seed=0)`), and OOF ST-RAE is always scored against the reported
credible interval `[conf_low, conf_high]` regardless of the training target — so
every number is directly comparable to the baseline column.

The intended first step is the config CLOSEST to the baseline (ECFP4 input,
per-isoform heads, point targets, no TDI arm, no width pull, shift_prior 0),
which must reproduce the baseline OOF ST-RAE within fold variance before any
switch is turned on. `validate_harness()` proves the folds and metric are
identical to the baseline independently of the model, so a gap can be attributed
to the learner rather than the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .features import featurize
from .losses import (
    DirectShiftHead,
    interval_hinge,
    st_rae as st_rae_torch,
    width_weighted_l1,
)
from .splits import scaffold_folds

torch.set_num_threads(1)  # determinism / reproducibility from a clean checkout

DATA_DIR = Path("data/cyp-challenge-train-test")
TRAIN_FILE = DATA_DIR / "cyp-challenge-TRAIN_TDI.csv"
BASELINE_OOF_FILE = Path("data/baseline_oof_predictions.csv")

ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
SMILES_COL = "SMILES"
SEED = 0
N_FOLDS = 5

# Baseline OOF ST-RAE (mean, fold std), from cyp.baseline / FINDINGS §6, on the
# identical per-isoform scaffold folds. Used as the comparison column.
BASELINE_STRAE = {
    "CYP1A2": (0.526, 0.017),
    "CYP2C9": (0.364, 0.019),
    "CYP2D6": (0.617, 0.012),
    "CYP3A4": (0.298, 0.019),
}


@dataclass
class TwoHeadConfig:
    # --- ablation switches (independent) ---
    interval_targets: bool = False      # (a) False = point weighted-L1, True = interval-hinge
    shared_encoder: bool = False        # (b) False = per-isoform, True = multi-task shared
    encoder: str = "ecfp"               # (c) "ecfp" | "dmpnn"
    tdi_mode: str = "derived"           # (d) "derived" | "classifier"
    width_pull: bool = False            # (e) 1/(1+width) L1 pull (interval mode)
    shift_prior: float = 0.0            # (f)
    use_tdi_arm: bool = False           # supervise mu+delta with the TDI arm (off = baseline-like)
    # --- training ---
    hidden: int = 512
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    seed: int = SEED

    def label(self) -> str:
        arm = "interval" if self.interval_targets else "point"
        enc = "shared" if self.shared_encoder else "per-iso"
        bits = [self.encoder, enc, arm]
        if self.use_tdi_arm:
            bits.append("tdi-arm")
        if self.width_pull:
            bits.append("wpull")
        if self.shift_prior:
            bits.append(f"sp{self.shift_prior:g}")
        return "+".join(bits)


def direct_col(iso):
    return f"{iso}_pIC50_direct_inhibition"


def _bounds(df, iso):
    lo = df[f"{iso}_pIC50_direct_inhibition_conf_low"].to_numpy(float)
    hi = df[f"{iso}_pIC50_direct_inhibition_conf_high"].to_numpy(float)
    return lo, hi


def _standardize(train_X, *others, clip=10.0):
    """Standardize on train stats, then clip to +/- `clip` sigma.

    ECFP counts and unbounded RDKit descriptors (e.g. Ipc ~1e14, rare bits)
    produce extreme standardized outliers (>70 sigma) that destabilize an MLP;
    LightGBM is scale-invariant and ignores them. Clipping tames the spikes
    without dropping features."""
    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0)
    sd[sd == 0] = 1.0
    def _z(X):
        return np.clip((X - mu) / sd, -clip, clip).astype(np.float32)
    return [_z(train_X), *[_z(X) for X in others]]


def _st_rae(pred, lo, hi, truth):
    return st_rae_torch(*(torch.tensor(a, dtype=torch.float64) for a in (pred, lo, hi, truth)))


def _std1(v):
    v = np.asarray(v, float)
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def _build_encoder(cfg: TwoHeadConfig, in_dim: int, n_iso: int) -> nn.Module:
    if cfg.encoder == "ecfp":
        return DirectShiftHead(in_dim, n_iso, cfg.hidden)
    if cfg.encoder == "dmpnn":
        raise NotImplementedError(
            "D-MPNN encoder switch is exposed but not yet wired; use encoder='ecfp'."
        )
    raise ValueError(f"unknown encoder {cfg.encoder!r}")


def _point_or_interval_loss(pred, target, lo, hi, weight, mask, cfg: TwoHeadConfig):
    """Regression term for one arm. `mask`/`weight`/`target`/`lo`/`hi` are (N,K)."""
    if cfg.interval_targets:
        loss = interval_hinge(pred, lo, hi, mask)
        if cfg.width_pull:
            loss = loss + width_weighted_l1(pred, target, hi - lo, mask)
        return loss
    # point: weighted L1 to the point estimate (weight = 1/(1+width), as in baseline)
    per = (pred - target).abs() * weight
    per = torch.where(mask.bool(), per, torch.zeros_like(per))
    return per.sum() / mask.float().sum().clamp(min=1.0)


def _train(cfg, Xtr, targets, in_dim, n_iso):
    """targets: dict of (N,n_iso) tensors: direct_y, direct_lo, direct_hi,
    direct_w, direct_mask, and optionally tdi_* when cfg.use_tdi_arm."""
    if cfg.tdi_mode == "classifier":
        raise NotImplementedError(
            "TDI classifier head is exposed but not yet wired; use tdi_mode='derived'."
        )
    torch.manual_seed(cfg.seed)
    model = _build_encoder(cfg, in_dim, n_iso)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        mu, delta = model(Xtr_t)
        loss = _point_or_interval_loss(
            mu, targets["direct_y"], targets["direct_lo"], targets["direct_hi"],
            targets["direct_w"], targets["direct_mask"], cfg,
        )
        if cfg.use_tdi_arm and "tdi_mask" in targets:
            loss = loss + _point_or_interval_loss(
                mu + delta, targets["tdi_y"], targets["tdi_lo"], targets["tdi_hi"],
                targets["tdi_w"], targets["tdi_mask"], cfg,
            )
        if cfg.shift_prior > 0:
            loss = loss + cfg.shift_prior * delta.abs().mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


def _predict_mu(model, X):
    with torch.no_grad():
        mu, _ = model(torch.tensor(X, dtype=torch.float32))
    return mu.numpy()


def run_per_isoform(cfg, df, X, ok):
    """Per-isoform independent models. Returns {iso: dict(strae_mean/std, pred_std,
    fold_id, idx, oof)}."""
    results = {}
    for j, iso in enumerate(ISOFORMS):
        y_all = df[direct_col(iso)].to_numpy(float)
        present = ~np.isnan(y_all) & ok
        idx = np.where(present)[0]
        smiles = df[SMILES_COL].to_numpy()[idx]
        fold_id, _ = scaffold_folds(smiles.tolist(), N_FOLDS, cfg.seed)
        y = y_all[idx]
        lo_all, hi_all = _bounds(df, iso)
        lo, hi = lo_all[idx], hi_all[idx]
        width = hi - lo
        width = np.where(np.isnan(width), np.nanmedian(width), width)
        weight = 1.0 / (1.0 + width)
        Xs = X[idx]

        # optional TDI arm for this isoform
        tdi_y = df[f"{iso}_pIC50_TDI_condition"].to_numpy(float)[idx]

        oof = np.full(len(idx), np.nan)
        fold_strae = []
        for f in range(N_FOLDS):
            va = fold_id == f
            tr = ~va
            Xtr, Xva = _standardize(Xs[tr], Xs[va])
            targets = {
                "direct_y": torch.tensor(y[tr, None], dtype=torch.float32),
                "direct_lo": torch.tensor(lo[tr, None], dtype=torch.float32),
                "direct_hi": torch.tensor(hi[tr, None], dtype=torch.float32),
                "direct_w": torch.tensor(weight[tr, None], dtype=torch.float32),
                "direct_mask": torch.ones((tr.sum(), 1), dtype=torch.bool),
            }
            if cfg.use_tdi_arm:
                tw = df[f"{iso}_pIC50_TDI_condition_conf_high"].to_numpy(float)[idx] \
                    - df[f"{iso}_pIC50_TDI_condition_conf_low"].to_numpy(float)[idx]
                tmask = ~np.isnan(tdi_y)
                targets.update({
                    "tdi_y": torch.tensor(np.nan_to_num(tdi_y[tr, None]), dtype=torch.float32),
                    "tdi_lo": torch.tensor(df[f"{iso}_pIC50_TDI_condition_conf_low"].to_numpy(float)[idx][tr, None], dtype=torch.float32),
                    "tdi_hi": torch.tensor(df[f"{iso}_pIC50_TDI_condition_conf_high"].to_numpy(float)[idx][tr, None], dtype=torch.float32),
                    "tdi_w": torch.tensor((1.0 / (1.0 + np.nan_to_num(tw, nan=np.nanmedian(tw))))[tr, None], dtype=torch.float32),
                    "tdi_mask": torch.tensor(tmask[tr, None], dtype=torch.bool),
                })
            model = _train(cfg, Xtr, targets, Xtr.shape[1], 1)
            pred = _predict_mu(model, Xva)[:, 0]
            oof[va] = pred
            fold_strae.append(_st_rae(pred, lo[va], hi[va], y[va]))

        results[iso] = {
            "strae_mean": float(np.mean(fold_strae)),
            "strae_std": _std1(fold_strae),
            "pred_std": _std1(oof),
            "fold_id": fold_id,
            "idx": idx,
            "oof": oof,
        }
    return results


def run_config(cfg, df=None, X=None, ok=None):
    if df is None:
        df = pd.read_csv(TRAIN_FILE)
    if X is None:
        X, ok, _ = featurize(df[SMILES_COL].tolist())
    if cfg.shared_encoder:
        raise NotImplementedError(
            "shared multi-task encoder is exposed but not yet wired; use shared_encoder=False."
        )
    return run_per_isoform(cfg, df, X, ok)


def validate_harness():
    """Prove folds and metric are identical to the baseline, independent of the
    model. Returns True if both checks pass."""
    if not BASELINE_OOF_FILE.exists():
        print(f"  (skip) {BASELINE_OOF_FILE} missing; run `make baseline` first.")
        return None
    base = pd.read_csv(BASELINE_OOF_FILE)
    df = pd.read_csv(TRAIN_FILE)
    ok_folds, ok_metric = True, True
    print("Harness validation (folds + ST-RAE metric vs baseline):")
    for iso in ISOFORMS:
        y_all = df[direct_col(iso)].to_numpy(float)
        idx = np.where(~np.isnan(y_all))[0]
        smiles = df[SMILES_COL].to_numpy()[idx]
        fold_here, _ = scaffold_folds(smiles.tolist(), N_FOLDS, SEED)
        fold_base = base[f"{iso}_reg_fold"].to_numpy()[idx].astype(int)
        same = bool(np.array_equal(fold_here, fold_base))
        ok_folds &= same
        # recompute per-fold ST-RAE from the baseline's saved OOF predictions
        lo_all, hi_all = _bounds(df, iso)
        oof = base[f"{iso}_pIC50_oof"].to_numpy(float)[idx]
        srae = [_st_rae(oof[fold_here == f], lo_all[idx][fold_here == f],
                        hi_all[idx][fold_here == f], y_all[idx][fold_here == f])
                for f in range(N_FOLDS)]
        rec_mean, _ = BASELINE_STRAE[iso]
        match = abs(np.mean(srae) - rec_mean) < 0.01
        ok_metric &= match
        print(f"  {iso}: folds identical={same}  recomputed ST-RAE={np.mean(srae):.3f} "
              f"(recorded {rec_mean:.3f}) match={match}")
    print(f"  => folds {'OK' if ok_folds else 'MISMATCH'}, metric {'OK' if ok_metric else 'MISMATCH'}")
    return ok_folds and ok_metric


def _print_comparison(title, results):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'isoform':8} {'twohead ST-RAE':>20} {'baseline ST-RAE':>18} {'pred std':>9} {'within fold var?':>16}")
    for iso in ISOFORMS:
        r = results[iso]
        b_mean, b_std = BASELINE_STRAE[iso]
        tol = max(r["strae_std"], b_std)
        within = abs(r["strae_mean"] - b_mean) <= tol
        print(f"{iso:8} {r['strae_mean']:8.3f} +/- {r['strae_std']:5.3f}    "
              f"{b_mean:8.3f} +/- {b_std:5.3f}   {r['pred_std']:8.3f}   {str(within):>16}")


def main():
    if not TRAIN_FILE.exists():
        print(f"Missing {TRAIN_FILE}. Run `make data` first.")
        return 1

    validate_harness()

    df = pd.read_csv(TRAIN_FILE)
    print("\nfeaturizing (ECFP4 + descriptors)...")
    X, ok, _ = featurize(df[SMILES_COL].tolist())

    cfg = TwoHeadConfig()  # defaults = closest to baseline
    print(f"\nconfig closest to baseline: {cfg.label()}")
    results = run_config(cfg, df, X, ok)
    _print_comparison("TWO-HEAD (baseline-closest config) vs BASELINE", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
