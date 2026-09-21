"""Interval-hinge losses and the two-head direct/shift formulation.

Central idea: every training target is an INTERVAL, never a point. A fitted DRC
gives a pIC50 point estimate plus a Bayesian credible interval; the competition
scores distance to the nearest bound of that interval, with zero error inside
it. So train against the interval.

The credible interval already encodes left-censoring (a compound below the
lowest tested dose is reported with `conf_low` on a low floor ~1.03 and a wide
interval). We therefore consume the reported `conf_low` / `conf_high` VERBATIM —
there is no separate `(-inf, 4.0]` case and no assumed limit of quantitation.
See CLAUDE.md "Confirmed data conventions" and FINDINGS.md for the evidence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# These are the organizers' TDI-label thresholds (a fact about the scoring
# definition), NOT a censoring limit. pIC50 = 4 is the assay's reliable lower
# bound; a >2-fold shift is log10(2).
TDI_MU_THRESHOLD = 4.0        # direct-arm pIC50 boundary in the label rule
LOG2 = 0.30103                # log10(2) — the 2-fold shift threshold
TDI_INFERRED = 4.30103        # TDI_MU_THRESHOLD + LOG2


def interval_hinge(pred, lo, hi, mask=None, reduction="mean"):
    """L1 distance to the nearest interval bound; zero inside the interval.

    Args:
        pred: (N,) or (N, T) predictions
        lo:   lower bounds; may be -inf for a left-open target
        hi:   upper bounds; may be +inf for a right-open target
        mask: bool tensor, True where a target is observed. Missing targets
              contribute nothing — the training matrix is sparse and that
              sparsity is structural, not something to impute. NaN bounds on
              masked rows are absorbed safely.

    This is exactly the competition's soft-threshold error, so the training
    objective and the leaderboard metric agree. That is the point.
    """
    # Sanitize NaN bounds (missing targets on masked rows) to a finite 0 BEFORE
    # any arithmetic. A NaN bound would forward to 0 via nan_to_num but leave a
    # NaN in the backward graph, and 0 * NaN = NaN poisons mu's gradient even
    # for masked rows. Infinities (legitimately open bounds) are preserved.
    lo = torch.nan_to_num(lo, nan=0.0, posinf=float("inf"), neginf=float("-inf"))
    hi = torch.nan_to_num(hi, nan=0.0, posinf=float("inf"), neginf=float("-inf"))
    below = torch.clamp(lo - pred, min=0.0)
    above = torch.clamp(pred - hi, min=0.0)
    # A degenerate lo=+inf / hi=-inf would give +inf; treat as no contribution.
    loss = torch.nan_to_num(below, nan=0.0, posinf=0.0) + \
           torch.nan_to_num(above, nan=0.0, posinf=0.0)

    if mask is not None:
        loss = loss * mask.float()
        if reduction == "mean":
            n = mask.float().sum().clamp(min=1.0)
            return loss.sum() / n
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def width_weighted_l1(pred, point, width, mask=None):
    """A gentle L1 pull toward the reported point estimate, down-weighted by
    interval width as ``1 / (1 + width)``.

    The hinge is flat inside the interval, so a wide interval yields no hinge
    gradient at all once the prediction lands inside it. This term keeps a
    small, non-zero signal on those rows — 23.6% of CYP3A4 direct rows are wider
    than 2 log units — while trusting the point estimate LESS as the interval
    widens (a wide interval means the assay barely knows the value).

    NaN point/width on masked-out rows are selected away before any arithmetic
    can propagate them.
    """
    # Sanitize NaN point/width (missing on masked rows) before arithmetic, for
    # the same reason as interval_hinge: 0 * NaN = NaN would poison the gradient.
    point = torch.nan_to_num(point, nan=0.0)
    width = torch.nan_to_num(width, nan=0.0)
    w = 1.0 / (1.0 + width)
    per_row = (pred - point).abs() * w
    if mask is not None:
        per_row = torch.where(mask.bool(), per_row, torch.zeros_like(per_row))
        n = mask.float().sum().clamp(min=1.0)
        return per_row.sum() / n
    return per_row.mean()


def make_intervals(point, lo, hi, widen=0.0):
    """Build (lo, hi) bound tensors from a DRC table — no special cases.

    Reported `conf_low` / `conf_high` are used verbatim. Rows with no reported
    bound fall back to the point estimate, which makes the hinge degenerate to
    plain L1 for those rows — correct behaviour. `widen` adds optional symmetric
    slack (default 0.0); only widen if OOF shows the model is punished for
    landing inside experimental noise.

    The result is guaranteed ordered (`lo <= hi`) so downstream code never sees
    an inverted interval, even if a source row has malformed bounds.
    """
    lo = torch.where(torch.isnan(lo), point, lo) - widen
    hi = torch.where(torch.isnan(hi), point, hi) + widen
    ordered_lo = torch.minimum(lo, hi)
    ordered_hi = torch.maximum(lo, hi)
    return ordered_lo, ordered_hi


class DirectShiftHead(nn.Module):
    """Per-isoform head predicting direct pIC50 and a non-negative shift.

    The TDI arm is mu + delta by construction. Turnover during the +NADPH
    preincubation can only convert a compound into a MORE potent inhibitor, so
    delta >= 0 is a fact about the assay, not a regularizer — enforce it in the
    architecture rather than hoping the model learns it.
    """

    def __init__(self, in_dim: int, n_isoforms: int, hidden: int = 256):
        super().__init__()
        self.n_isoforms = n_isoforms
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden, n_isoforms)
        self.delta_raw = nn.Linear(hidden, n_isoforms)

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu(h)
        delta = F.softplus(self.delta_raw(h))      # >= 0, smooth at the boundary
        return mu, delta


def tdi_label_from_arms(mu, delta, hard: bool = True, temperature: float = 0.1):
    """Derive the boolean TDI label from the two arms.

    The organizers' definition (mu = direct pIC50, mu + delta = TDI-arm pIC50):
      mu >= 4  ->  positive iff delta > log10(2)
      mu <  4  ->  positive iff mu + delta > 4.301   ("inferred positive")
      both arms below 4 -> negative ("assigned negative")

    hard=True returns booleans for submission. hard=False returns a
    differentiable relaxation so the MCC-relevant decision can be nudged during
    training if you choose to add an auxiliary classification term.
    """
    tdi_arm = mu + delta
    if hard:
        high = (mu >= TDI_MU_THRESHOLD) & (delta > LOG2)
        low = (mu < TDI_MU_THRESHOLD) & (tdi_arm > TDI_INFERRED)
        return high | low

    s_high = torch.sigmoid((delta - LOG2) / temperature)
    s_low = torch.sigmoid((tdi_arm - TDI_INFERRED) / temperature)
    gate = torch.sigmoid((mu - TDI_MU_THRESHOLD) / temperature)
    return gate * s_high + (1.0 - gate) * s_low


def _active_mask(direct_mask, tdi_mask, like):
    active = None
    for m in (direct_mask, tdi_mask):
        if m is None:
            continue
        active = m.bool() if active is None else (active | m.bool())
    if active is None:
        active = torch.ones_like(like, dtype=torch.bool)
    return active


def two_head_loss(mu, delta, direct_lo, direct_hi, direct_mask,
                  tdi_lo=None, tdi_hi=None, tdi_mask=None,
                  direct_point=None, tdi_point=None,
                  shift_prior: float = 5e-3, width_pull: float = 1.0):
    """Joint objective over both arms.

    Both arms are supervised as intervals. The shared trunk means the plentiful
    direct-arm data regularizes the sparser shift estimate, and vice versa.

    Row sets and what supervises what:
      - Fitted-DRC rows: `direct_mask` supervises mu against [conf_low, conf_high].
      - Rows with a measured TDI arm: `tdi_mask` supervises mu + delta.
      - Direct-less rows (TDI arm present, direct arm NEVER assayed): set
        `direct_mask = False` for them so mu carries no direct-arm target. mu is
        then supervised ONLY through mu + delta. No extra upper bound on mu is
        needed or justified: softplus makes delta >= 0, so mu <= mu + delta
        already — the TDI arm bounds mu from above for free. Supervising mu
        toward a censored value here would fabricate a shift on compounds that
        are frequently potent (see FINDINGS.md §4).

    direct_point / tdi_point: reported point estimates. When given, add the
    1/(1+width) L1 pull (see `width_weighted_l1`) so wide-interval rows still
    receive gradient.

    shift_prior: small L1 penalty pulling delta toward zero, applied on every
    row with any observed arm. It RESOLVES THE mu/delta SPLIT on direct-less
    rows: with mu unpinned, pulling delta -> 0 pushes mu -> the (potent) TDI
    arm, i.e. "the potency is direct unless the data says otherwise", which
    matches the assigned-negative labels far better than assuming inactivity.
    But a larger value biases the SCORED direct head upward on ~35% of CYP3A4
    rows, so this is a tuned hyperparameter, not a free choice. Start at 5e-3.
    """
    loss = interval_hinge(mu, direct_lo, direct_hi, direct_mask)
    if direct_point is not None:
        width = direct_hi - direct_lo
        loss = loss + width_pull * width_weighted_l1(mu, direct_point, width, direct_mask)

    if tdi_lo is not None:
        loss = loss + interval_hinge(mu + delta, tdi_lo, tdi_hi, tdi_mask)
        if tdi_point is not None:
            width = tdi_hi - tdi_lo
            loss = loss + width_pull * width_weighted_l1(mu + delta, tdi_point, width, tdi_mask)

    if shift_prior > 0:
        active = _active_mask(direct_mask, tdi_mask, delta)
        pulled = torch.where(active, delta.abs(), torch.zeros_like(delta))
        loss = loss + shift_prior * (pulled.sum() / active.float().sum().clamp(min=1.0))

    return loss


@torch.no_grad()
def st_rae(pred, lo, hi, truth, mask=None):
    """Soft-Threshold Relative Absolute Error — the leaderboard metric.

    Relative to the predict-the-mean baseline, so ~1.0 means the model adds
    nothing over a constant. Always report this on scaffold-split OOF
    predictions; raw MAE improvements routinely vanish here.
    """
    err = interval_hinge(pred, lo, hi, mask, reduction="none")
    if mask is not None:
        truth = truth[mask]
        err = err[mask]
    denom = (truth - truth.mean()).abs().sum().clamp(min=1e-8)
    return (err.sum() / denom).item()
